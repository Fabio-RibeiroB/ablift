import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from statistics import NormalDist

import numpy as np
from click.testing import CliRunner
from openpyxl import Workbook
from statsmodels.stats.power import NormalIndPower

from ablift.cli import cli
from ablift.connectors import build_duration_request_from_rows, read_table
from ablift.engine import msprt_statistic
from ablift.planning import (
    bayesian_duration_arpu,
    bayesian_duration_conversion,
    fixed_horizon_duration_arpu,
    fixed_horizon_duration_conversion,
    sequential_duration_arpu,
    sequential_duration_conversion,
)


class ArpuPlanningTests(unittest.TestCase):
    def test_fixed_horizon_matches_statsmodels_and_textbook(self):
        cases = [
            (10.0, 20.0, 0.05, 50_000, 2, 0.05, 0.8),
            (25.0, 40.0, 0.08, 100_000, 3, 0.05, 0.8),
            (12.5, 30.0, -0.10, 40_000, 4, 0.01, 0.9),
            (-15.0, 20.0, 0.10, 20_000, 2, 0.05, 0.8),
        ]
        nd = NormalDist()
        power_calc = NormalIndPower()
        for mean, sd, mde, traffic, n_variants, alpha, power in cases:
            with self.subTest(mean=mean, sd=sd, mde=mde, n_variants=n_variants):
                plan = fixed_horizon_duration_arpu(
                    baseline_mean=mean,
                    baseline_sd=sd,
                    relative_mde=mde,
                    daily_total_traffic=traffic,
                    n_variants=n_variants,
                    alpha=alpha,
                    power=power,
                )
                delta = abs(mean * mde)
                alpha_adj = alpha / (n_variants - 1)
                expected_textbook = math.ceil(
                    2.0
                    * (sd**2)
                    * ((nd.inv_cdf(1 - alpha_adj / 2) + nd.inv_cdf(power)) ** 2)
                    / (delta**2)
                )
                expected_statsmodels = math.ceil(
                    float(
                        power_calc.solve_power(
                            effect_size=delta / sd,
                            alpha=alpha_adj,
                            power=power,
                            ratio=1.0,
                            alternative="two-sided",
                        )
                    )
                )
                self.assertEqual(plan.method, "fixed_horizon")
                self.assertEqual(plan.n_per_variant, expected_textbook)
                self.assertAlmostEqual(plan.n_per_variant, expected_statsmodels, delta=1)
                self.assertAlmostEqual(
                    plan.estimated_days, plan.n_per_variant / (traffic / n_variants)
                )
                self.assertEqual(plan.assumptions["metric"], "arpu")
                self.assertAlmostEqual(plan.diagnostics["alpha_per_comparison"], alpha_adj)
                self.assertAlmostEqual(plan.diagnostics["absolute_delta"], delta)

    def test_baseline_cv_matches_equivalent_sd(self):
        mean, cv, mde, traffic = 12.5, 2.0, 0.05, 50_000
        sd = abs(mean) * cv

        fixed_sd = fixed_horizon_duration_arpu(mean, mde, traffic, baseline_sd=sd)
        fixed_cv = fixed_horizon_duration_arpu(mean, mde, traffic, baseline_cv=cv)
        self.assertEqual(fixed_sd.n_per_variant, fixed_cv.n_per_variant)
        self.assertAlmostEqual(fixed_sd.estimated_days, fixed_cv.estimated_days)
        self.assertEqual(fixed_cv.assumptions["baseline_cv"], cv)
        self.assertAlmostEqual(fixed_cv.assumptions["baseline_sd"], sd)
        self.assertNotIn("baseline_cv", fixed_sd.assumptions)

        seq_sd = sequential_duration_arpu(mean, mde, traffic, baseline_sd=sd)
        seq_cv = sequential_duration_arpu(mean, mde, traffic, baseline_cv=cv)
        self.assertEqual(seq_sd.n_per_variant, seq_cv.n_per_variant)
        self.assertEqual(seq_cv.assumptions["baseline_cv"], cv)

        bayes_sd = bayesian_duration_arpu(
            mean, mde, traffic, baseline_sd=sd, max_days=5, sims=40, posterior_draws=300
        )
        bayes_cv = bayesian_duration_arpu(
            mean, mde, traffic, baseline_cv=cv, max_days=5, sims=40, posterior_draws=300
        )
        self.assertEqual(bayes_sd.estimated_days, bayes_cv.estimated_days)
        self.assertEqual(bayes_sd.n_per_variant, bayes_cv.n_per_variant)
        self.assertEqual(bayes_cv.assumptions["baseline_cv"], cv)

    def test_sequential_plan_achieves_target_power(self):
        baseline_mean, baseline_sd, mde, power = 10.0, 20.0, 0.15, 0.8
        plan = sequential_duration_arpu(
            baseline_mean=baseline_mean,
            baseline_sd=baseline_sd,
            relative_mde=mde,
            daily_total_traffic=10_000,
            power=power,
        )
        n = plan.n_per_variant
        assert n is not None
        tau = plan.assumptions["sequential_tau"]
        self.assertAlmostEqual(tau, baseline_mean * 0.1)

        rng = np.random.default_rng(1)
        sims = 2000
        mu_t = baseline_mean * (1.0 + mde)
        # Simulate exact sample means and sample variances for normal ARPU draws.
        c_means = rng.normal(baseline_mean, baseline_sd / math.sqrt(n), size=sims)
        t_means = rng.normal(mu_t, baseline_sd / math.sqrt(n), size=sims)
        c_vars = (baseline_sd**2) * rng.chisquare(n - 1, size=sims) / (n - 1)
        t_vars = (baseline_sd**2) * rng.chisquare(n - 1, size=sims) / (n - 1)

        hits = 0
        for cm, tm, cv_i, tv_i in zip(c_means, t_means, c_vars, t_vars, strict=True):
            se = math.sqrt((float(cv_i) + float(tv_i)) / n)
            hits += msprt_statistic(float(tm - cm), se, tau) >= 20
        self.assertAlmostEqual(hits / sims, power, delta=0.04)
        self.assertGreater(n, plan.diagnostics["fixed_horizon_n_per_variant"])
        self.assertGreater(plan.diagnostics["sequential_inflation"], 1.0)
        self.assertEqual(plan.assumptions["metric"], "arpu")

    def test_sequential_tau_default_and_override(self):
        tiny = sequential_duration_arpu(1e-8, 0.5, 10_000, baseline_sd=1e-8)
        self.assertEqual(tiny.assumptions["sequential_tau"], 1e-6)

        default = sequential_duration_arpu(10.0, 0.10, 20_000, baseline_sd=20.0)
        same = sequential_duration_arpu(10.0, 0.10, 20_000, baseline_sd=20.0, sequential_tau=1.0)
        wide = sequential_duration_arpu(10.0, 0.10, 20_000, baseline_sd=20.0, sequential_tau=100.0)
        self.assertAlmostEqual(default.assumptions["sequential_tau"], 1.0)
        self.assertEqual(default.n_per_variant, same.n_per_variant)
        self.assertGreater(wide.n_per_variant, default.n_per_variant)

    def test_bayesian_arpu_determinism_assurance_and_speed(self):
        kwargs = {
            "baseline_mean": 10.0,
            "baseline_sd": 15.0,
            "relative_mde": 0.10,
            "daily_total_traffic": 20_000,
            "max_expected_loss": 0.05,
            "max_days": 15,
            "sims": 120,
            "posterior_draws": 600,
            "seed": 42,
        }
        a = bayesian_duration_arpu(**kwargs)
        b = bayesian_duration_arpu(**kwargs)
        self.assertEqual(a.estimated_days, b.estimated_days)
        self.assertEqual(a.n_per_variant, b.n_per_variant)
        self.assertEqual(
            a.diagnostics["assurance_at_selected_day"],
            b.diagnostics["assurance_at_selected_day"],
        )
        self.assertIsNotNone(a.estimated_days)
        self.assertGreaterEqual(a.diagnostics["assurance_at_selected_day"], 0.8)
        self.assertEqual(a.assumptions["metric"], "arpu")

        # Huge n should still finish quickly thanks to O(1) sufficient-statistic sampling.
        t0 = time.monotonic()
        huge = bayesian_duration_arpu(
            baseline_mean=10.0,
            baseline_sd=20.0,
            relative_mde=0.05,
            daily_total_traffic=10_000_000,
            max_expected_loss=0.05,
            max_days=3,
            sims=60,
            posterior_draws=500,
            seed=7,
        )
        elapsed = time.monotonic() - t0
        self.assertIsNotNone(huge.estimated_days)
        self.assertLess(elapsed, 5.0)

        # Unreachable horizon returns None cleanly.
        none_plan = bayesian_duration_arpu(
            baseline_mean=10.0,
            baseline_sd=50.0,
            relative_mde=0.01,
            daily_total_traffic=200,
            max_days=2,
            sims=30,
            posterior_draws=200,
        )
        self.assertIsNone(none_plan.estimated_days)
        self.assertIsNone(none_plan.n_per_variant)
        self.assertIsNone(none_plan.diagnostics["assurance_at_selected_day"])

        zero_days = bayesian_duration_arpu(
            baseline_mean=10.0,
            baseline_sd=15.0,
            relative_mde=0.1,
            daily_total_traffic=10_000,
            max_days=0,
            sims=10,
        )
        self.assertIsNone(zero_days.estimated_days)

    def test_conversion_plans_include_metric_assumption(self):
        self.assertEqual(
            fixed_horizon_duration_conversion(0.04, 0.05, 50_000).assumptions["metric"],
            "conversion_rate",
        )
        self.assertEqual(
            sequential_duration_conversion(0.04, 0.05, 50_000).assumptions["metric"],
            "conversion_rate",
        )
        self.assertEqual(
            bayesian_duration_conversion(
                0.04, 0.05, 50_000, max_days=1, sims=10, posterior_draws=100
            ).assumptions["metric"],
            "conversion_rate",
        )

    def test_invalid_inputs_raise(self):
        for fn in (fixed_horizon_duration_arpu, sequential_duration_arpu, bayesian_duration_arpu):
            with self.subTest(fn=fn.__name__):
                with self.assertRaisesRegex(ValueError, "baseline_mean"):
                    fn(0.0, 0.05, 10_000, baseline_sd=5.0)
                with self.assertRaisesRegex(ValueError, "relative_mde"):
                    fn(10.0, 0.0, 10_000, baseline_sd=5.0)
                with self.assertRaisesRegex(ValueError, "not both"):
                    fn(10.0, 0.05, 10_000, baseline_sd=5.0, baseline_cv=0.5)
                with self.assertRaisesRegex(ValueError, "baseline_sd or baseline_cv"):
                    fn(10.0, 0.05, 10_000)
                with self.assertRaisesRegex(ValueError, "baseline_sd"):
                    fn(10.0, 0.05, 10_000, baseline_sd=0.0)
                with self.assertRaisesRegex(ValueError, "baseline_sd"):
                    fn(10.0, 0.05, 10_000, baseline_sd=-2.0)
                with self.assertRaisesRegex(ValueError, "baseline_cv"):
                    fn(10.0, 0.05, 10_000, baseline_cv=0.0)
                with self.assertRaisesRegex(ValueError, "baseline_cv"):
                    fn(10.0, 0.05, 10_000, baseline_cv=-1.0)
                with self.assertRaisesRegex(ValueError, "daily_total_traffic"):
                    fn(10.0, 0.05, 0, baseline_sd=5.0)
                with self.assertRaisesRegex(ValueError, "n_variants"):
                    fn(10.0, 0.05, 10_000, baseline_sd=5.0, n_variants=1)

        with self.assertRaisesRegex(ValueError, "sequential_tau"):
            sequential_duration_arpu(10.0, 0.05, 10_000, baseline_sd=5.0, sequential_tau=0.0)


class ArpuDurationCliAndConnectorTests(unittest.TestCase):
    ARPU_BASE = [
        "--metric",
        "arpu",
        "--baseline-mean",
        "12.5",
        "--baseline-sd",
        "25.0",
        "--relative-mde",
        "0.05",
        "--daily-traffic",
        "50000",
    ]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, name: str, content: str) -> Path:
        path = self.dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def run_cli(self, args):
        return CliRunner().invoke(cli, [str(a) for a in args])

    def test_cli_all_three_methods_arpu(self):
        for method in ("sequential", "fixed", "bayesian"):
            extra = (
                ["--max-days", "3", "--max-expected-loss", "0.05"] if method == "bayesian" else []
            )
            res = self.run_cli(["duration", "--method", method, *self.ARPU_BASE, *extra])
            self.assertEqual(res.exit_code, 0, f"{method}: {res.output}")
            data = json.loads(res.output)
            self.assertEqual(data["schema_version"], "1")
            self.assertEqual(data["assumptions"]["metric"], "arpu")
            self.assertEqual(data["assumptions"]["baseline_mean"], 12.5)
            self.assertEqual(data["assumptions"]["baseline_sd"], 25.0)

    def test_cli_baseline_cv_matches_baseline_sd(self):
        sd_res = self.run_cli(["duration", "--method", "fixed", *self.ARPU_BASE])
        cv_res = self.run_cli(
            [
                "duration",
                "--method",
                "fixed",
                "--metric",
                "arpu",
                "--baseline-mean",
                "12.5",
                "--baseline-cv",
                "2.0",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ]
        )
        self.assertEqual(sd_res.exit_code, 0, sd_res.output)
        self.assertEqual(cv_res.exit_code, 0, cv_res.output)
        sd_data, cv_data = json.loads(sd_res.output), json.loads(cv_res.output)
        self.assertEqual(sd_data["n_per_variant"], cv_data["n_per_variant"])
        self.assertEqual(cv_data["assumptions"]["baseline_cv"], 2.0)

    def test_connector_parses_arpu_columns_csv_and_xlsx(self):
        csv_path = self.write(
            "arpu.csv",
            "Metric,Method,Baseline_Mean,Baseline_CV,Relative_MDE,Daily_Traffic,N_Variants\n"
            "ARPU,Fixed,12.5,2.0,0.05,50000,3\n",
        )
        req = build_duration_request_from_rows(read_table(str(csv_path)))
        self.assertEqual(
            req,
            {
                "metric": "arpu",
                "method": "fixed",
                "baseline_mean": 12.5,
                "baseline_cv": 2.0,
                "relative_mde": 0.05,
                "daily_traffic": 50000,
                "n_variants": 3,
            },
        )
        res = self.run_cli(["duration", "--input", csv_path])
        self.assertEqual(res.exit_code, 0, res.output)
        data = json.loads(res.output)
        self.assertEqual(data["method"], "fixed_horizon")
        self.assertEqual(data["assumptions"]["metric"], "arpu")
        self.assertEqual(data["assumptions"]["baseline_sd"], 25.0)

        xlsx_path = self.dir / "arpu.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Plan"
        ws.append(
            [
                "metric",
                "method",
                "baseline_mean",
                "baseline_sd",
                "relative_mde",
                "daily_traffic",
            ]
        )
        ws.append(["arpu", "sequential", 10.0, 15.0, 0.1, 40000])
        wb.save(xlsx_path)

        xlsx_res = self.run_cli(["duration", "--input", xlsx_path, "--sheet", "Plan"])
        self.assertEqual(xlsx_res.exit_code, 0, xlsx_res.output)
        xlsx_data = json.loads(xlsx_res.output)
        self.assertEqual(xlsx_data["method"], "sequential")
        self.assertEqual(xlsx_data["assumptions"]["metric"], "arpu")

    def test_cli_overrides_over_file(self):
        arpu_csv = self.write(
            "arpu.csv",
            "metric,method,baseline_mean,baseline_sd,relative_mde,daily_traffic,alpha,power\n"
            "arpu,sequential,12.5,25.0,0.05,50000,0.05,0.8\n",
        )
        # Override baseline_sd in file with --baseline-cv on CLI.
        res_cv = self.run_cli(["duration", "--input", arpu_csv, "--baseline-cv", "1.0"])
        self.assertEqual(res_cv.exit_code, 0, res_cv.output)
        data_cv = json.loads(res_cv.output)
        self.assertEqual(data_cv["assumptions"]["baseline_sd"], 12.5)
        self.assertEqual(data_cv["assumptions"]["baseline_cv"], 1.0)

        # Override method from sequential to bayesian ignores alpha/power from file.
        res_bayes = self.run_cli(
            ["duration", "--input", arpu_csv, "--method", "bayesian", "--max-days", "2"]
        )
        self.assertEqual(res_bayes.exit_code, 0, res_bayes.output)
        self.assertEqual(json.loads(res_bayes.output)["method"], "bayesian")

        # Override metric from arpu to conversion_rate ignores ARPU columns from file.
        res_conv = self.run_cli(
            [
                "duration",
                "--input",
                arpu_csv,
                "--metric",
                "conversion_rate",
                "--baseline-rate",
                "0.04",
            ]
        )
        self.assertEqual(res_conv.exit_code, 0, res_conv.output)
        self.assertEqual(json.loads(res_conv.output)["assumptions"]["metric"], "conversion_rate")

        # Override metric from conversion_rate to arpu ignores baseline_rate from file.
        conv_csv = self.write(
            "conv.csv",
            "metric,method,baseline_rate,relative_mde,daily_traffic\n"
            "conversion_rate,fixed,0.04,0.05,50000\n",
        )
        res_arpu = self.run_cli(
            [
                "duration",
                "--input",
                conv_csv,
                "--metric",
                "arpu",
                "--baseline-mean",
                "10.0",
                "--baseline-sd",
                "20.0",
            ]
        )
        self.assertEqual(res_arpu.exit_code, 0, res_arpu.output)
        self.assertEqual(json.loads(res_arpu.output)["assumptions"]["metric"], "arpu")

        # Override baseline_cv in file with --baseline-sd on CLI.
        cv_csv = self.write(
            "arpu_cv.csv",
            "metric,method,baseline_mean,baseline_cv,relative_mde,daily_traffic\n"
            "arpu,sequential,10.0,2.0,0.05,50000\n",
        )
        res_sd = self.run_cli(["duration", "--input", cv_csv, "--baseline-sd", "15.0"])
        self.assertEqual(res_sd.exit_code, 0, res_sd.output)
        sd_assumptions = json.loads(res_sd.output)["assumptions"]
        self.assertEqual(sd_assumptions["baseline_sd"], 15.0)
        self.assertNotIn("baseline_cv", sd_assumptions)

    def test_rejected_combinations_exit_2(self):
        cases = [
            # --baseline-rate with --metric arpu
            [*self.ARPU_BASE, "--baseline-rate", "0.04"],
            [
                "--metric",
                "arpu",
                "--baseline-rate",
                "0.04",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            # ARPU flags with default conversion_rate
            [
                "--baseline-rate",
                "0.04",
                "--baseline-mean",
                "10",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            [
                "--baseline-rate",
                "0.04",
                "--baseline-sd",
                "20",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            [
                "--baseline-rate",
                "0.04",
                "--baseline-cv",
                "2.0",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            # Explicit --metric conversion_rate with ARPU flags
            [
                "--metric",
                "conversion_rate",
                "--baseline-mean",
                "10",
                "--baseline-sd",
                "20",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            # Both --baseline-sd and --baseline-cv
            [*self.ARPU_BASE, "--baseline-cv", "2.0"],
            # Neither --baseline-sd nor --baseline-cv
            [
                "--metric",
                "arpu",
                "--baseline-mean",
                "12.5",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            # Missing required ARPU flags
            ["--metric", "arpu"],
            # Method-inapplicable flags with --metric arpu
            [*self.ARPU_BASE, "--method", "bayesian", "--alpha", "0.05"],
            [*self.ARPU_BASE, "--method", "bayesian", "--power", "0.8"],
            [*self.ARPU_BASE, "--method", "bayesian", "--sequential-tau", "1.0"],
            [*self.ARPU_BASE, "--method", "fixed", "--sequential-tau", "1.0"],
            [*self.ARPU_BASE, "--method", "fixed", "--prob-threshold", "0.95"],
            [*self.ARPU_BASE, "--method", "sequential", "--max-days", "30"],
        ]
        for args in cases:
            with self.subTest(args=args):
                res = self.run_cli(["duration", *args])
                self.assertEqual(res.exit_code, 2, res.output)

    def test_rejected_file_combinations_exit_2(self):
        # Both baseline_sd and baseline_cv in file
        both_csv = self.write(
            "both.csv",
            "metric,baseline_mean,baseline_sd,baseline_cv,relative_mde,daily_traffic\n"
            "arpu,10,20,2,0.05,50000\n",
        )
        res = self.run_cli(["duration", "--input", both_csv])
        self.assertEqual(res.exit_code, 2, res.output)

        # baseline_rate in an arpu file
        mixed_csv = self.write(
            "mixed.csv",
            "metric,baseline_rate,baseline_mean,baseline_sd,relative_mde,daily_traffic\n"
            "arpu,0.04,10,20,0.05,50000\n",
        )
        res = self.run_cli(["duration", "--input", mixed_csv])
        self.assertEqual(res.exit_code, 2, res.output)

        # baseline_mean in a conversion_rate file
        conv_bad = self.write(
            "conv_bad.csv",
            "baseline_rate,baseline_mean,relative_mde,daily_traffic\n0.04,10,0.05,50000\n",
        )
        res = self.run_cli(["duration", "--input", conv_bad])
        self.assertEqual(res.exit_code, 2, res.output)

        # Unknown metric in file
        bad_metric = self.write(
            "bad_metric.csv",
            "metric,baseline_mean,baseline_sd,relative_mde,daily_traffic\nrevenue,10,20,0.05,50000\n",
        )
        res = self.run_cli(["duration", "--input", bad_metric])
        self.assertEqual(res.exit_code, 2, res.output)

    def test_invalid_numeric_arpu_flags_exit_1(self):
        cases = [
            [
                "--metric",
                "arpu",
                "--baseline-mean",
                "0",
                "--baseline-sd",
                "10",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            [
                "--metric",
                "arpu",
                "--baseline-mean",
                "10",
                "--baseline-sd",
                "-1",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            [
                "--metric",
                "arpu",
                "--baseline-mean",
                "10",
                "--baseline-cv",
                "0",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "50000",
            ],
            [
                "--metric",
                "arpu",
                "--baseline-mean",
                "10",
                "--baseline-sd",
                "10",
                "--relative-mde",
                "0",
                "--daily-traffic",
                "50000",
            ],
            [
                "--metric",
                "arpu",
                "--baseline-mean",
                "10",
                "--baseline-sd",
                "10",
                "--relative-mde",
                "0.05",
                "--daily-traffic",
                "0",
            ],
        ]
        for args in cases:
            with self.subTest(args=args):
                res = self.run_cli(["duration", "--method", "sequential", *args])
                self.assertEqual(res.exit_code, 1, res.output)
                self.assertIn("Error", res.output)
                self.assertNotIn("Traceback", res.output)


if __name__ == "__main__":
    unittest.main()
