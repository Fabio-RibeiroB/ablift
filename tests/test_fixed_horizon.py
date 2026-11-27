import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from scipy.stats import t as scipy_t
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from ablift.cli import cli
from ablift.engine import (
    analyze,
    parse_payload,
    student_t_ppf,
    student_t_two_sided_p,
)
from ablift.planning import fixed_horizon_duration_conversion
from ablift.reporting import build_markdown_report


class StudentTReferenceTests(unittest.TestCase):
    def test_student_t_two_sided_p_matches_scipy(self):
        for df in (1.0, 1.5, 2.0, 4.7, 10.0, 35.2, 200.0, 10_000.0, 1e6, 1e8):
            for t_val in (0.0, 0.3, 1.0, 2.1, 4.5, 9.0):
                with self.subTest(df=df, t_val=t_val):
                    got = student_t_two_sided_p(t_val, df)
                    expected = float(2.0 * scipy_t.sf(abs(t_val), df))
                    self.assertAlmostEqual(got, expected, places=7)
                    self.assertAlmostEqual(student_t_two_sided_p(-t_val, df), got, places=14)

    def test_student_t_ppf_matches_scipy(self):
        for df in (1.0, 1.5, 2.0, 4.7, 10.0, 35.2, 200.0, 10_000.0, 1e6, 1e8):
            for alpha in (0.2, 0.05, 0.01, 0.001):
                q = 1.0 - alpha / 2.0
                with self.subTest(df=df, alpha=alpha):
                    got = student_t_ppf(q, df)
                    expected = float(scipy_t.ppf(q, df))
                    self.assertAlmostEqual(got, expected, delta=max(1e-6, abs(expected) * 1e-6))
                    self.assertAlmostEqual(student_t_ppf(1.0 - q, df), -got, places=12)


class FixedHorizonConversionTests(unittest.TestCase):
    def test_two_variant_matches_statsmodels_ztest_and_unpooled_ci(self):
        cases = [
            (10_000, 500, 10_000, 620, 0.05),
            (5_000, 200, 7_500, 270, 0.01),
            (2_000, 100, 2_000, 105, 0.10),
        ]
        for n1, x1, n2, x2, alpha in cases:
            with self.subTest(n1=n1, x1=x1, n2=n2, x2=x2, alpha=alpha):
                payload = {
                    "experiment_name": "fixed_cr",
                    "method": "fixed",
                    "primary_metric": "conversion_rate",
                    "alpha": alpha,
                    "expected_split": [n1, n2],
                    "variants": [
                        {"name": "control", "visitors": n1, "conversions": x1, "is_control": True},
                        {"name": "treatment", "visitors": n2, "conversions": x2},
                    ],
                }
                result = analyze(parse_payload(payload))
                comp = result.comparisons[0]
                ref_z, ref_p = proportions_ztest([x2, x1], [n2, n1])
                ref_ci_low, ref_ci_high = confint_proportions_2indep(
                    x2, n2, x1, n1, method="wald", compare="diff", alpha=alpha
                )

                self.assertAlmostEqual(comp.test_statistic, float(ref_z), places=10)
                self.assertAlmostEqual(comp.raw_p_value, float(ref_p), places=10)
                self.assertAlmostEqual(comp.p_value, float(ref_p), places=10)
                self.assertAlmostEqual(comp.ci_low, float(ref_ci_low), places=10)
                self.assertAlmostEqual(comp.ci_high, float(ref_ci_high), places=10)
                self.assertEqual(comp.significant, bool(ref_p <= alpha))
                self.assertIsNone(comp.degrees_of_freedom)
                self.assertIsNone(comp.e_value)
                self.assertIsNone(comp.e_threshold)
                self.assertIsNone(comp.probability_beats_control)
                self.assertIsNone(result.bayesian_arms)

    def test_multivariant_bonferroni_matches_duration_fixed(self):
        # Construct v1 with raw p-value between 0.025 and 0.05 so it is significant
        # in 2-arm (alpha=0.05) but NOT in 3-arm (alpha_per_comparison=0.025).
        two_arm = analyze(
            parse_payload(
                {
                    "experiment_name": "two_arm",
                    "method": "fixed",
                    "alpha": 0.05,
                    "variants": [
                        {
                            "name": "control",
                            "visitors": 10000,
                            "conversions": 500,
                            "is_control": True,
                        },
                        {"name": "v1", "visitors": 10000, "conversions": 565},
                    ],
                }
            )
        )
        three_arm = analyze(
            parse_payload(
                {
                    "experiment_name": "three_arm",
                    "method": "fixed",
                    "alpha": 0.05,
                    "variants": [
                        {
                            "name": "control",
                            "visitors": 10000,
                            "conversions": 500,
                            "is_control": True,
                        },
                        {"name": "v1", "visitors": 10000, "conversions": 565},
                        {"name": "v2", "visitors": 10000, "conversions": 500},
                    ],
                }
            )
        )
        plan = fixed_horizon_duration_conversion(
            baseline_rate=0.05,
            relative_mde=0.1,
            daily_total_traffic=10000,
            n_variants=3,
            alpha=0.05,
        )

        self.assertTrue(two_arm.comparisons[0].significant)
        self.assertGreater(two_arm.comparisons[0].raw_p_value, 0.025)
        self.assertLess(two_arm.comparisons[0].raw_p_value, 0.05)

        comp_v1 = three_arm.comparisons[0]
        self.assertFalse(comp_v1.significant)
        self.assertAlmostEqual(comp_v1.p_value, comp_v1.raw_p_value * 2.0, places=12)

        mc = three_arm.analysis_settings.multiple_comparisons
        self.assertEqual(mc["correction"], "bonferroni")
        self.assertEqual(mc["n_comparisons"], 2)
        self.assertAlmostEqual(mc["alpha_per_comparison"], plan.diagnostics["alpha_per_comparison"])

        # Simultaneous Bonferroni CI matches statsmodels at alpha / 2.
        ref_lo, ref_hi = confint_proportions_2indep(
            565, 10000, 500, 10000, method="wald", compare="diff", alpha=0.025
        )
        self.assertAlmostEqual(comp_v1.ci_low, float(ref_lo), places=10)
        self.assertAlmostEqual(comp_v1.ci_high, float(ref_hi), places=10)

    def test_zero_conversion_edge_case(self):
        result = analyze(
            parse_payload(
                {
                    "experiment_name": "zeros",
                    "method": "fixed",
                    "variants": [
                        {"name": "control", "visitors": 500, "conversions": 0, "is_control": True},
                        {"name": "v1", "visitors": 500, "conversions": 0},
                    ],
                }
            )
        )
        comp = result.comparisons[0]
        self.assertAlmostEqual(comp.p_value, 1.0)
        self.assertFalse(comp.significant)
        self.assertEqual(comp.ci_low, 0.0)
        self.assertEqual(comp.ci_high, 0.0)


class FixedHorizonArpuTests(unittest.TestCase):
    def test_welch_t_test_matches_scipy_for_small_and_large_samples(self):
        rng = np.random.default_rng(42)
        cases = [
            (rng.normal(10.0, 2.0, size=12), rng.normal(13.0, 4.5, size=18), 0.05),
            (rng.normal(25.0, 15.0, size=500), rng.normal(27.5, 20.0, size=600), 0.01),
            (rng.exponential(5.0, size=3000), rng.exponential(5.3, size=3000), 0.05),
        ]
        for c_vals, t_vals, alpha in cases:
            with self.subTest(n1=len(c_vals), n2=len(t_vals), alpha=alpha):
                payload = {
                    "experiment_name": "welch_arpu",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "alpha": alpha,
                    "expected_split": [len(c_vals), len(t_vals)],
                    "variants": [
                        {
                            "name": "control",
                            "visitors": len(c_vals),
                            "conversions": 0,
                            "revenue_sum": float(np.sum(c_vals)),
                            "revenue_sum_squares": float(np.sum(c_vals**2)),
                            "is_control": True,
                        },
                        {
                            "name": "treatment",
                            "visitors": len(t_vals),
                            "conversions": 0,
                            "revenue_sum": float(np.sum(t_vals)),
                            "revenue_sum_squares": float(np.sum(t_vals**2)),
                        },
                    ],
                }
                result = analyze(parse_payload(payload))
                comp = result.comparisons[0]

                ref = ttest_ind(t_vals, c_vals, equal_var=False)
                ref_ci = ref.confidence_interval(confidence_level=1.0 - alpha)

                self.assertAlmostEqual(comp.test_statistic, float(ref.statistic), places=10)
                self.assertAlmostEqual(comp.degrees_of_freedom, float(ref.df), places=9)
                self.assertAlmostEqual(comp.raw_p_value, float(ref.pvalue), places=10)
                self.assertAlmostEqual(comp.p_value, float(ref.pvalue), places=10)
                self.assertAlmostEqual(comp.ci_low, float(ref_ci.low), places=8)
                self.assertAlmostEqual(comp.ci_high, float(ref_ci.high), places=8)
                self.assertEqual(comp.significant, bool(ref.pvalue <= alpha))

    def test_welch_arpu_multivariant_bonferroni(self):
        rng = np.random.default_rng(101)
        c = rng.normal(20.0, 5.0, size=200)
        t1 = rng.normal(21.5, 6.0, size=200)
        t2 = rng.normal(19.5, 4.0, size=200)

        def row(name: str, vals: np.ndarray, is_control: bool = False) -> dict:
            return {
                "name": name,
                "visitors": len(vals),
                "conversions": 0,
                "revenue_sum": float(np.sum(vals)),
                "revenue_sum_squares": float(np.sum(vals**2)),
                "is_control": is_control,
            }

        result = analyze(
            parse_payload(
                {
                    "experiment_name": "welch_multi",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "alpha": 0.05,
                    "variants": [row("control", c, True), row("t1", t1), row("t2", t2)],
                }
            )
        )
        for comp, t_vals in zip(result.comparisons, (t1, t2), strict=True):
            ref = ttest_ind(t_vals, c, equal_var=False)
            ref_ci = ref.confidence_interval(confidence_level=1.0 - 0.025)
            self.assertAlmostEqual(comp.raw_p_value, float(ref.pvalue), places=10)
            self.assertAlmostEqual(comp.p_value, min(1.0, float(ref.pvalue) * 2.0), places=10)
            self.assertAlmostEqual(comp.ci_low, float(ref_ci.low), places=8)
            self.assertAlmostEqual(comp.ci_high, float(ref_ci.high), places=8)

    def test_arpu_edge_cases_valid_json(self):
        # Single visitor cannot estimate variance.
        single = analyze(
            parse_payload(
                {
                    "experiment_name": "n1",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "variants": [
                        {
                            "name": "control",
                            "visitors": 1,
                            "revenue_sum": 10.0,
                            "revenue_sum_squares": 100.0,
                            "is_control": True,
                        },
                        {
                            "name": "v1",
                            "visitors": 1,
                            "revenue_sum": 15.0,
                            "revenue_sum_squares": 225.0,
                        },
                    ],
                }
            )
        )
        comp = single.comparisons[0]
        self.assertEqual(comp.p_value, 1.0)
        self.assertFalse(comp.significant)
        self.assertIsNone(comp.ci_low)
        self.assertIsNone(comp.degrees_of_freedom)

        # Distinct zero-variance constants stay strict JSON (no inf/nan).
        zero_var = analyze(
            parse_payload(
                {
                    "experiment_name": "const",
                    "method": "fixed",
                    "primary_metric": "arpu",
                    "variants": [
                        {
                            "name": "control",
                            "visitors": 10,
                            "revenue_sum": 50.0,
                            "revenue_sum_squares": 250.0,
                            "is_control": True,
                        },
                        {
                            "name": "v1",
                            "visitors": 10,
                            "revenue_sum": 80.0,
                            "revenue_sum_squares": 640.0,
                        },
                    ],
                }
            )
        )
        serialized = json.dumps(zero_var.to_dict(), allow_nan=False)
        self.assertIn('"significant": true', serialized)
        self.assertEqual(zero_var.comparisons[0].ci_low, 3.0)
        self.assertEqual(zero_var.comparisons[0].ci_high, 3.0)


class FixedHorizonRecommendationAndReportTests(unittest.TestCase):
    def test_winner_recommendation_notes_single_look(self):
        result = analyze(
            parse_payload(
                {
                    "experiment_name": "fixed_win",
                    "method": "fixed",
                    "variants": [
                        {
                            "name": "control",
                            "visitors": 10000,
                            "conversions": 500,
                            "is_control": True,
                        },
                        {"name": "v1", "visitors": 10000, "conversions": 650},
                    ],
                }
            )
        )
        assert result.recommendation is not None
        self.assertEqual(result.recommendation.action, "ship_v1")
        self.assertIn(
            "evaluated once at the planned experiment end", result.recommendation.rationale
        )
        self.assertIn("peeking", result.recommendation.next_best_action)

    def test_inconclusive_recommendation_warns_against_peeking(self):
        result = analyze(
            parse_payload(
                {
                    "experiment_name": "fixed_flat",
                    "method": "fixed",
                    "variants": [
                        {
                            "name": "control",
                            "visitors": 1000,
                            "conversions": 50,
                            "is_control": True,
                        },
                        {"name": "v1", "visitors": 1000, "conversions": 54},
                    ],
                }
            )
        )
        assert result.recommendation is not None
        self.assertEqual(result.recommendation.action, "do_not_ship")
        self.assertIn("single planned", result.recommendation.rationale)
        self.assertIn("--method sequential", result.recommendation.next_best_action)

    def test_all_losers_returns_do_not_ship(self):
        result = analyze(
            parse_payload(
                {
                    "experiment_name": "fixed_loss",
                    "method": "fixed",
                    "variants": [
                        {
                            "name": "control",
                            "visitors": 10000,
                            "conversions": 600,
                            "is_control": True,
                        },
                        {"name": "v1", "visitors": 10000, "conversions": 420},
                        {"name": "v2", "visitors": 10000, "conversions": 400},
                    ],
                }
            )
        )
        assert result.recommendation is not None
        self.assertEqual(result.recommendation.action, "do_not_ship")
        self.assertIn("negative lift", result.recommendation.rationale)

    def test_markdown_report_fixed_conversion_and_arpu(self):
        cr_report = build_markdown_report(
            analyze(
                parse_payload(
                    {
                        "experiment_name": "fixed_cr_rep",
                        "method": "fixed",
                        "variants": [
                            {
                                "name": "control",
                                "visitors": 10000,
                                "conversions": 500,
                                "is_control": True,
                            },
                            {"name": "v1", "visitors": 10000, "conversions": 620},
                        ],
                    }
                )
            )
        )
        self.assertIn("- Method: `fixed`", cr_report)
        self.assertIn("Fixed-horizon (valid for a single planned look only", cr_report)
        self.assertIn("Confidence interval (absolute lift):", cr_report)
        self.assertIn("z-statistic:", cr_report)
        self.assertIn("Raw p-value:", cr_report)
        self.assertIn("Adjusted p-value:", cr_report)
        self.assertIn("Significant:", cr_report)
        for forbidden in (
            "Posterior samples",
            "Priors",
            "credible interval",
            "P(treatment",
            "Expected loss",
            "Joint Posterior Summary",
            "e-value",
            "e-threshold",
            "Always-valid",
            "mSPRT",
        ):
            self.assertNotIn(forbidden, cr_report)

        arpu_report = build_markdown_report(
            analyze(
                parse_payload(
                    {
                        "experiment_name": "fixed_arpu_rep",
                        "method": "fixed",
                        "primary_metric": "arpu",
                        "variants": [
                            {
                                "name": "control",
                                "visitors": 1000,
                                "revenue_sum": 5000.0,
                                "revenue_sum_squares": 400000.0,
                                "is_control": True,
                            },
                            {
                                "name": "v1",
                                "visitors": 1000,
                                "revenue_sum": 5600.0,
                                "revenue_sum_squares": 450000.0,
                            },
                        ],
                    }
                )
            )
        )
        self.assertIn("Welch t-statistic:", arpu_report)
        self.assertIn("(df=", arpu_report)


class FixedHorizonCliTests(unittest.TestCase):
    def test_cli_fixed_succeeds_with_repo_pyproject_defaults(self):
        runner = CliRunner()
        root = Path(__file__).resolve().parents[1] / "examples"
        for filename, metric in (
            ("conversion_multivariant.csv", "conversion_rate"),
            ("arpu_sequential.csv", "arpu"),
        ):
            with self.subTest(filename=filename):
                result = runner.invoke(
                    cli,
                    ["analyze", "--input", str(root / filename), "--method", "fixed"],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                data = json.loads(result.output)
                self.assertEqual(data["method"], "fixed")
                self.assertEqual(data["analysis_settings"]["primary_metric"], metric)
                self.assertEqual(
                    data["analysis_settings"]["multiple_comparisons"]["correction"], "bonferroni"
                )
                self.assertIsNone(data["bayesian_arms"])
                for comp in data["comparisons"]:
                    self.assertIsNotNone(comp["p_value"])
                    self.assertIsNotNone(comp["raw_p_value"])
                    self.assertIsNotNone(comp["test_statistic"])
                    if metric == "arpu":
                        self.assertIsNotNone(comp["degrees_of_freedom"])
                    else:
                        self.assertIsNone(comp["degrees_of_freedom"])

    def test_cli_rejects_sequential_and_bayesian_flags_for_fixed(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "data.csv"
            csv_path.write_text("v,n,c\ncontrol,1000,40\nv1,1000,60\n", encoding="utf-8")
            json_path = Path(tmp) / "fixed.json"
            json_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "fixed_json",
                        "method": "fixed",
                        "variants": [
                            {
                                "name": "control",
                                "visitors": 1000,
                                "conversions": 40,
                                "is_control": True,
                            },
                            {"name": "v1", "visitors": 1000, "conversions": 60},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            bad_flag_cases = [
                (["--method", "fixed", "--sequential-tau", "0.01"], "--sequential-tau"),
                (["--method", "fixed", "--samples", "5000"], "--samples"),
                (["--method", "fixed", "--random-seed", "42"], "--random-seed"),
                (["--method", "fixed", "--enable-recommendation"], "--enable-recommendation"),
                (["--method", "fixed", "--no-recommendation"], "--no-recommendation"),
                (["--method", "fixed", "--prob-threshold", "0.95"], "--prob-threshold"),
                (["--method", "fixed", "--max-expected-loss", "0.001"], "--max-expected-loss"),
            ]
            for extra_flags, expected_msg in bad_flag_cases:
                with self.subTest(flags=extra_flags):
                    res = runner.invoke(cli, ["analyze", "--input", str(csv_path), *extra_flags])
                    self.assertEqual(res.exit_code, 2, res.output)
                    self.assertIn(expected_msg, res.output)

            # Also rejected when method='fixed' comes from JSON rather than CLI flag.
            res_json = runner.invoke(
                cli, ["analyze", "--input", str(json_path), "--sequential-tau", "0.01"]
            )
            self.assertEqual(res_json.exit_code, 2, res_json.output)
            self.assertIn("--sequential-tau", res_json.output)

            # Works even when cwd has a pyproject.toml with Bayesian defaults.
            (Path(tmp) / "pyproject.toml").write_text(
                '[tool.ablift]\nmethod = "bayesian"\nsamples = 2000\nrandom_seed = 9\n'
                "[tool.ablift.decision_policy]\nenabled = true\n",
                encoding="utf-8",
            )
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                res_cfg = runner.invoke(
                    cli, ["analyze", "--input", str(csv_path), "--method", "fixed"]
                )
            finally:
                os.chdir(old_cwd)
            self.assertEqual(res_cfg.exit_code, 0, res_cfg.output)
            self.assertEqual(json.loads(res_cfg.output)["method"], "fixed")

    def test_json_sequential_tau_rejected_with_method_fixed(self):
        with self.assertRaisesRegex(ValueError, "sequential_tau"):
            analyze(
                parse_payload(
                    {
                        "experiment_name": "bad",
                        "method": "fixed",
                        "sequential_tau": 0.01,
                        "variants": [
                            {
                                "name": "control",
                                "visitors": 1000,
                                "conversions": 40,
                                "is_control": True,
                            },
                            {"name": "v1", "visitors": 1000, "conversions": 60},
                        ],
                    }
                )
            )

    def test_analyze_help_mentions_fixed(self):
        res = CliRunner().invoke(cli, ["analyze", "--help"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("fixed", res.output)
        self.assertIn("single planned look", res.output)


if __name__ == "__main__":
    unittest.main()
