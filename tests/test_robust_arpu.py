from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from test_schema import validate_json_schema

from ablift.cli import cli
from ablift.connectors import UsageOptionError, build_payload_from_rows
from ablift.engine import analyze, parse_payload
from ablift.reporting import build_markdown_report
from ablift.schemas import get_schema
from ablift.stats import (
    aggregated_arpu_tail_diagnostics,
    bootstrap_mean_diff_ci,
    row_level_tail_diagnostics,
    winsorize_pooled,
)


def _rows_from_arrays(control: np.ndarray, treatment: np.ndarray) -> list[dict[str, str]]:
    return [{"variant": "control", "revenue": f"{v:.6f}"} for v in control] + [
        {"variant": "treatment", "revenue": f"{v:.6f}"} for v in treatment
    ]


class RobustArpuStatsTests(unittest.TestCase):
    def test_pooled_winsorization_reduces_pareto_variance_and_preserves_type_i_error(self):
        rng = np.random.default_rng(2026)
        n_trials = 250
        n_per_arm = 400
        alpha = 0.05
        false_positives = 0
        var_reductions: list[float] = []

        for _ in range(n_trials):
            # Heavy-tailed Pareto(a=2.1) revenue under H0 (identical control & treatment)
            c = rng.pareto(a=2.1, size=n_per_arm) * 10.0
            t = rng.pareto(a=2.1, size=n_per_arm) * 10.0
            capped, diag = winsorize_pooled({"control": c, "treatment": t}, quantile=0.98)
            var_reductions.append(diag["variance_reduction_ratio"])

            payload = {
                "experiment_name": "pareto_h0",
                "method": "fixed",
                "primary_metric": "arpu",
                "alpha": alpha,
                "variants": [
                    {
                        "name": "control",
                        "visitors": n_per_arm,
                        "revenue_sum": float(np.sum(capped["control"])),
                        "revenue_sum_squares": float(np.sum(capped["control"] ** 2)),
                        "is_control": True,
                    },
                    {
                        "name": "treatment",
                        "visitors": n_per_arm,
                        "revenue_sum": float(np.sum(capped["treatment"])),
                        "revenue_sum_squares": float(np.sum(capped["treatment"] ** 2)),
                        "is_control": False,
                    },
                ],
            }
            res = analyze(parse_payload(payload))
            if res.comparisons[0].significant:
                false_positives += 1

        fp_rate = false_positives / n_trials
        # Variance shrinks substantially on Pareto data and Type I error stays near nominal 0.05.
        self.assertGreater(float(np.mean(var_reductions)), 0.45)
        self.assertLess(fp_rate, 0.09)
        self.assertGreater(fp_rate, 0.01)

    def test_pooled_cap_shared_across_arms_and_diagnostics_accurate(self):
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t = np.array([1.0, 2.0, 3.0, 4.0, 500.0])
        capped, diag = winsorize_pooled({"control": c, "treatment": t}, quantile=0.90)
        expected_cap = float(np.quantile(np.concatenate([c, t]), 0.90))

        self.assertAlmostEqual(diag["cap_value"], expected_cap, places=12)
        self.assertEqual(diag["capped_count"], 1)
        self.assertEqual(diag["per_variant_capped"], {"control": 0, "treatment": 1})
        self.assertAlmostEqual(float(np.max(capped["treatment"])), expected_cap, places=12)
        self.assertGreater(diag["variance_reduction_ratio"], 0.9)

    def test_winsorize_pooled_rejects_invalid_quantiles(self):
        groups = {"control": np.array([1.0, 2.0]), "treatment": np.array([1.0, 2.0])}
        for bad in (0.5, 1.0, 0.0, -0.5, 1.2, float("nan"), True):
            with self.assertRaises(ValueError):
                winsorize_pooled(groups, bad)  # type: ignore[arg-type]

    def test_heavy_tail_flag_fires_on_lognormal_and_stays_silent_on_normal(self):
        rng = np.random.default_rng(77)

        # Normal-ish positive revenue -> silent
        normal_c = rng.normal(50.0, 5.0, size=500)
        normal_t = rng.normal(51.0, 5.0, size=500)
        normal_diag = row_level_tail_diagnostics({"control": normal_c, "treatment": normal_t})
        self.assertFalse(normal_diag["heavy_tailed"])

        # Heavy-tailed lognormal revenue -> fires
        heavy_c = rng.lognormal(mean=1.0, sigma=2.2, size=500)
        heavy_t = rng.lognormal(mean=1.0, sigma=2.2, size=500)
        heavy_diag = row_level_tail_diagnostics({"control": heavy_c, "treatment": heavy_t})
        self.assertTrue(heavy_diag["heavy_tailed"])
        self.assertGreater(heavy_diag["max_arm_skewness"], 2.0)

        # Aggregated CV check: low CV silent, high CV (> 3.0) fires
        agg_normal = aggregated_arpu_tail_diagnostics(
            [
                ("control", 500, float(np.sum(normal_c)), float(np.sum(normal_c**2))),
                ("treatment", 500, float(np.sum(normal_t)), float(np.sum(normal_t**2))),
            ]
        )
        self.assertFalse(agg_normal["heavy_tailed"])

        agg_heavy = aggregated_arpu_tail_diagnostics(
            [
                ("control", 500, float(np.sum(heavy_c)), float(np.sum(heavy_c**2))),
                ("treatment", 500, float(np.sum(heavy_t)), float(np.sum(heavy_t**2))),
            ]
        )
        self.assertTrue(agg_heavy["heavy_tailed"])
        self.assertGreater(agg_heavy["max_cv"], 3.0)

    def test_bootstrap_mean_diff_ci_matches_welch_on_normal_data(self):
        rng = np.random.default_rng(1234)
        c = rng.normal(20.0, 4.0, size=400)
        t = rng.normal(21.5, 4.0, size=400)

        lo, hi = bootstrap_mean_diff_ci(c, t, n_bootstrap=2000, alpha=0.05, seed=7)
        lo2, hi2 = bootstrap_mean_diff_ci(c, t, n_bootstrap=2000, alpha=0.05, seed=7)
        self.assertEqual((lo, hi), (lo2, hi2))

        rows = _rows_from_arrays(c, t)
        payload = build_payload_from_rows(rows, defaults={"method": "fixed"})
        res = analyze(parse_payload(payload))
        comp = res.comparisons[0]
        self.assertAlmostEqual(lo, comp.ci_low, delta=0.12)
        self.assertAlmostEqual(hi, comp.ci_high, delta=0.12)


class RobustArpuIntegrationAndCliTests(unittest.TestCase):
    def test_heavy_tailed_revenue_risk_flag_in_engine_and_report(self):
        rng = np.random.default_rng(101)
        # Single huge whale in row-level data triggers heavy_tailed_revenue
        c = rng.normal(20.0, 2.0, size=100)
        t = rng.normal(20.0, 2.0, size=100)
        t[0] = 2500.0

        rows = _rows_from_arrays(c, t)
        raw_payload = build_payload_from_rows(rows, defaults={"method": "fixed"})
        raw_res = analyze(parse_payload(raw_payload))
        self.assertEqual(validate_json_schema(raw_res.to_dict(), get_schema("analyze-output")), [])
        self.assertIn("heavy_tailed_revenue", raw_res.recommendation.risk_flags)
        raw_report = build_markdown_report(raw_res)
        self.assertIn("heavy_tailed_revenue", raw_report)

        # Winsorizing tames the single whale and eliminates the flag
        win_payload = build_payload_from_rows(
            rows, defaults={"method": "fixed", "winsorize": 0.98, "bootstrap_samples": 500}
        )
        win_res = analyze(parse_payload(win_payload))
        self.assertEqual(validate_json_schema(win_res.to_dict(), get_schema("analyze-output")), [])
        self.assertNotIn("heavy_tailed_revenue", win_res.recommendation.risk_flags)
        self.assertIn("winsorization", win_res.analysis_settings.input_interpretation)
        self.assertIn("bootstrap_ci", win_res.analysis_settings.input_interpretation)

        win_report = build_markdown_report(win_res)
        self.assertIn("Winsorization:", win_report)
        self.assertIn("Bootstrap CI", win_report)

    def test_aggregated_arpu_emits_heavy_tailed_revenue_only_when_cv_exceeds_3(self):
        # High-CV aggregated ARPU: mean=10, sd~40 -> CV~4.0 -> fires
        heavy_payload = {
            "experiment_name": "agg_heavy",
            "method": "fixed",
            "primary_metric": "arpu",
            "variants": [
                {
                    "name": "control",
                    "visitors": 1000,
                    "revenue_sum": 10000.0,
                    "revenue_sum_squares": 1_700_000.0,
                    "is_control": True,
                },
                {
                    "name": "treatment",
                    "visitors": 1000,
                    "revenue_sum": 10500.0,
                    "revenue_sum_squares": 1_800_000.0,
                    "is_control": False,
                },
            ],
        }
        heavy_res = analyze(parse_payload(heavy_payload))
        self.assertIn("heavy_tailed_revenue", heavy_res.recommendation.risk_flags)
        self.assertTrue(
            heavy_res.analysis_settings.input_interpretation["arpu_tail_diagnostics"][
                "heavy_tailed"
            ]
        )

        # Normal-CV aggregated ARPU: mean=10, sd~5 -> CV~0.5 -> silent
        normal_payload = {
            "experiment_name": "agg_normal",
            "method": "fixed",
            "primary_metric": "arpu",
            "variants": [
                {
                    "name": "control",
                    "visitors": 1000,
                    "revenue_sum": 10000.0,
                    "revenue_sum_squares": 125_000.0,
                    "is_control": True,
                },
                {
                    "name": "treatment",
                    "visitors": 1000,
                    "revenue_sum": 10500.0,
                    "revenue_sum_squares": 136_000.0,
                    "is_control": False,
                },
            ],
        }
        normal_res = analyze(parse_payload(normal_payload))
        self.assertNotIn("heavy_tailed_revenue", normal_res.recommendation.risk_flags)
        self.assertFalse(
            normal_res.analysis_settings.input_interpretation["arpu_tail_diagnostics"][
                "heavy_tailed"
            ]
        )

    def test_cli_winsorize_and_bootstrap_end_to_end_and_usage_errors(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Row-level ARPU CSV succeeds with --winsorize and --bootstrap-samples
            row_csv = Path(tmp) / "row_arpu.csv"
            with row_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["variant", "revenue"])
                for i in range(60):
                    w.writerow(["control", 10.0 + (i % 5)])
                    w.writerow(["treatment", 12.0 + (i % 5) if i < 59 else 5000.0])

            ok = runner.invoke(
                cli,
                [
                    "analyze",
                    "--input",
                    str(row_csv),
                    "--method",
                    "fixed",
                    "--winsorize",
                    "0.95",
                    "--bootstrap-samples",
                    "400",
                    "--random-seed",
                    "11",
                ],
            )
            self.assertEqual(ok.exit_code, 0, ok.output)
            out = json.loads(ok.output)
            interp = out["analysis_settings"]["input_interpretation"]
            self.assertEqual(interp["winsorization"]["quantile"], 0.95)
            self.assertEqual(interp["bootstrap_ci"]["treatment"]["n_bootstrap"], 400)

            # 2. Aggregated CSV + --winsorize -> exit code 2 usage error
            agg_csv = Path(tmp) / "agg_arpu.csv"
            agg_csv.write_text(
                "variant,visitors,rev_sum,rev_sq\ncontrol,100,1000,20000\ntreatment,100,1100,22000\n",
                encoding="utf-8",
            )
            bad_agg = runner.invoke(
                cli, ["analyze", "--input", str(agg_csv), "--winsorize", "0.99"]
            )
            self.assertEqual(bad_agg.exit_code, 2, bad_agg.output)
            self.assertIn("row-level ARPU", bad_agg.output)

            # 3. Aggregated JSON + --winsorize -> exit code 2 usage error
            agg_json = Path(tmp) / "agg_arpu.json"
            agg_json.write_text(
                json.dumps(
                    {
                        "experiment_name": "j",
                        "primary_metric": "arpu",
                        "variants": [
                            {
                                "name": "control",
                                "visitors": 100,
                                "revenue_sum": 1000.0,
                                "revenue_sum_squares": 20000.0,
                                "is_control": True,
                            },
                            {
                                "name": "treatment",
                                "visitors": 100,
                                "revenue_sum": 1100.0,
                                "revenue_sum_squares": 22000.0,
                                "is_control": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bad_json = runner.invoke(
                cli, ["analyze", "--input", str(agg_json), "--winsorize", "0.99"]
            )
            self.assertEqual(bad_json.exit_code, 2, bad_json.output)
            self.assertIn("JSON", bad_json.output)

            # 4. Binary row-level conversion CSV + --winsorize -> exit code 2 usage error
            conv_csv = Path(tmp) / "conv.csv"
            conv_csv.write_text(
                "variant,converted\ncontrol,0\ncontrol,1\ntreatment,0\ntreatment,1\n",
                encoding="utf-8",
            )
            bad_conv = runner.invoke(
                cli, ["analyze", "--input", str(conv_csv), "--winsorize", "0.99"]
            )
            self.assertEqual(bad_conv.exit_code, 2, bad_conv.output)
            self.assertIn("row-level ARPU", bad_conv.output)

            # 5. Out-of-range --winsorize or too-small --bootstrap-samples -> exit code 2
            bad_q = runner.invoke(cli, ["analyze", "--input", str(row_csv), "--winsorize", "0.5"])
            self.assertEqual(bad_q.exit_code, 2, bad_q.output)

            bad_boot = runner.invoke(
                cli, ["analyze", "--input", str(row_csv), "--bootstrap-samples", "20"]
            )
            self.assertEqual(bad_boot.exit_code, 2, bad_boot.output)

    def test_connector_raises_usage_option_error_for_aggregated_bootstrap(self):
        rows = [
            {"variant": "control", "visitors": "100", "rev": "500", "rev2": "5000"},
            {"variant": "treatment", "visitors": "100", "rev": "550", "rev2": "5500"},
        ]
        with self.assertRaises(UsageOptionError):
            build_payload_from_rows(rows, defaults={"bootstrap_samples": 500})


if __name__ == "__main__":
    unittest.main()
