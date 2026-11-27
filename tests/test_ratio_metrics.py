from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from test_schema import validate_json_schema

from ablift.cli import cli
from ablift.connectors import build_payload_from_rows
from ablift.engine import analyze, parse_payload
from ablift.reporting import build_markdown_report
from ablift.schemas import get_schema
from ablift.stats import delta_ratio_z_test, ratio_mean_and_var, validate_ratio_aggregates


def _agg_from_pairs(name: str, y: np.ndarray, x: np.ndarray, *, is_control: bool) -> dict:
    return {
        "name": name,
        "visitors": int(len(y)),
        "numerator_sum": float(np.sum(y)),
        "numerator_sum_squares": float(np.sum(y * y)),
        "denominator_sum": float(np.sum(x)),
        "denominator_sum_squares": float(np.sum(x * x)),
        "numerator_denominator_sum": float(np.sum(y * x)),
        "is_control": is_control,
    }


def _simulate_ctr_arm(
    rng: np.random.Generator, n_users: int, base_ctr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate clustered user-level (clicks, impressions) with strong intra-user correlation.

    Each user draws a latent click propensity from a dispersed Beta distribution and sees
    multiple impressions, so naive impression-level binomial SE severely underestimates
    variance while the user-level bivariate Delta method stays accurate.
    """
    impressions = rng.negative_binomial(n=2, p=0.15, size=n_users) + 1
    conc = 2.0
    user_rate = rng.beta(base_ctr * conc, (1.0 - base_ctr) * conc, size=n_users)
    clicks = rng.binomial(impressions, user_rate)
    return clicks.astype(float), impressions.astype(float)


class RatioStatsUnitTests(unittest.TestCase):
    def test_delta_method_matches_exact_hand_calculation(self):
        y = np.array([1.0, 2.0, 0.0, 5.0, 2.0])
        x = np.array([2.0, 4.0, 1.0, 8.0, 5.0])
        n = len(y)
        ratio, var = ratio_mean_and_var(
            n,
            float(np.sum(y)),
            float(np.sum(y * y)),
            float(np.sum(x)),
            float(np.sum(x * x)),
            float(np.sum(y * x)),
        )
        expected_ratio = float(np.sum(y) / np.sum(x))
        s_y2 = float(np.var(y, ddof=1))
        s_x2 = float(np.var(x, ddof=1))
        s_xy = float(np.cov(y, x, ddof=1)[0, 1])
        x_bar = float(np.mean(x))
        expected_var = (s_y2 - 2 * expected_ratio * s_xy + expected_ratio**2 * s_x2) / (
            n * x_bar * x_bar
        )
        self.assertAlmostEqual(ratio, expected_ratio, places=12)
        self.assertAlmostEqual(var, expected_var, places=12)

    def test_delta_method_se_matches_cluster_bootstrap_and_beats_naive_binomial(self):
        rng = np.random.default_rng(2026)
        y_c, x_c = _simulate_ctr_arm(rng, 2500, 0.08)
        y_t, x_t = _simulate_ctr_arm(rng, 2500, 0.085)

        r_c, var_c = ratio_mean_and_var(
            len(y_c),
            float(np.sum(y_c)),
            float(np.sum(y_c * y_c)),
            float(np.sum(x_c)),
            float(np.sum(x_c * x_c)),
            float(np.sum(y_c * x_c)),
        )
        r_t, var_t = ratio_mean_and_var(
            len(y_t),
            float(np.sum(y_t)),
            float(np.sum(y_t * y_t)),
            float(np.sum(x_t)),
            float(np.sum(x_t * x_t)),
            float(np.sum(y_t * x_t)),
        )
        delta_se = math.sqrt(var_c + var_t)

        # User-level cluster bootstrap of R_t - R_c
        boot_diffs = np.empty(2000)
        n_c, n_t = len(y_c), len(y_t)
        for i in range(len(boot_diffs)):
            ic = rng.integers(0, n_c, size=n_c)
            it = rng.integers(0, n_t, size=n_t)
            boot_diffs[i] = np.sum(y_t[it]) / np.sum(x_t[it]) - np.sum(y_c[ic]) / np.sum(x_c[ic])
        boot_se = float(np.std(boot_diffs, ddof=1))

        # Naive impression-level binomial SE ignores user clustering and underestimates SE
        naive_se = math.sqrt(
            r_c * (1.0 - r_c) / float(np.sum(x_c)) + r_t * (1.0 - r_t) / float(np.sum(x_t))
        )
        self.assertLess(abs(delta_se - boot_se) / boot_se, 0.08)
        self.assertGreater(delta_se, naive_se * 1.4)

    def test_zero_denominator_users_supported_for_aov(self):
        # 60% of users place 0 orders (denominator == 0, numerator == 0)
        y = np.array([0.0, 0.0, 0.0, 45.0, 120.0, 0.0, 30.0, 0.0, 65.0, 0.0])
        x = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        ratio, var = ratio_mean_and_var(
            len(y),
            float(np.sum(y)),
            float(np.sum(y * y)),
            float(np.sum(x)),
            float(np.sum(x * x)),
            float(np.sum(y * x)),
        )
        self.assertAlmostEqual(ratio, 260.0 / 5.0, places=10)
        self.assertGreater(var, 0.0)

    def test_zero_variance_edge_cases(self):
        # Every user has identical ratio y/x = 2.0 -> delta variance == 0
        y = np.array([2.0, 4.0, 6.0, 8.0])
        x = np.array([1.0, 2.0, 3.0, 4.0])
        ratio, var = ratio_mean_and_var(
            len(y),
            float(np.sum(y)),
            float(np.sum(y * y)),
            float(np.sum(x)),
            float(np.sum(x * x)),
            float(np.sum(y * x)),
        )
        self.assertAlmostEqual(ratio, 2.0, places=12)
        self.assertAlmostEqual(var, 0.0, places=12)

        p_eq, z_eq, se_eq = delta_ratio_z_test(2.0, 0.0, 2.0, 0.0)
        self.assertEqual((p_eq, z_eq, se_eq), (1.0, 0.0, 0.0))

        p_neq, z_neq, se_neq = delta_ratio_z_test(2.0, 0.0, 3.0, 0.0)
        self.assertEqual((p_neq, z_neq, se_neq), (0.0, None, 0.0))

    def test_validate_ratio_aggregates_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least 2 visitors"):
            validate_ratio_aggregates(1, 1.0, 1.0, 1.0, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "denominator_sum must be positive"):
            validate_ratio_aggregates(10, 5.0, 10.0, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "denominator_sum must be positive"):
            validate_ratio_aggregates(10, 5.0, 10.0, -2.0, 5.0, -1.0)
        with self.assertRaisesRegex(ValueError, "finite number"):
            validate_ratio_aggregates(10, float("nan"), 10.0, 5.0, 10.0, 5.0)
        with self.assertRaisesRegex(ValueError, "numerator_sum_squares"):
            validate_ratio_aggregates(10, 10.0, 5.0, 10.0, 20.0, 10.0)
        with self.assertRaisesRegex(ValueError, "denominator_sum_squares"):
            validate_ratio_aggregates(10, 10.0, 20.0, 10.0, 5.0, 10.0)
        with self.assertRaisesRegex(ValueError, "Cauchy-Schwarz"):
            # s_yy = 10, s_xx = 10 -> |s_xy| <= 10; set num_den_sum so s_xy = 50
            validate_ratio_aggregates(10, 10.0, 20.0, 10.0, 20.0, 60.0)


class RatioEngineIntegrationTests(unittest.TestCase):
    def _build_payload(self, method: str = "fixed", n_treatments: int = 1) -> dict:
        rng = np.random.default_rng(42)
        y_c, x_c = _simulate_ctr_arm(rng, 2500, 0.10)
        variants = [_agg_from_pairs("control", y_c, x_c, is_control=True)]
        for i in range(n_treatments):
            y_t, x_t = _simulate_ctr_arm(rng, 2500, 0.15 + 0.01 * i)
            variants.append(_agg_from_pairs(f"treatment_{i + 1}", y_t, x_t, is_control=False))
        return {
            "experiment_name": "ctr_ratio_test",
            "method": method,
            "primary_metric": "ratio",
            "alpha": 0.05,
            "samples": 15000,
            "random_seed": 7,
            "variants": variants,
        }

    def test_all_three_methods_detect_true_ratio_lift_and_validate_schemas(self):
        in_schema = get_schema("analyze-input")
        out_schema = get_schema("analyze-output")

        for method in ("fixed", "sequential", "bayesian"):
            payload = self._build_payload(method=method)
            if method == "bayesian":
                payload["decision_policy"] = {
                    "enabled": True,
                    "bayes_prob_beats_control": 0.95,
                    "max_expected_loss": 0.005,
                }
            self.assertEqual(validate_json_schema(payload, in_schema), [])
            result = analyze(parse_payload(payload))
            result_dict = result.to_dict()
            self.assertEqual(validate_json_schema(result_dict, out_schema), [])

            comp = result.comparisons[0]
            self.assertEqual(comp.metric, "ratio")
            self.assertGreater(comp.absolute_lift, 0.0)
            self.assertIsNotNone(comp.ci_low)
            self.assertIsNotNone(comp.ci_high)
            self.assertGreater(comp.ci_low, 0.0)

            if method == "fixed":
                self.assertTrue(comp.significant)
                self.assertIsNotNone(comp.test_statistic)
                self.assertIsNone(comp.degrees_of_freedom)
                self.assertLess(comp.p_value, 0.01)
            elif method == "sequential":
                self.assertTrue(comp.significant)
                self.assertGreater(comp.e_value, comp.e_threshold)
            else:
                self.assertEqual(
                    result.analysis_settings.priors, {"family": "delta_normal_asymptotic"}
                )
                self.assertGreater(comp.probability_beats_control, 0.99)
                self.assertIsNotNone(result.recommendation)
                self.assertEqual(result.recommendation.action, "ship_treatment_1")

            report = build_markdown_report(result)
            self.assertIn("Control Ratio:", report)
            self.assertIn("Treatment Ratio:", report)
            if method == "fixed":
                self.assertIn("z-statistic:", report)

    def test_multi_arm_ratio_bonferroni_and_joint_bayesian(self):
        fixed_res = analyze(parse_payload(self._build_payload(method="fixed", n_treatments=2)))
        self.assertEqual(len(fixed_res.comparisons), 2)
        self.assertEqual(fixed_res.analysis_settings.multiple_comparisons["n_comparisons"], 2)
        for comp in fixed_res.comparisons:
            self.assertAlmostEqual(comp.p_value, min(1.0, comp.raw_p_value * 2.0), places=12)

        bayes_res = analyze(parse_payload(self._build_payload(method="bayesian", n_treatments=2)))
        self.assertIsNotNone(bayes_res.bayesian_arms)
        self.assertEqual(len(bayes_res.bayesian_arms), 3)
        self.assertAlmostEqual(
            sum(a.probability_best for a in bayes_res.bayesian_arms), 1.0, places=6
        )

    def test_ratio_as_statistical_guardrail_and_secondary_metric(self):
        rng = np.random.default_rng(123)
        y_c, x_c = _simulate_ctr_arm(rng, 1500, 0.20)
        y_t_bad, x_t_bad = _simulate_ctr_arm(rng, 1500, 0.14)
        y_sec_c, x_sec_c = _simulate_ctr_arm(rng, 1500, 0.10)
        y_sec_t, x_sec_t = _simulate_ctr_arm(rng, 1500, 0.11)

        payload = {
            "experiment_name": "primary_cr_ratio_submetrics",
            "method": "fixed",
            "primary_metric": "conversion_rate",
            "variants": [
                {"name": "control", "visitors": 1500, "conversions": 150, "is_control": True},
                {"name": "treatment", "visitors": 1500, "conversions": 200, "is_control": False},
            ],
            "guardrails": [
                {
                    "name": "ad_ctr",
                    "metric": "ratio",
                    "direction": "increase",
                    "max_relative_change": 0.05,
                    "variants": [
                        {
                            k: v
                            for k, v in _agg_from_pairs(
                                "control", y_c, x_c, is_control=True
                            ).items()
                            if k != "is_control"
                        },
                        {
                            k: v
                            for k, v in _agg_from_pairs(
                                "treatment", y_t_bad, x_t_bad, is_control=False
                            ).items()
                            if k != "is_control"
                        },
                    ],
                }
            ],
            "secondary_metrics": [
                {
                    "name": "session_ctr",
                    "metric": "ratio",
                    "variants": [
                        {
                            k: v
                            for k, v in _agg_from_pairs(
                                "control", y_sec_c, x_sec_c, is_control=True
                            ).items()
                            if k != "is_control"
                        },
                        {
                            k: v
                            for k, v in _agg_from_pairs(
                                "treatment", y_sec_t, x_sec_t, is_control=False
                            ).items()
                            if k != "is_control"
                        },
                    ],
                }
            ],
        }
        self.assertEqual(validate_json_schema(payload, get_schema("analyze-input")), [])
        result = analyze(parse_payload(payload))
        self.assertEqual(validate_json_schema(result.to_dict(), get_schema("analyze-output")), [])

        self.assertFalse(result.guardrails_passed)
        self.assertEqual(result.guardrails[0].mode, "statistical")
        self.assertEqual(result.guardrails[0].comparison.metric, "ratio")
        self.assertEqual(result.recommendation.action, "do_not_ship")
        self.assertEqual(len(result.secondary_metrics), 1)
        self.assertEqual(result.secondary_metrics[0].metric, "ratio")
        self.assertEqual(result.secondary_metrics[0].comparisons[0].metric, "ratio")

    def test_missing_ratio_fields_rejected(self):
        payload = {
            "experiment_name": "missing_ratio_fields",
            "method": "fixed",
            "primary_metric": "ratio",
            "variants": [
                {"name": "control", "visitors": 100, "numerator_sum": 10.0, "is_control": True},
                {"name": "treatment", "visitors": 100, "numerator_sum": 12.0, "is_control": False},
            ],
        }
        with self.assertRaisesRegex(ValueError, "requires"):
            analyze(parse_payload(payload))


class RatioConnectorsAndCliTests(unittest.TestCase):
    def test_row_level_csv_matches_aggregated_json_exactly(self):
        rng = np.random.default_rng(99)
        y_c, x_c = _simulate_ctr_arm(rng, 200, 0.10)
        y_t, x_t = _simulate_ctr_arm(rng, 200, 0.14)

        rows = [
            {"variant": "control", "clicks": str(y), "impressions": str(x), "is_control": "true"}
            for y, x in zip(y_c, x_c, strict=True)
        ] + [
            {"variant": "treatment", "clicks": str(y), "impressions": str(x), "is_control": "false"}
            for y, x in zip(y_t, x_t, strict=True)
        ]

        row_payload = build_payload_from_rows(
            rows, defaults={"primary_metric": "ratio", "method": "fixed"}
        )
        self.assertEqual(row_payload["input_interpretation"]["source_type"], "row_level")
        self.assertEqual(
            row_payload["input_interpretation"]["column_roles"],
            {
                "variant": "variant",
                "numerator": "clicks",
                "denominator": "impressions",
                "is_control": "is_control",
            },
        )
        row_res = analyze(parse_payload(row_payload))

        json_payload = {
            "experiment_name": "table_input_experiment",
            "method": "fixed",
            "primary_metric": "ratio",
            "variants": [
                _agg_from_pairs("control", y_c, x_c, is_control=True),
                _agg_from_pairs("treatment", y_t, x_t, is_control=False),
            ],
        }
        json_res = analyze(parse_payload(json_payload))

        self.assertAlmostEqual(
            row_res.comparisons[0].control_rate, json_res.comparisons[0].control_rate, places=12
        )
        self.assertAlmostEqual(
            row_res.comparisons[0].treatment_rate, json_res.comparisons[0].treatment_rate, places=12
        )
        self.assertAlmostEqual(
            row_res.comparisons[0].p_value, json_res.comparisons[0].p_value, places=12
        )
        self.assertAlmostEqual(
            row_res.comparisons[0].ci_low, json_res.comparisons[0].ci_low, places=12
        )
        self.assertAlmostEqual(
            row_res.comparisons[0].ci_high, json_res.comparisons[0].ci_high, places=12
        )

    def test_aggregated_ratio_csv_auto_detects_ratio_metric(self):
        rows = [
            {
                "variant": "control",
                "visitors": "100",
                "num_sum": "50",
                "num_sq": "120",
                "den_sum": "500",
                "den_sq": "3500",
                "num_den": "400",
            },
            {
                "variant": "treatment",
                "visitors": "100",
                "num_sum": "65",
                "num_sq": "150",
                "den_sum": "510",
                "den_sq": "3600",
                "num_den": "450",
            },
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual(payload["primary_metric"], "ratio")
        self.assertEqual(payload["input_interpretation"]["source_type"], "aggregated")
        res = analyze(parse_payload({**payload, "method": "fixed"}))
        self.assertAlmostEqual(res.comparisons[0].control_rate, 0.1, places=12)
        self.assertAlmostEqual(res.comparisons[0].treatment_rate, 65.0 / 510.0, places=12)

    def test_cli_row_level_ratio_csv_and_negative_denominator_error(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ctr.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["variant", "clicks", "impressions"])
                for _ in range(30):
                    writer.writerow(["control", 1, 10])
                    writer.writerow(["control", 2, 12])
                    writer.writerow(["treatment", 3, 10])
                    writer.writerow(["treatment", 4, 12])

            res = runner.invoke(
                cli,
                [
                    "analyze",
                    "--input",
                    str(csv_path),
                    "--primary-metric",
                    "ratio",
                    "--method",
                    "fixed",
                ],
            )
            self.assertEqual(res.exit_code, 0, res.output)
            data = json.loads(res.output)
            self.assertEqual(data["analysis_settings"]["primary_metric"], "ratio")
            self.assertTrue(data["comparisons"][0]["significant"])

            bad_path = Path(tmp) / "bad.csv"
            bad_path.write_text(
                "variant,clicks,impressions\ncontrol,1,-5\ncontrol,2,5\ntreatment,1,5\ntreatment,2,5\n",
                encoding="utf-8",
            )
            bad_res = runner.invoke(
                cli,
                ["analyze", "--input", str(bad_path), "--primary-metric", "ratio"],
            )
            self.assertNotEqual(bad_res.exit_code, 0)
            self.assertIn("cannot be negative", bad_res.output)

    def test_ratio_connector_rejects_wrong_column_counts(self):
        dup_wrong = [
            {"v": "control", "a": "1", "b": "2", "c": "3"},
            {"v": "control", "a": "1", "b": "2", "c": "3"},
        ]
        with self.assertRaisesRegex(ValueError, "Row-level ratio input"):
            build_payload_from_rows(dup_wrong, defaults={"primary_metric": "ratio"})

        agg_wrong = [
            {"v": "control", "n": "100", "c": "10"},
            {"v": "treatment", "n": "100", "c": "12"},
        ]
        with self.assertRaisesRegex(ValueError, "primary_metric 'ratio' needs"):
            build_payload_from_rows(agg_wrong, defaults={"primary_metric": "ratio"})


if __name__ == "__main__":
    unittest.main()
