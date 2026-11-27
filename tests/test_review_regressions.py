"""Regression tests for issues found in the PR #7 review."""

import unittest

from ablift.connectors import build_payload_from_rows, detect_input_shape
from ablift.engine import analyze, parse_payload


def rows(header, *data):
    return [dict(zip(header, r, strict=True)) for r in data]


class DuplicateAggregatedRows(unittest.TestCase):
    def test_aggregated_table_with_repeated_names_is_rejected_not_read_as_row_level(self):
        table = rows(
            ["name", "visitors", "conversions"],
            ["A", "1000", "100"],
            ["A", "1000", "90"],
            ["B", "1000", "120"],
        )
        with self.assertRaisesRegex(ValueError, "appear in more than one row"):
            detect_input_shape(table)

    def test_row_level_with_control_column_still_detected(self):
        table = rows(
            ["v", "converted", "is_control"],
            ["a", "1", "true"],
            ["a", "0", "true"],
            ["b", "1", "false"],
            ["b", "1", "false"],
        )
        self.assertEqual(detect_input_shape(table), "row_level")
        payload = build_payload_from_rows(table)
        self.assertTrue(payload["variants"][0]["is_control"])


class ControlColumnHeaderHint(unittest.TestCase):
    def test_all_false_control_column_errors_instead_of_becoming_revenue(self):
        table = rows(
            ["name", "visitors", "conversions", "is_control"],
            ["A", "1000", "100", "0"],
            ["B", "1000", "120", "0"],
        )
        payload = build_payload_from_rows(table)
        self.assertEqual(payload["primary_metric"], "conversion_rate")
        self.assertEqual(
            payload["input_interpretation"]["column_roles"]["is_control"], "is_control"
        )
        with self.assertRaisesRegex(ValueError, "Exactly one variant"):
            analyze(parse_payload(payload))

    def test_binary_revenue_sum_squares_is_not_mistaken_for_control(self):
        # Codex review: a 0/1 revenue_sum_squares column must stay a metric.
        table = rows(
            ["name", "visitors", "orders", "rev", "rev_sq"],
            ["A", "1000", "1", "0", "0"],
            ["B", "1000", "1", "1", "1"],
        )
        payload = build_payload_from_rows(table)
        roles = payload["input_interpretation"]["column_roles"]
        self.assertEqual(roles.get("revenue_sum_squares"), "rev_sq")
        self.assertNotIn("is_control", roles)
        self.assertTrue(payload["variants"][0]["is_control"])

    def test_unnamed_word_control_column_still_detected(self):
        table = rows(
            ["name", "visitors", "orders", "flag"],
            ["A", "1000", "40", "no"],
            ["B", "1000", "45", "yes"],
        )
        payload = build_payload_from_rows(table)
        self.assertTrue(payload["variants"][1]["is_control"])


class AdjustedSequentialPValue(unittest.TestCase):
    def test_p_value_is_bonferroni_adjusted_and_consistent_with_significance(self):
        for conv in (400, 440, 460, 480, 520):
            payload = {
                "experiment_name": "x",
                "method": "sequential",
                "variants": [
                    {"name": "c", "visitors": 20000, "conversions": 400, "is_control": True},
                    {"name": "v1", "visitors": 20000, "conversions": conv},
                    {"name": "v2", "visitors": 20000, "conversions": 400},
                ],
            }
            res = analyze(parse_payload(payload))
            for comp in res.comparisons:
                self.assertAlmostEqual(comp.p_value, min(1.0, 2.0 / comp.e_value))
                self.assertEqual(comp.p_value <= 0.05, comp.significant, conv)
                self.assertGreaterEqual(res.recommendation.decision_confidence, 0.0)
                self.assertLessEqual(res.recommendation.decision_confidence, 1.0)


class CodexReviewFindings(unittest.TestCase):
    def test_decimal_binary_outcomes_count_as_successes(self):
        table = rows(
            ["v", "converted"],
            ["a", "1.0"],
            ["a", "0.0"],
            ["a", "1.0"],
            ["b", "1.0"],
            ["b", "0.0"],
        )
        payload = build_payload_from_rows(table)
        self.assertEqual(payload["primary_metric"], "conversion_rate")
        variants = {v["name"]: v for v in payload["variants"]}
        self.assertEqual(variants["a"]["conversions"], 2)
        self.assertEqual(variants["b"]["conversions"], 1)

    def test_xlsx_float_outcomes_count_as_successes(self):
        table = [
            {"v": "a", "y": 1.0},
            {"v": "a", "y": 0.0},
            {"v": "b", "y": 1.0},
            {"v": "b", "y": 1},
        ]
        variants = {v["name"]: v for v in build_payload_from_rows(table)["variants"]}
        self.assertEqual(variants["a"]["conversions"], 1)
        self.assertEqual(variants["b"]["conversions"], 2)

    def test_conversion_rate_json_requires_conversions(self):
        payload = {
            "experiment_name": "x",
            "method": "bayesian",
            "variants": [
                {"name": "c", "visitors": 1000, "conversions": 40, "is_control": True},
                {"name": "v1", "visitors": 1000},
            ],
        }
        with self.assertRaisesRegex(ValueError, "requires 'conversions'.*v1"):
            parse_payload(payload)

    def test_arpu_json_may_omit_conversions(self):
        payload = {
            "experiment_name": "x",
            "method": "bayesian",
            "primary_metric": "arpu",
            "samples": 2000,
            "variants": [
                {
                    "name": "c",
                    "visitors": 100,
                    "revenue_sum": 500.0,
                    "revenue_sum_squares": 9000.0,
                    "is_control": True,
                },
                {
                    "name": "v1",
                    "visitors": 100,
                    "revenue_sum": 550.0,
                    "revenue_sum_squares": 9900.0,
                },
            ],
        }
        self.assertEqual(analyze(parse_payload(payload)).comparisons[0].metric, "arpu")

    def test_fractional_counts_are_rejected_not_truncated(self):
        from ablift.connectors import build_duration_request_from_rows

        with self.assertRaisesRegex(ValueError, "n_variants.*whole number"):
            build_duration_request_from_rows(
                [{"baseline_rate": "0.04", "relative_mde": "0.05", "n_variants": "2.5"}]
            )
        req = build_duration_request_from_rows([{"daily_traffic": "50000.0", "max_days": 30}])
        self.assertEqual((req["daily_traffic"], req["max_days"]), (50000, 30))
        with self.assertRaisesRegex(ValueError, "visitors.*whole number"):
            build_payload_from_rows(
                rows(["name", "visitors", "orders"], ["A", "1000.5", "40"], ["B", "1000", "45"])
            )


class PayloadAndCliCorrectnessTests(unittest.TestCase):
    def test_unknown_top_level_payload_key_is_rejected(self):
        payload = {
            "experiment_name": "typo",
            "primaryMetric": "arpu",
            "variants": [
                {"name": "c", "visitors": 1000, "conversions": 40, "is_control": True},
                {"name": "v1", "visitors": 1000, "conversions": 50},
            ],
        }
        with self.assertRaisesRegex(ValueError, "Unknown top-level field.*primaryMetric"):
            parse_payload(payload)

    def test_bayesian_disabled_policy_does_not_leak_into_sequential_or_fixed(self):
        for method in ("sequential", "fixed"):
            with self.subTest(method=method):
                payload = {
                    "experiment_name": "leak_check",
                    "method": method,
                    "decision_policy": {"enabled": False},
                    "variants": [
                        {"name": "c", "visitors": 30000, "conversions": 1200, "is_control": True},
                        {"name": "v1", "visitors": 30000, "conversions": 1410},
                    ],
                }
                result = analyze(parse_payload(payload))
                self.assertTrue(result.analysis_settings.decision_policy["enabled"])
                self.assertEqual(
                    result.analysis_settings.decision_policy["source"],
                    "implicit_method_defaults",
                )
                self.assertIsNotNone(result.recommendation)

    def test_output_help_and_schema_epilog_are_populated(self):
        from click.testing import CliRunner

        from ablift.cli import cli

        runner = CliRunner()
        for cmd in ("duration", "power"):
            res = runner.invoke(cli, [cmd, "--help"])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("Path to output JSON.", res.output)
        schema_help = runner.invoke(cli, ["schema", "--help"])
        self.assertEqual(schema_help.exit_code, 0, schema_help.output)
        self.assertIn("power-output", schema_help.output)


if __name__ == "__main__":
    unittest.main()
