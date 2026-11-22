"""Regression tests for the 0.5.1 fixes: confidence semantics, flag validation, column order."""

import json
import unittest

from click.testing import CliRunner

from ablift.cli import cli
from ablift.connectors import build_payload_from_rows, detect_input_shape
from ablift.engine import analyze, parse_payload


def rows(header, *data):
    return [dict(zip(header, r, strict=True)) for r in data]


def sequential(c_conv, t_conv, n=10_000):
    return analyze(
        parse_payload(
            {
                "method": "sequential",
                "primary_metric": "conversion_rate",
                "variants": [
                    {"name": "control", "visitors": n, "conversions": c_conv, "is_control": True},
                    {"name": "v1", "visitors": n, "conversions": t_conv},
                ],
            }
        )
    )


class SequentialContinueConfidence(unittest.TestCase):
    def test_confidence_rises_with_evidence_and_matches_one_minus_p(self):
        weak = sequential(500, 502)
        stronger = sequential(500, 560)
        for res in (weak, stronger):
            self.assertEqual(res.recommendation.action, "continue_collecting_data")
            self.assertAlmostEqual(
                res.recommendation.decision_confidence, 1.0 - res.comparisons[0].p_value
            )
        self.assertGreater(
            stronger.recommendation.decision_confidence, weak.recommendation.decision_confidence
        )

    def test_continue_confidence_stays_below_ship_threshold(self):
        res = sequential(500, 560)
        self.assertLess(res.recommendation.decision_confidence, 0.95)


class DurationIgnoredFlags(unittest.TestCase):
    base = ["--baseline-rate", "0.04", "--relative-mde", "0.08", "--daily-traffic", "60000"]

    def invoke(self, *args):
        return CliRunner().invoke(cli, ["duration", *self.base, *args])

    def test_bayesian_rejects_alpha_and_power(self):
        res = self.invoke("--method", "bayesian", "--alpha", "0.01", "--power", "0.9")
        self.assertEqual(res.exit_code, 2, res.output)
        self.assertIn("--alpha", res.output)
        self.assertIn("--power", res.output)

    def test_frequentist_methods_reject_bayesian_settings(self):
        for method in ("sequential", "fixed"):
            res = self.invoke("--method", method, "--prob-threshold", "0.9")
            self.assertEqual(res.exit_code, 2, res.output)
            self.assertIn("--prob-threshold", res.output)

    def test_fixed_rejects_sequential_tau(self):
        res = self.invoke("--method", "fixed", "--sequential-tau", "0.004")
        self.assertEqual(res.exit_code, 2, res.output)

    def test_file_column_not_used_by_method_is_rejected(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("a.csv", "w") as f:
                f.write("method,baseline_rate,relative_mde,daily_traffic,alpha\n")
                f.write("bayesian,0.04,0.08,60000,0.05\n")
            res = runner.invoke(cli, ["duration", "--input", "a.csv"])
        self.assertEqual(res.exit_code, 2, res.output)
        self.assertIn("input-file column", res.output)

    def test_method_flag_overriding_file_method_ignores_other_methods_columns(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("a.csv", "w") as f:
                f.write("method,baseline_rate,relative_mde,daily_traffic,alpha,power\n")
                f.write("sequential,0.04,0.08,60000,0.05,0.8\n")
            res = runner.invoke(
                cli, ["duration", "--input", "a.csv", "--method", "bayesian", "--max-days", "5"]
            )
        self.assertEqual(res.exit_code, 0, res.output)

    def test_applicable_flags_still_accepted(self):
        res = self.invoke("--method", "sequential", "--alpha", "0.01", "--sequential-tau", "0.004")
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertEqual(json.loads(res.output)["assumptions"]["alpha"], 0.01)


class MisplacedControlColumn(unittest.TestCase):
    def test_aggregated_control_before_visitors(self):
        table = rows(
            ["name", "is_control", "visitors", "conversions"],
            ["A", "true", "1000", "100"],
            ["B", "false", "1000", "120"],
        )
        with self.assertRaisesRegex(ValueError, "must be the last column"):
            build_payload_from_rows(table)

    def test_aggregated_control_between_visitors_and_metrics_is_not_read_as_conversions(self):
        # Previously read silently as conversions=1/0 with revenue columns after it.
        table = rows(
            ["name", "visitors", "is_control", "revenue_sum", "revenue_sum_squares"],
            ["A", "1000", "1", "5000", "90000"],
            ["B", "1000", "0", "5200", "95000"],
        )
        with self.assertRaisesRegex(ValueError, "must be the last column"):
            build_payload_from_rows(table)

    def test_row_level_control_before_outcome(self):
        table = rows(
            ["name", "is_control", "converted"],
            ["A", "true", "1"],
            ["A", "true", "0"],
            ["B", "false", "1"],
            ["B", "false", "1"],
        )
        with self.assertRaisesRegex(ValueError, "must be the last column"):
            detect_input_shape(table)

    def test_metric_columns_named_control_with_binary_values_are_metrics(self):
        aggregated = [
            rows(["name", "visitors", "control_conversions"], ["A", "2", "1"], ["B", "2", "0"]),
            rows(["name", "control_visitors", "conversions"], ["A", "1", "1"], ["B", "1", "0"]),
        ]
        for table in aggregated:
            self.assertEqual(len(build_payload_from_rows(table)["variants"]), 2)
        row_level = rows(
            ["name", "control_converted"], ["A", "1"], ["A", "0"], ["B", "1"], ["B", "1"]
        )
        self.assertEqual(detect_input_shape(row_level), "row_level")

    def test_flag_like_header_variants_are_rejected(self):
        for header in ("control", "Is_Control", "control_group", "is-control"):
            table = rows(
                ["name", header, "visitors", "conversions"],
                ["A", "1", "1000", "100"],
                ["B", "0", "1000", "120"],
            )
            with self.assertRaisesRegex(ValueError, "must be the last column", msg=header):
                build_payload_from_rows(table)

    def test_control_header_with_non_binary_values_is_a_metric(self):
        # "control" in a header alone is not enough; values must be binary.
        table = rows(
            ["name", "control_visitors", "conversions"],
            ["A", "1000", "100"],
            ["B", "1000", "120"],
        )
        payload = build_payload_from_rows(table)
        self.assertEqual(payload["variants"][0]["visitors"], 1000)


if __name__ == "__main__":
    unittest.main()
