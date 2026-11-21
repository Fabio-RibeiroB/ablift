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


if __name__ == "__main__":
    unittest.main()
