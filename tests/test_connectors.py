import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from bayestest.connectors import (
    build_duration_request_from_rows,
    build_payload_from_rows,
    detect_primary_metric,
    read_table,
)


class ConnectorTests(unittest.TestCase):
    def test_read_csv_and_build_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "exp.csv"
            csv_path.write_text(
                "variant_name,users,orders,is_control\ncontrol,1000,40,true\nv1,1000,45,false\n",
                encoding="utf-8",
            )

            rows = read_table(str(csv_path))
            mapping = {
                "experiment_name": "exp",
                "method": "bayesian",
                "primary_metric": "conversion_rate",
                "columns": {
                    "variant": "variant_name",
                    "visitors": "users",
                    "conversions": "orders",
                    "is_control": "is_control",
                },
            }

            payload = build_payload_from_rows(rows, mapping)
            self.assertEqual(payload["variants"][0]["name"], "control")
            self.assertTrue(payload["variants"][0]["is_control"])
            self.assertEqual(payload["variants"][1]["conversions"], 45)

    def test_read_xlsx(self):
        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "exp.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Sheet1"
            ws.append(["variant_name", "users", "orders", "is_control"])
            ws.append(["control", 1000, 40, True])
            ws.append(["v1", 1000, 44, False])
            wb.save(xlsx_path)

            rows = read_table(str(xlsx_path), sheet="Sheet1")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["variant_name"], "control")

    def test_build_duration_request_from_rows(self):
        rows = [
            {
                "method": "bayesian",
                "baseline_rate": "0.04",
                "relative_mde": "0.05",
                "daily_traffic": "50000",
                "n_variants": "3",
                "alpha": "0.05",
                "power": "0.8",
                "max_looks": "10",
                "prob_threshold": "0.95",
                "max_expected_loss": "0.001",
                "assurance_target": "0.8",
                "max_days": "30",
            }
        ]
        mapping = {
            "columns": {
                "method": "method",
                "baseline_rate": "baseline_rate",
                "relative_mde": "relative_mde",
                "daily_traffic": "daily_traffic",
                "n_variants": "n_variants",
                "alpha": "alpha",
                "power": "power",
                "max_looks": "max_looks",
                "prob_threshold": "prob_threshold",
                "max_expected_loss": "max_expected_loss",
                "assurance_target": "assurance_target",
                "max_days": "max_days",
            }
        }
        req = build_duration_request_from_rows(rows, mapping)
        self.assertEqual(req["method"], "bayesian")
        self.assertEqual(req["n_variants"], 3)
        self.assertEqual(req["max_days"], 30)

    def test_build_payload_without_mapping_uses_aliases(self):
        rows = [
            {"variant_name": "control", "sessions": "1000", "orders": "40"},
            {"variant_name": "v1", "sessions": "1000", "orders": "45"},
        ]

        payload = build_payload_from_rows(rows)
        self.assertEqual(payload["experiment_name"], "table_input_experiment")
        self.assertEqual(payload["method"], "bayesian")
        self.assertTrue(payload["variants"][0]["is_control"])
        self.assertEqual(payload["variants"][1]["conversions"], 45)

    def test_detect_primary_metric_from_revenue_columns(self):
        rows = [
            {
                "variant_name": "control",
                "users": "1000",
                "orders": "40",
                "revenue_sum": "5000",
                "revenue_sum_squares": "40000",
            }
        ]
        self.assertEqual(detect_primary_metric(rows), "arpu")


if __name__ == "__main__":
    unittest.main()
