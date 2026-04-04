import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from ablift.connectors import (
    build_duration_request_from_rows,
    build_payload_from_rows,
    detect_input_shape,
    read_table,
)


class ConnectorTests(unittest.TestCase):
    def test_detect_shape_row_level(self):
        rows = [
            {"variant": "control", "converted": "1"},
            {"variant": "control", "converted": "0"},
            {"variant": "v1", "converted": "1"},
        ]
        self.assertEqual(detect_input_shape(rows), "row_level")

    def test_detect_shape_aggregated(self):
        rows = [
            {"variant": "control", "visitors": "1000", "conversions": "40"},
            {"variant": "v1", "visitors": "1000", "conversions": "45"},
        ]
        self.assertEqual(detect_input_shape(rows), "aggregated")

    def test_row_level_conversion(self):
        rows = [
            {"variant": "control", "converted": "1"},
            {"variant": "control", "converted": "0"},
            {"variant": "control", "converted": "1"},
            {"variant": "v1", "converted": "1"},
            {"variant": "v1", "converted": "1"},
            {"variant": "v1", "converted": "0"},
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual(payload["primary_metric"], "conversion_rate")
        variants = {v["name"]: v for v in payload["variants"]}
        self.assertTrue(variants["control"]["is_control"])
        self.assertFalse(variants["v1"]["is_control"])
        self.assertEqual(variants["control"]["visitors"], 3)
        self.assertEqual(variants["control"]["conversions"], 2)
        self.assertEqual(variants["v1"]["conversions"], 2)

    def test_row_level_arpu(self):
        rows = [
            {"variant": "control", "revenue": "10.5"},
            {"variant": "control", "revenue": "20.0"},
            {"variant": "v1", "revenue": "15.0"},
            {"variant": "v1", "revenue": "25.0"},
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual(payload["primary_metric"], "arpu")
        variants = {v["name"]: v for v in payload["variants"]}
        self.assertAlmostEqual(variants["control"]["revenue_sum"], 30.5)
        self.assertAlmostEqual(variants["control"]["revenue_sum_squares"], 10.5**2 + 20.0**2)
        self.assertTrue(variants["control"]["is_control"])

    def test_row_level_control_column(self):
        rows = [
            {"variant": "a", "converted": "1", "is_control": "false"},
            {"variant": "a", "converted": "0", "is_control": "false"},
            {"variant": "b", "converted": "1", "is_control": "true"},
            {"variant": "b", "converted": "1", "is_control": "true"},
        ]
        payload = build_payload_from_rows(rows)
        variants = {v["name"]: v for v in payload["variants"]}
        self.assertFalse(variants["a"]["is_control"])
        self.assertTrue(variants["b"]["is_control"])

    def test_aggregated_conversion(self):
        rows = [
            {"name": "control", "users": "1000", "orders": "40"},
            {"name": "v1", "users": "1000", "orders": "45"},
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual(payload["primary_metric"], "conversion_rate")
        variants = {v["name"]: v for v in payload["variants"]}
        self.assertTrue(variants["control"]["is_control"])
        self.assertEqual(variants["v1"]["conversions"], 45)

    def test_aggregated_arpu_approximate(self):
        rows = [
            {"variant": "control", "visitors": "1000", "revenue": "5000.50"},
            {"variant": "v1", "visitors": "1000", "revenue": "5400.00"},
        ]
        payload = build_payload_from_rows(rows)
        self.assertEqual(payload["primary_metric"], "arpu")
        self.assertTrue(payload["input_interpretation"]["arpu_approximate"])
        variants = {v["name"]: v for v in payload["variants"]}
        self.assertAlmostEqual(variants["control"]["revenue_sum"], 5000.50)

    def test_aggregated_control_column(self):
        rows = [
            {"variant": "a", "visitors": "1000", "conversions": "40", "is_control": "0"},
            {"variant": "b", "visitors": "1000", "conversions": "45", "is_control": "1"},
        ]
        payload = build_payload_from_rows(rows)
        variants = {v["name"]: v for v in payload["variants"]}
        self.assertFalse(variants["a"]["is_control"])
        self.assertTrue(variants["b"]["is_control"])

    def test_read_csv_and_build_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "exp.csv"
            csv_path.write_text(
                "variant,visitors,conversions\ncontrol,1000,40\nv1,1000,45\n",
                encoding="utf-8",
            )
            rows = read_table(str(csv_path))
            payload = build_payload_from_rows(rows)
            self.assertEqual(payload["variants"][0]["name"], "control")
            self.assertTrue(payload["variants"][0]["is_control"])
            self.assertEqual(payload["variants"][1]["conversions"], 45)

    def test_read_xlsx(self):
        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "exp.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Sheet1"
            ws.append(["variant", "visitors", "conversions"])
            ws.append(["control", 1000, 40])
            ws.append(["v1", 1000, 44])
            wb.save(xlsx_path)

            rows = read_table(str(xlsx_path), sheet="Sheet1")
            payload = build_payload_from_rows(rows)
            variants = {v["name"]: v for v in payload["variants"]}
            self.assertTrue(variants["control"]["is_control"])
            self.assertEqual(variants["v1"]["conversions"], 44)

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
        req = build_duration_request_from_rows(rows)
        self.assertEqual(req["method"], "bayesian")
        self.assertEqual(req["n_variants"], 3)
        self.assertEqual(req["max_days"], 30)


if __name__ == "__main__":
    unittest.main()
