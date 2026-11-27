import json
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from ablift.cli import cli

JSON_PAYLOAD = {
    "experiment_name": "stdin_json_exp",
    "method": "bayesian",
    "samples": 2000,
    "random_seed": 7,
    "variants": [
        {"name": "control", "visitors": 1000, "conversions": 40, "is_control": True},
        {"name": "v1", "visitors": 1000, "conversions": 52, "is_control": False},
    ],
}

CSV_AGGREGATED = "variant,visitors,conversions\ncontrol,1000,40\nv1,1000,52\n"

CSV_AGGREGATED_ARPU = (
    "variant,visitors,revenue_sum,revenue_sum_squares\n"
    "control,1000,5000,400000\n"
    "v1,1000,5500,450000\n"
)

CSV_ROW_LEVEL_CONVERSION = (
    "variant,converted\ncontrol,1\ncontrol,0\ncontrol,0\ncontrol,1\nv1,1\nv1,1\nv1,0\nv1,1\n"
)

CSV_ROW_LEVEL_ARPU = (
    "variant,revenue\ncontrol,0\ncontrol,10.5\ncontrol,20.0\nv1,0\nv1,15.0\nv1,30.0\n"
)

DURATION_CSV = (
    "method,baseline_rate,relative_mde,daily_traffic,n_variants\nsequential,0.04,0.05,50000,3\n"
)

DURATION_JSON = {
    "method": "sequential",
    "baseline_rate": 0.04,
    "relative_mde": 0.05,
    "daily_traffic": 50000,
    "n_variants": 3,
}


class StdinCliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def assert_clean_error(self, result, code=1):
        self.assertEqual(result.exit_code, code, result.output)
        self.assertIn("Error", result.output)
        self.assertNotIn("Traceback", result.output)
        self.assertIsInstance(result.exception, SystemExit, repr(result.exception))

    def test_analyze_stdin_json_matches_file(self):
        raw = json.dumps(JSON_PAYLOAD)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.json"
            path.write_text(raw, encoding="utf-8")
            from_file = self.runner.invoke(cli, ["analyze", "--input", str(path)])
        from_stdin = self.runner.invoke(cli, ["analyze", "--input", "-"], input=raw)

        self.assertEqual(from_file.exit_code, 0, from_file.output)
        self.assertEqual(from_stdin.exit_code, 0, from_stdin.output)
        self.assertEqual(json.loads(from_stdin.output), json.loads(from_file.output))

    def test_analyze_stdin_json_explicit_format_and_bom(self):
        raw = "\ufeff" + json.dumps(JSON_PAYLOAD)
        result = self.runner.invoke(
            cli,
            ["analyze", "--input", "-", "--input-format", "json", "--method", "sequential"],
            input=raw,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["experiment_name"], "stdin_json_exp")
        self.assertEqual(data["method"], "sequential")

    def test_analyze_example_input_piped_to_stdin(self):
        example = self.runner.invoke(cli, ["example-input"])
        self.assertEqual(example.exit_code, 0, example.output)
        result = self.runner.invoke(
            cli, ["analyze", "--input", "-", "--samples", "2000"], input=example.output
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["experiment_name"], "checkout_button_color")
        self.assertIsNotNone(data["recommendation"])

    def test_analyze_stdin_csv_aggregated_defaults_name_to_stdin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stdin.csv"
            path.write_text(CSV_AGGREGATED, encoding="utf-8")
            from_file = self.runner.invoke(
                cli, ["analyze", "--input", str(path), "--samples", "2000"]
            )
        from_stdin = self.runner.invoke(
            cli,
            ["analyze", "--input", "-", "--input-format", "csv", "--samples", "2000"],
            input=CSV_AGGREGATED,
        )

        self.assertEqual(from_file.exit_code, 0, from_file.output)
        self.assertEqual(from_stdin.exit_code, 0, from_stdin.output)
        data = json.loads(from_stdin.output)
        self.assertEqual(data["experiment_name"], "stdin")
        self.assertEqual(data, json.loads(from_file.output))

    def test_analyze_stdin_csv_experiment_name_override(self):
        result = self.runner.invoke(
            cli,
            [
                "analyze",
                "--input",
                "-",
                "--input-format",
                "csv",
                "--experiment-name",
                "checkout_slice",
                "--samples",
                "2000",
            ],
            input="\ufeff" + CSV_AGGREGATED,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["experiment_name"], "checkout_slice")

    def test_analyze_stdin_csv_aggregated_arpu(self):
        result = self.runner.invoke(
            cli,
            ["analyze", "--input", "-", "--input-format", "csv", "--samples", "2000"],
            input=CSV_AGGREGATED_ARPU,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["experiment_name"], "stdin")
        self.assertEqual(data["analysis_settings"]["primary_metric"], "arpu")
        self.assertEqual(
            data["analysis_settings"]["input_interpretation"]["source_type"], "aggregated"
        )

    def test_analyze_stdin_csv_row_level_conversion(self):
        result = self.runner.invoke(
            cli,
            ["analyze", "--input", "-", "--input-format", "csv", "--samples", "2000"],
            input=CSV_ROW_LEVEL_CONVERSION,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["experiment_name"], "stdin")
        self.assertEqual(data["analysis_settings"]["primary_metric"], "conversion_rate")
        self.assertEqual(
            data["analysis_settings"]["input_interpretation"]["source_type"], "row_level"
        )

    def test_analyze_stdin_csv_row_level_arpu(self):
        result = self.runner.invoke(
            cli,
            ["analyze", "--input", "-", "--input-format", "csv", "--samples", "2000"],
            input=CSV_ROW_LEVEL_ARPU,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["experiment_name"], "stdin")
        self.assertEqual(data["analysis_settings"]["primary_metric"], "arpu")
        self.assertEqual(
            data["analysis_settings"]["input_interpretation"]["source_type"], "row_level"
        )

    def test_analyze_input_format_overrides_file_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt_json = Path(tmp) / "payload.txt"
            txt_json.write_text(json.dumps(JSON_PAYLOAD), encoding="utf-8")
            res_json = self.runner.invoke(
                cli, ["analyze", "--input", str(txt_json), "--input-format", "json"]
            )
            self.assertEqual(res_json.exit_code, 0, res_json.output)
            self.assertEqual(json.loads(res_json.output)["experiment_name"], "stdin_json_exp")

            txt_csv = Path(tmp) / "table.dat"
            txt_csv.write_text(CSV_AGGREGATED, encoding="utf-8")
            res_csv = self.runner.invoke(
                cli,
                ["analyze", "--input", str(txt_csv), "--input-format", "csv", "--samples", "2000"],
            )
            self.assertEqual(res_csv.exit_code, 0, res_csv.output)
            self.assertEqual(json.loads(res_csv.output)["experiment_name"], "table")

    def test_duration_stdin_csv_and_json_match_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dur.csv"
            path.write_text(DURATION_CSV, encoding="utf-8")
            from_file = self.runner.invoke(cli, ["duration", "--input", str(path)])

        from_csv_auto = self.runner.invoke(cli, ["duration", "--input", "-"], input=DURATION_CSV)
        from_csv_explicit = self.runner.invoke(
            cli, ["duration", "--input", "-", "--input-format", "csv"], input=DURATION_CSV
        )
        from_json_auto = self.runner.invoke(
            cli,
            ["duration", "--input", "-"],
            input="\n  " + json.dumps(DURATION_JSON) + "\n",
        )
        from_json_explicit = self.runner.invoke(
            cli,
            ["duration", "--input", "-", "--input-format", "json"],
            input=json.dumps(DURATION_JSON),
        )

        expected = json.loads(from_file.output)
        for label, res in (
            ("csv_auto", from_csv_auto),
            ("csv_explicit", from_csv_explicit),
            ("json_auto", from_json_auto),
            ("json_explicit", from_json_explicit),
        ):
            with self.subTest(case=label):
                self.assertEqual(res.exit_code, 0, res.output)
                self.assertEqual(json.loads(res.output), expected)

    def test_duration_stdin_json_flag_override_and_unused_setting_validation(self):
        raw = json.dumps({**DURATION_JSON, "method": "fixed"})
        overridden = self.runner.invoke(
            cli,
            ["duration", "--input", "-", "--method", "sequential", "--baseline-rate", "0.1"],
            input=raw,
        )
        self.assertEqual(overridden.exit_code, 0, overridden.output)
        data = json.loads(overridden.output)
        self.assertEqual(data["method"], "sequential")
        self.assertEqual(data["assumptions"]["baseline_rate"], 0.1)

        bad_combo = json.dumps(
            {
                "method": "bayesian",
                "baseline_rate": 0.04,
                "relative_mde": 0.08,
                "daily_traffic": 60000,
                "alpha": 0.05,
            }
        )
        rejected = self.runner.invoke(cli, ["duration", "--input", "-"], input=bad_combo)
        self.assertEqual(rejected.exit_code, 2, rejected.output)
        self.assertIn("--alpha", rejected.output)

    def test_duration_json_file_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assumptions.json"
            path.write_text(json.dumps(DURATION_JSON), encoding="utf-8")
            result = self.runner.invoke(cli, ["duration", "--input", str(path)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["method"], "sequential")

    def test_empty_and_whitespace_stdin_clean_error(self):
        cases = [
            ["analyze", "--input", "-"],
            ["analyze", "--input", "-", "--input-format", "csv"],
            ["duration", "--input", "-"],
            ["duration", "--input", "-", "--input-format", "json"],
            ["duration", "--input", "-", "--input-format", "csv"],
        ]
        for args in cases:
            for stdin_text in ("", "   \n\t  "):
                with self.subTest(args=args, stdin=repr(stdin_text)):
                    res = self.runner.invoke(cli, args, input=stdin_text)
                    self.assert_clean_error(res, code=1)
                    self.assertIn("empty", res.output.lower())

    def test_malformed_stdin_clean_errors(self):
        cases = [
            (["analyze", "--input", "-"], "{not valid json"),
            (["analyze", "--input", "-"], "[1, 2, 3]"),
            (["analyze", "--input", "-", "--input-format", "csv"], "a,b,c\n"),
            (
                ["analyze", "--input", "-", "--input-format", "csv"],
                "name,visitors\ncontrol,10\nv1,10\n",
            ),
            (["duration", "--input", "-"], "{not valid json"),
            (["duration", "--input", "-", "--input-format", "json"], "[1, 2]"),
            (["duration", "--input", "-"], '{"baseline_rate": 0.04, "bogus": 1}'),
            (
                ["duration", "--input", "-"],
                '{"baseline_rate": true, "relative_mde": 0.05, "daily_traffic": 1000}',
            ),
            (
                ["duration", "--input", "-"],
                '{"baseline_rate": 0.04, "relative_mde": 0.05, "daily_traffic": 100.5}',
            ),
            (
                ["duration", "--input", "-"],
                '{"baseline_rate": [0.04], "relative_mde": 0.05, "daily_traffic": 1000}',
            ),
            (
                ["duration", "--input", "-"],
                '{"method": 123, "baseline_rate": 0.04, "relative_mde": 0.05, "daily_traffic": 1000}',
            ),
            (
                ["duration", "--input", "-", "--input-format", "csv"],
                "baseline_rate,relative_mde,daily_traffic\n",
            ),
            (
                ["duration", "--input", "-", "--input-format", "csv"],
                "baseline_rate,bogus\n0.04,1\n",
            ),
        ]
        for args, stdin_text in cases:
            with self.subTest(args=args, stdin=stdin_text):
                res = self.runner.invoke(cli, args, input=stdin_text)
                self.assert_clean_error(res, code=1)

    def test_sheet_rejected_with_stdin_and_non_excel(self):
        res_analyze_stdin = self.runner.invoke(
            cli, ["analyze", "--input", "-", "--sheet", "Sheet1"], input=json.dumps(JSON_PAYLOAD)
        )
        self.assert_clean_error(res_analyze_stdin, code=2)
        self.assertIn("--sheet", res_analyze_stdin.output)

        res_duration_stdin = self.runner.invoke(
            cli, ["duration", "--input", "-", "--sheet", "Sheet1"], input=DURATION_CSV
        )
        self.assert_clean_error(res_duration_stdin, code=2)
        self.assertIn("--sheet", res_duration_stdin.output)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "in.csv"
            csv_path.write_text(CSV_AGGREGATED, encoding="utf-8")
            res_csv_sheet = self.runner.invoke(
                cli, ["analyze", "--input", str(csv_path), "--sheet", "Sheet1"]
            )
        self.assert_clean_error(res_csv_sheet, code=2)
        self.assertIn("--sheet", res_csv_sheet.output)

        res_duration_no_input = self.runner.invoke(cli, ["duration", "--sheet", "Sheet1"])
        self.assert_clean_error(res_duration_no_input, code=2)

        res_format_no_input = self.runner.invoke(cli, ["duration", "--input-format", "json"])
        self.assert_clean_error(res_format_no_input, code=2)

    def test_missing_file_path_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist.json")
            res_analyze = self.runner.invoke(cli, ["analyze", "--input", missing])
            res_duration = self.runner.invoke(cli, ["duration", "--input", missing])
        self.assertEqual(res_analyze.exit_code, 2, res_analyze.output)
        self.assertNotIn("Traceback", res_analyze.output)
        self.assertEqual(res_duration.exit_code, 2, res_duration.output)
        self.assertNotIn("Traceback", res_duration.output)

    def test_help_documents_stdin_and_input_format(self):
        analyze_help = self.runner.invoke(cli, ["analyze", "--help"])
        self.assertEqual(analyze_help.exit_code, 0, analyze_help.output)
        self.assertIn("--input-format", analyze_help.output)
        self.assertIn("--input -", analyze_help.output)

        duration_help = self.runner.invoke(cli, ["duration", "--help"])
        self.assertEqual(duration_help.exit_code, 0, duration_help.output)
        self.assertIn("--input-format", duration_help.output)
        self.assertIn("--input -", duration_help.output)


if __name__ == "__main__":
    unittest.main()
