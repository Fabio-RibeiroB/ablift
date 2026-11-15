import json
import os
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from ablift.cli import cli


class AnalyzeCliTests(unittest.TestCase):
    def test_analyze_accepts_json_input(self):
        runner = CliRunner()
        payload = {
            "experiment_name": "exp",
            "method": "bayesian",
            "variants": [
                {"name": "control", "visitors": 1000, "conversions": 40, "is_control": True},
                {"name": "v1", "visitors": 1000, "conversions": 45, "is_control": False},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            result = runner.invoke(cli, ["analyze", "--input", str(input_path)])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"experiment_name": "exp"', result.output)

    def test_analyze_accepts_csv_without_mapping(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            input_path.write_text(
                "variant_name,sessions,orders\ncontrol,1000,40\nv1,1000,45\n",
                encoding="utf-8",
            )
            result = runner.invoke(cli, ["analyze", "--input", str(input_path)])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["control_variant"], "control")
        self.assertIsNone(payload["recommendation"])
        self.assertEqual(
            payload["analysis_settings"]["input_interpretation"]["source_type"], "aggregated"
        )

    def test_output_includes_schema_version(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            input_path.write_text("v,n,c\ncontrol,1000,40\nv1,1000,45\n", encoding="utf-8")
            result = runner.invoke(cli, ["analyze", "--input", str(input_path)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["schema_version"], "1")

    def test_invalid_json_gives_clean_error(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.json"
            input_path.write_text('{"method": "bayesian"}', encoding="utf-8")
            result = runner.invoke(cli, ["analyze", "--input", str(input_path)])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("variants", result.output)
        self.assertNotIn("Traceback", result.output)

    def test_bundled_examples_analyze(self):
        root = Path(__file__).resolve().parents[1] / "examples"
        runner = CliRunner()
        cases = [
            ("conversion_multivariant.csv", "conversion_rate", []),
            ("arpu_bayesian.csv", "arpu", []),
            ("arpu_bayesian.xlsx", "arpu", ["--sheet", "Sheet1"]),
            ("arpu_sequential.csv", "arpu", ["--method", "sequential"]),
        ]
        for name, metric, extra in cases:
            result = runner.invoke(
                cli, ["analyze", "--input", str(root / name), "--samples", "2000", *extra]
            )
            self.assertEqual(result.exit_code, 0, f"{name}: {result.output}")
            payload = json.loads(result.output)
            self.assertEqual(payload["analysis_settings"]["primary_metric"], metric, name)


class DurationCliTests(unittest.TestCase):
    def test_missing_assumptions_error_instead_of_prompt(self):
        result = CliRunner().invoke(cli, ["duration", "--baseline-rate", "0.04"], input="")
        self.assertEqual(result.exit_code, 2)
        self.assertIn("--relative-mde", result.output)
        self.assertIn("--daily-traffic", result.output)

    def test_sequential_is_default_and_longer_than_fixed(self):
        args = ["--baseline-rate", "0.04", "--relative-mde", "0.05", "--daily-traffic", "50000"]
        seq = CliRunner().invoke(cli, ["duration", *args])
        fixed = CliRunner().invoke(cli, ["duration", "--method", "fixed", *args])
        self.assertEqual(seq.exit_code, 0, seq.output)
        self.assertEqual(fixed.exit_code, 0, fixed.output)
        seq_plan, fixed_plan = json.loads(seq.output), json.loads(fixed.output)
        self.assertEqual(seq_plan["method"], "sequential")
        self.assertGreater(seq_plan["n_per_variant"], fixed_plan["n_per_variant"])

    def test_bundled_duration_examples(self):
        root = Path(__file__).resolve().parents[1] / "examples"
        for args in (
            ["--input", str(root / "duration_inputs.csv")],
            ["--input", str(root / "duration_inputs.xlsx"), "--sheet", "Sheet1"],
        ):
            result = CliRunner().invoke(cli, ["duration", *args])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIsNotNone(json.loads(result.output)["n_per_variant"])


class AnalyzeConfigTests(unittest.TestCase):
    def test_analyze_uses_explicit_config_for_recommendation_policy(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            pyproject_path = Path(tmp) / "pyproject.toml"
            input_path.write_text(
                "variant_name,sessions,orders\ncontrol,1000,40\nv1,1000,45\n",
                encoding="utf-8",
            )
            pyproject_path.write_text(
                "[tool.ablift]\n"
                'method = "bayesian"\n'
                "samples = 10000\n"
                "\n"
                "[tool.ablift.decision_policy]\n"
                "enabled = true\n"
                "bayes_prob_beats_control = 0.8\n"
                "max_expected_loss = 0.01\n",
                encoding="utf-8",
            )
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(cli, ["analyze", "--input", "input.csv"])
            finally:
                os.chdir(old_cwd)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"samples": 10000', result.output)
        self.assertNotIn('"recommendation": null', result.output)
        self.assertIn('"decision_policy"', result.output)

    def test_analyze_help_includes_examples(self):
        runner = CliRunner()

        result = runner.invoke(cli, ["analyze", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Examples:", result.output)
        self.assertIn("ablift analyze --input experiment.csv", result.output)
        self.assertIn("--enable-recommendation --prob-threshold 0.9", result.output)

    def test_analyze_surfaces_actionable_column_error(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.csv"
            input_path.write_text(
                "group_name,sessions\ncontrol,1000\nv1,1000\n",
                encoding="utf-8",
            )
            result = runner.invoke(cli, ["analyze", "--input", str(input_path)])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("at least 3 columns", result.output)


if __name__ == "__main__":
    unittest.main()
