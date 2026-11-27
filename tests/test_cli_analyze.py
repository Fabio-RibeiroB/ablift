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
            ["--input", str(root / "duration_arpu.csv")],
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


class AnalyzeExpectedSplitCliTests(unittest.TestCase):
    def test_csv_with_positional_expected_split_passes_and_normalizes(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rollout.csv"
            input_path.write_text(
                "variant,visitors,conversions\ncontrol,90000,3600\nv1,10000,500\n",
                encoding="utf-8",
            )
            for split_arg in ("0.9,0.1", "90,10", " 90 , 10 "):
                with self.subTest(split_arg=split_arg):
                    result = runner.invoke(
                        cli,
                        [
                            "analyze",
                            "--input",
                            str(input_path),
                            "--method",
                            "sequential",
                            "--expected-split",
                            split_arg,
                        ],
                    )
                    self.assertEqual(result.exit_code, 0, result.output)
                    data = json.loads(result.output)
                    self.assertTrue(data["srm"]["passed"])
                    self.assertEqual(data["srm"]["expected_proportions"], [0.9, 0.1])
                    self.assertEqual(data["srm"]["expected"], [90000.0, 10000.0])
                    self.assertEqual(data["recommendation"]["action"], "ship_v1")

    def test_csv_with_named_expected_split_pairs(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rollout.csv"
            input_path.write_text(
                "variant,visitors,conversions\ncontrol,80000,3200\nv1,20000,900\n",
                encoding="utf-8",
            )
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "--input",
                    str(input_path),
                    "--method",
                    "sequential",
                    "--expected-split",
                    "v1=0.2,control=0.8",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertTrue(data["srm"]["passed"])
        self.assertEqual(data["srm"]["expected_proportions"], [0.8, 0.2])
        self.assertEqual(data["srm"]["expected"], [80000.0, 20000.0])

    def test_row_level_csv_with_expected_split(self):
        runner = CliRunner()
        rows = ["variant,converted"] + ["control,0"] * 75 + ["control,1"] * 5
        rows += ["v1,0"] * 18 + ["v1,1"] * 2
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rows.csv"
            input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "--input",
                    str(input_path),
                    "--samples",
                    "1000",
                    "--expected-split",
                    "control=80,v1=20",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertTrue(data["srm"]["passed"])
        self.assertEqual(data["srm"]["expected_proportions"], [0.8, 0.2])
        self.assertEqual(data["srm"]["observed"], [80, 20])

    def test_json_expected_share_and_expected_split_list_and_dict(self):
        runner = CliRunner()
        payloads = [
            {
                "experiment_name": "share_exp",
                "method": "sequential",
                "variants": [
                    {
                        "name": "control",
                        "visitors": 90000,
                        "conversions": 3600,
                        "is_control": True,
                        "expected_share": 0.9,
                    },
                    {
                        "name": "v1",
                        "visitors": 10000,
                        "conversions": 500,
                        "is_control": False,
                        "expected_share": 0.1,
                    },
                ],
            },
            {
                "experiment_name": "list_exp",
                "method": "sequential",
                "expected_split": [90, 10],
                "variants": [
                    {"name": "control", "visitors": 90000, "conversions": 3600, "is_control": True},
                    {"name": "v1", "visitors": 10000, "conversions": 500, "is_control": False},
                ],
            },
            {
                "experiment_name": "dict_exp",
                "method": "sequential",
                "expected_split": {"v1": 1, "control": 9},
                "variants": [
                    {"name": "control", "visitors": 90000, "conversions": 3600, "is_control": True},
                    {"name": "v1", "visitors": 10000, "conversions": 500, "is_control": False},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for idx, payload in enumerate(payloads):
                with self.subTest(name=payload["experiment_name"]):
                    path = Path(tmp) / f"in_{idx}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    result = runner.invoke(cli, ["analyze", "--input", str(path)])
                    self.assertEqual(result.exit_code, 0, result.output)
                    data = json.loads(result.output)
                    self.assertTrue(data["srm"]["passed"])
                    self.assertEqual(data["srm"]["expected_proportions"], [0.9, 0.1])
                    self.assertEqual(data["recommendation"]["action"], "ship_v1")

    def test_cli_expected_split_overrides_json_split_share_and_pyproject(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            pyproject_path = Path(tmp) / "pyproject.toml"
            pyproject_path.write_text(
                "[tool.ablift]\n"
                'method = "sequential"\n'
                "[tool.ablift.expected_split]\n"
                "control = 0.5\n"
                "v1 = 0.5\n",
                encoding="utf-8",
            )
            # JSON with per-variant expected_share=0.5/0.5 (would fail SRM for 90k/10k)
            json_share = Path(tmp) / "share.json"
            json_share.write_text(
                json.dumps(
                    {
                        "experiment_name": "override_share",
                        "variants": [
                            {
                                "name": "control",
                                "visitors": 90000,
                                "conversions": 3600,
                                "is_control": True,
                                "expected_share": 0.5,
                            },
                            {
                                "name": "v1",
                                "visitors": 10000,
                                "conversions": 500,
                                "is_control": False,
                                "expected_share": 0.5,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            # JSON with dict expected_split (would fail SRM for 90k/10k)
            json_dict = Path(tmp) / "dict.json"
            json_dict.write_text(
                json.dumps(
                    {
                        "experiment_name": "override_dict",
                        "expected_split": {"control": 0.5, "v1": 0.5},
                        "variants": [
                            {
                                "name": "control",
                                "visitors": 90000,
                                "conversions": 3600,
                                "is_control": True,
                            },
                            {
                                "name": "v1",
                                "visitors": 10000,
                                "conversions": 500,
                                "is_control": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            # JSON with per-variant expected_share=90/10 overrides pyproject expected_split=50/50
            json_share_beats_pyproject = Path(tmp) / "share_beats_pyproject.json"
            json_share_beats_pyproject.write_text(
                json.dumps(
                    {
                        "experiment_name": "share_beats_pyproject",
                        "variants": [
                            {
                                "name": "control",
                                "visitors": 90000,
                                "conversions": 3600,
                                "is_control": True,
                                "expected_share": 0.9,
                            },
                            {
                                "name": "v1",
                                "visitors": 10000,
                                "conversions": 500,
                                "is_control": False,
                                "expected_share": 0.1,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                r1 = runner.invoke(
                    cli,
                    ["analyze", "--input", "share.json", "--expected-split", "0.9,0.1"],
                )
                r2 = runner.invoke(
                    cli,
                    [
                        "analyze",
                        "--input",
                        "dict.json",
                        "--expected-split",
                        "control=0.9,v1=0.1",
                    ],
                )
                r3 = runner.invoke(cli, ["analyze", "--input", "share_beats_pyproject.json"])
            finally:
                os.chdir(old_cwd)

        for res in (r1, r2, r3):
            self.assertEqual(res.exit_code, 0, res.output)
            data = json.loads(res.output)
            self.assertTrue(data["srm"]["passed"])
            self.assertEqual(data["srm"]["expected_proportions"], [0.9, 0.1])

    def test_cli_rejects_malformed_and_invalid_expected_split_cleanly(self):
        runner = CliRunner()
        bad_flags = [
            "",
            "   ",
            "0.9,",
            ",0.1",
            "0.9,,0.1",
            "control=0.8,0.2",
            "=0.5,v1=0.5",
            "control=,v1=0.5",
            "control=0.5=0.1,v1=0.5",
            "control=0.5,control=0.5",
            "0.9,abc",
            "control=abc,v1=0.2",
            "0,1",
            "-0.5,1.5",
            "nan,0.5",
            "inf,0.5",
            "-inf,0.5",
            "0.5",
            "0.5,0.25,0.25",
            "control=0.8,unknown=0.2",
            "control=1.0",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "in.csv"
            input_path.write_text(
                "variant,visitors,conversions\ncontrol,1000,40\nv1,1000,45\n",
                encoding="utf-8",
            )
            for flag in bad_flags:
                with self.subTest(flag=flag):
                    result = runner.invoke(
                        cli,
                        [
                            "analyze",
                            "--input",
                            str(input_path),
                            "--expected-split",
                            flag,
                        ],
                    )
                    self.assertEqual(result.exit_code, 1, f"{flag!r}: {result.output}")
                    self.assertIn("Error:", result.output)
                    self.assertNotIn("Traceback", result.output)

    def test_cli_rejects_json_with_both_expected_split_and_expected_share(self):
        runner = CliRunner()
        payload = {
            "experiment_name": "both",
            "expected_split": [0.9, 0.1],
            "variants": [
                {
                    "name": "control",
                    "visitors": 9000,
                    "conversions": 360,
                    "is_control": True,
                    "expected_share": 0.9,
                },
                {
                    "name": "v1",
                    "visitors": 1000,
                    "conversions": 50,
                    "is_control": False,
                    "expected_share": 0.1,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "both.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = runner.invoke(cli, ["analyze", "--input", str(path)])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("Cannot specify both", result.output)
        self.assertNotIn("Traceback", result.output)


if __name__ == "__main__":
    unittest.main()
