import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from click.testing import CliRunner
from openpyxl import Workbook

from ablift.cli import cli

CSV_BASIC = "variant,visitors,conversions\ncontrol,1000,40\nv1,1000,52\n"

BASE_PAYLOAD = {
    "experiment_name": "from_json",
    "variants": [
        {"name": "control", "visitors": 1000, "conversions": 40, "is_control": True},
        {"name": "v1", "visitors": 1000, "conversions": 52, "is_control": False},
    ],
}

DURATION_ARGS = ["--baseline-rate", "0.04", "--relative-mde", "0.05", "--daily-traffic", "50000"]


@contextmanager
def chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, name, text):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_json(self, name, payload):
        return self.write(name, json.dumps(payload))

    def run_cli(self, args, **kwargs):
        return CliRunner().invoke(cli, [str(a) for a in args], **kwargs)

    def assert_clean_error(self, result, code=1):
        self.assertEqual(result.exit_code, code, result.output)
        self.assertIn("Error", result.output)
        self.assertNotIn("Traceback", result.output)
        self.assertIsInstance(result.exception, SystemExit, repr(result.exception))


class OutputFileTests(TmpDirCase):
    def test_output_writes_json_file_and_silences_stdout(self):
        src = self.write("in.csv", CSV_BASIC)
        out = self.dir / "a" / "b" / "result.json"
        result = self.run_cli(["analyze", "--input", src, "--output", out, "--samples", 2000])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), "")
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "1")
        self.assertEqual(data["experiment_name"], "in")

    def test_report_writes_markdown_and_keeps_json_on_stdout(self):
        src = self.write("in.csv", CSV_BASIC)
        report = self.dir / "deep" / "er" / "report.md"
        result = self.run_cli(["analyze", "--input", src, "--report", report, "--samples", 2000])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["experiment_name"], "in")
        text = report.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# A/B Decision Report: in"))
        self.assertIn("## Variant Comparisons", text)

    def test_output_and_report_together(self):
        src = self.write("in.csv", CSV_BASIC)
        out, report = self.dir / "o" / "r.json", self.dir / "p" / "r.md"
        result = self.run_cli(
            ["analyze", "--input", src, "--output", out, "--report", report, "--samples", 2000]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), "")
        self.assertTrue(out.exists())
        self.assertTrue(report.read_text(encoding="utf-8").startswith("# A/B Decision Report"))

    def test_duration_output_creates_nested_dirs(self):
        out = self.dir / "x" / "y" / "plan.json"
        result = self.run_cli(["duration", *DURATION_ARGS, "--output", out])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), "")
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["schema_version"], "1")


class PrecedenceTests(TmpDirCase):
    def run_in_dir(self, args):
        with chdir(self.dir):
            return self.run_cli(args)

    def pyproject(self, body):
        self.write("pyproject.toml", body)

    def test_flag_beats_input_file_beats_pyproject_beats_default(self):
        payload = dict(BASE_PAYLOAD, samples=3000, random_seed=11, method="bayesian")
        src = self.write_json("in.json", payload)
        self.pyproject("[tool.ablift]\nsamples = 5000\nrandom_seed = 5\nalpha = 0.1\n")

        settings = json.loads(self.run_in_dir(["analyze", "--input", src.name]).output)[
            "analysis_settings"
        ]
        self.assertEqual(settings["samples"], 3000)  # input file > pyproject
        self.assertEqual(settings["random_seed"], 11)

        out = self.run_in_dir(
            ["analyze", "--input", src.name, "--samples", 2000, "--random-seed", 99]
        )
        settings = json.loads(out.output)["analysis_settings"]
        self.assertEqual(settings["samples"], 2000)  # flag > input file
        self.assertEqual(settings["random_seed"], 99)
        self.assertEqual(settings["alpha"], 0.1)  # pyproject > default

    def test_pyproject_beats_default_for_csv(self):
        src = self.write("in.csv", CSV_BASIC)
        self.pyproject("[tool.ablift]\nsamples = 2500\nrandom_seed = 5\n")
        settings = json.loads(self.run_in_dir(["analyze", "--input", src.name]).output)[
            "analysis_settings"
        ]
        self.assertEqual((settings["samples"], settings["random_seed"]), (2500, 5))

        settings = json.loads(
            self.run_in_dir(["analyze", "--input", src.name, "--samples", 2000]).output
        )["analysis_settings"]
        self.assertEqual(settings["samples"], 2000)
        self.assertEqual(settings["random_seed"], 5)

    def test_defaults_when_nothing_configured(self):
        src = self.write("in.csv", CSV_BASIC)
        data = json.loads(
            self.run_in_dir(["analyze", "--input", src.name, "--samples", 2000]).output
        )
        self.assertEqual(data["method"], "bayesian")
        self.assertEqual(data["analysis_settings"]["random_seed"], 7)

    def test_method_across_layers(self):
        src = self.write_json("in.json", dict(BASE_PAYLOAD, method="bayesian"))
        self.pyproject('[tool.ablift]\nmethod = "sequential"\n')
        self.assertEqual(
            json.loads(self.run_in_dir(["analyze", "--input", src.name, "--samples", 2000]).output)[
                "method"
            ],
            "bayesian",
        )
        self.assertEqual(
            json.loads(
                self.run_in_dir(["analyze", "--input", src.name, "--method", "sequential"]).output
            )["method"],
            "sequential",
        )
        # pyproject alone beats the built-in default for a CSV
        csv = self.write("in.csv", CSV_BASIC)
        self.assertEqual(
            json.loads(self.run_in_dir(["analyze", "--input", csv.name]).output)["method"],
            "sequential",
        )

    def test_decision_policy_layers(self):
        src = self.write("in.csv", CSV_BASIC)
        self.pyproject(
            "[tool.ablift]\nsamples = 2000\n\n[tool.ablift.decision_policy]\n"
            "enabled = true\nbayes_prob_beats_control = 0.8\nmax_expected_loss = 0.01\n"
        )
        data = json.loads(self.run_in_dir(["analyze", "--input", src.name]).output)
        self.assertIsNotNone(data["recommendation"])
        policy = data["analysis_settings"]["decision_policy"]
        self.assertEqual(policy["thresholds"]["bayes_prob_beats_control"], 0.8)

        # flag overrides a single threshold, the rest comes from pyproject
        data = json.loads(
            self.run_in_dir(["analyze", "--input", src.name, "--prob-threshold", 0.7]).output
        )
        policy = data["analysis_settings"]["decision_policy"]
        self.assertEqual(policy["thresholds"]["bayes_prob_beats_control"], 0.7)
        self.assertEqual(policy["thresholds"]["max_expected_loss"], 0.01)

    def test_no_recommendation_disables_pyproject_policy(self):
        src = self.write("in.csv", CSV_BASIC)
        self.pyproject(
            "[tool.ablift]\nsamples = 2000\n\n[tool.ablift.decision_policy]\n"
            "enabled = true\nbayes_prob_beats_control = 0.8\nmax_expected_loss = 0.01\n"
        )
        result = self.run_in_dir(["analyze", "--input", src.name, "--no-recommendation"])
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertIsNone(data["recommendation"])
        self.assertFalse(data["analysis_settings"]["decision_policy"]["enabled"])

    def test_prob_threshold_alone_enables_policy(self):
        src = self.write("in.csv", CSV_BASIC)
        result = self.run_cli(
            ["analyze", "--input", src, "--prob-threshold", 0.9, "--samples", 2000]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertIsNotNone(data["recommendation"])
        self.assertTrue(data["analysis_settings"]["decision_policy"]["enabled"])

    def test_max_expected_loss_alone_enables_policy(self):
        src = self.write("in.csv", CSV_BASIC)
        result = self.run_cli(
            ["analyze", "--input", src, "--max-expected-loss", 0.01, "--samples", 2000]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNotNone(json.loads(result.output)["recommendation"])

    def test_no_recommendation_on_json_with_policy(self):
        payload = dict(
            BASE_PAYLOAD,
            samples=2000,
            decision_policy={"enabled": True, "bayes_prob_beats_control": 0.9},
        )
        src = self.write_json("in.json", payload)
        self.assertIsNotNone(
            json.loads(self.run_cli(["analyze", "--input", src]).output)["recommendation"]
        )
        off = self.run_cli(["analyze", "--input", src, "--no-recommendation"])
        self.assertIsNone(json.loads(off.output)["recommendation"])

    def test_malformed_pyproject_is_clean_error(self):
        src = self.write("in.csv", CSV_BASIC)
        self.pyproject("[tool.ablift\n")
        result = self.run_in_dir(["analyze", "--input", src.name])
        self.assert_clean_error(result)


class AnalyzeOptionTests(TmpDirCase):
    def test_sequential_on_csv_with_tau_and_alpha(self):
        src = self.write("in.csv", CSV_BASIC)
        result = self.run_cli(
            [
                "analyze",
                "--input",
                src,
                "--method",
                "sequential",
                "--sequential-tau",
                0.02,
                "--alpha",
                0.1,
            ]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["method"], "sequential")
        self.assertEqual(data["analysis_settings"]["sequential_tau"], 0.02)
        self.assertEqual(data["analysis_settings"]["alpha"], 0.1)
        self.assertIsNotNone(data["comparisons"][0]["e_value"])

    def test_primary_metric_override_on_five_column_csv(self):
        src = self.write(
            "five.csv",
            "variant,visitors,conversions,revenue,revenue_sq\n"
            "control,1000,100,5000,400000\nv1,1000,110,5600,450000\n",
        )
        default = json.loads(self.run_cli(["analyze", "--input", src, "--samples", 2000]).output)
        self.assertEqual(default["analysis_settings"]["primary_metric"], "arpu")
        override = json.loads(
            self.run_cli(
                [
                    "analyze",
                    "--input",
                    src,
                    "--primary-metric",
                    "conversion_rate",
                    "--samples",
                    2000,
                ]
            ).output
        )
        self.assertEqual(override["analysis_settings"]["primary_metric"], "conversion_rate")
        self.assertEqual(override["comparisons"][0]["metric"], "conversion_rate")

    def test_experiment_name_override_vs_file_stem(self):
        src = self.write("my_experiment.csv", CSV_BASIC)
        stem = json.loads(self.run_cli(["analyze", "--input", src, "--samples", 2000]).output)
        self.assertEqual(stem["experiment_name"], "my_experiment")
        named = json.loads(
            self.run_cli(
                ["analyze", "--input", src, "--experiment-name", "custom", "--samples", 2000]
            ).output
        )
        self.assertEqual(named["experiment_name"], "custom")

    def test_xlsx_sheet_selection(self):
        wb = Workbook()
        wb.active.title = "Other"
        wb.active.append(["a", "b"])
        ws = wb.create_sheet("Results")
        for row in (["v", "n", "c"], ["control", 1000, 40], ["v1", 1000, 50]):
            ws.append(row)
        path = self.dir / "book.xlsx"
        wb.save(path)
        result = self.run_cli(["analyze", "--input", path, "--sheet", "Results", "--samples", 2000])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["experiment_name"], "book")


class AnalyzeErrorTests(TmpDirCase):
    def analyze_error(self, name, text, code=1, extra=()):
        src = self.write(name, text)
        result = self.run_cli(["analyze", "--input", src, *extra])
        self.assert_clean_error(result, code)
        return result

    def test_malformed_json(self):
        self.analyze_error("bad.json", "{not json")

    def test_json_missing_variants(self):
        result = self.analyze_error("nov.json", '{"method": "bayesian"}')
        self.assertIn("variants", result.output)

    def test_unsupported_extension(self):
        result = self.analyze_error("data.txt", "a,b,c")
        self.assertIn("Unsupported", result.output)

    def test_csv_too_few_columns(self):
        result = self.analyze_error("few.csv", "name,visitors\ncontrol,10\nv1,10\n")
        self.assertIn("at least 3 columns", result.output)

    def test_csv_empty(self):
        self.analyze_error("empty.csv", "")

    def test_csv_header_only(self):
        self.analyze_error("hdr.csv", "a,b,c\n")

    def test_csv_conversions_exceed_visitors(self):
        self.analyze_error("over.csv", "v,n,c\ncontrol,100,200\nv1,100,5\n")

    def test_json_conversions_exceed_visitors(self):
        payload = json.loads(json.dumps(BASE_PAYLOAD))
        payload["variants"][1]["conversions"] = 5000
        result = self.analyze_error("over.json", json.dumps(payload))
        self.assertIn("cannot exceed visitors", result.output)

    def test_json_single_variant(self):
        payload = {"variants": BASE_PAYLOAD["variants"][:1]}
        result = self.analyze_error("one.json", json.dumps(payload))
        self.assertIn("At least 2 variants", result.output)

    def test_json_bad_variant_fields(self):
        payload = {"variants": [{"name": "a", "bogus": 1}, {"name": "b", "bogus": 2}]}
        self.analyze_error("fields.json", json.dumps(payload))

    def test_invalid_alpha_flag(self):
        self.analyze_error("in.csv", CSV_BASIC, extra=["--alpha", 2])

    def test_unknown_sheet_name(self):
        wb = Workbook()
        wb.active.title = "Real"
        for row in (["v", "n", "c"], ["control", 100, 5], ["v1", 100, 6]):
            wb.active.append(row)
        path = self.dir / "book.xlsx"
        wb.save(path)
        result = self.run_cli(["analyze", "--input", path, "--sheet", "Nope"])
        self.assert_clean_error(result)
        self.assertIn("Nope", result.output)

    def test_json_array_instead_of_object(self):
        # Regression (was a bug): ablift/cli.py _deep_merge(input_payload, defaults) raises an uncaught
        # TypeError for a top-level JSON array; expected a clean "Input must be a JSON
        # object." error with exit code 1 and no traceback.
        src = self.write("arr.json", "[1, 2, 3]")
        result = self.run_cli(["analyze", "--input", src])
        self.assert_clean_error(result)

    def test_nonexistent_input_path_exits_2(self):
        result = self.run_cli(["analyze", "--input", self.dir / "missing.csv"])
        self.assertEqual(result.exit_code, 2)
        self.assertNotIn("Traceback", result.output)

    def test_analyze_requires_input(self):
        result = self.run_cli(["analyze"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("--input", result.output)


class DurationCliTests(TmpDirCase):
    def test_each_method_returns_contract(self):
        for method in ("sequential", "fixed", "bayesian"):
            extra = ["--max-days", 5] if method == "bayesian" else []
            result = self.run_cli(["duration", "--method", method, *DURATION_ARGS, *extra])
            self.assertEqual(result.exit_code, 0, f"{method}: {result.output}")
            data = json.loads(result.output)
            self.assertEqual(data["schema_version"], "1")
            self.assertIn("method", data)
            self.assertIn("n_per_variant", data)
            self.assertIn("estimated_days", data)
            self.assertIn("assumptions", data)

    def test_more_variants_take_longer(self):
        two = json.loads(self.run_cli(["duration", *DURATION_ARGS]).output)
        four = json.loads(self.run_cli(["duration", *DURATION_ARGS, "--n-variants", 4]).output)
        self.assertGreater(four["estimated_days"], two["estimated_days"])

    def test_csv_input_with_flag_override(self):
        src = self.write(
            "d.csv",
            "baseline_rate,relative_mde,daily_traffic,method\n0.04,0.05,50000,fixed\n",
        )
        from_file = json.loads(self.run_cli(["duration", "--input", src]).output)
        self.assertEqual(from_file["method"], "fixed_horizon")
        flagged = json.loads(
            self.run_cli(["duration", "--input", src, "--method", "sequential"]).output
        )
        self.assertEqual(flagged["method"], "sequential")
        rate = json.loads(
            self.run_cli(["duration", "--input", src, "--baseline-rate", 0.1]).output
        )["assumptions"]["baseline_rate"]
        self.assertEqual(rate, 0.1)

    def test_unknown_column(self):
        src = self.write(
            "d.csv", "baseline_rate,relative_mde,daily_traffic,bogus\n0.04,0.05,1000,1\n"
        )
        result = self.run_cli(["duration", "--input", src])
        self.assert_clean_error(result)
        self.assertIn("bogus", result.output)

    def test_invalid_values(self):
        cases = [
            {"--baseline-rate": "1.5"},
            {"--n-variants": "1"},
            {"--relative-mde": "0"},
            {"--power": "2"},
            {"--alpha": "0"},
            {"--daily-traffic": "0"},
        ]
        for override in cases:
            args = {"--baseline-rate": "0.04", "--relative-mde": "0.05", "--daily-traffic": "50000"}
            args.update(override)
            argv = ["duration"]
            for key, value in args.items():
                argv += [key, value]
            with self.subTest(override=override):
                self.assert_clean_error(self.run_cli(argv))

    def test_missing_args_exit_2_and_name_flags(self):
        result = self.run_cli(["duration"])
        self.assertEqual(result.exit_code, 2)
        for flag in ("--baseline-rate", "--relative-mde", "--daily-traffic"):
            self.assertIn(flag, result.output)

    def test_non_numeric_flag(self):
        result = self.run_cli(["duration", "--baseline-rate", "abc"])
        self.assertEqual(result.exit_code, 2)

    def test_bayesian_tiny_traffic_gives_null_days(self):
        result = self.run_cli(
            [
                "duration",
                "--method",
                "bayesian",
                "--baseline-rate",
                0.04,
                "--relative-mde",
                0.05,
                "--daily-traffic",
                100,
                "--max-days",
                1,
            ]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(json.loads(result.output)["estimated_days"])


class MetaCommandTests(TmpDirCase):
    def test_version(self):
        result = self.run_cli(["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ablift", result.output)
        self.assertRegex(result.output, r"\d+\.\d+")

    def test_help_lists_exact_commands(self):
        result = self.run_cli(["--help"])
        self.assertEqual(result.exit_code, 0)
        section = result.output.split("Commands:")[1]
        names = [line.split()[0] for line in section.strip().splitlines() if line.strip()]
        self.assertEqual(
            sorted(names), ["analyze", "doctor", "duration", "example-input", "schema"]
        )
        self.assertNotIn("analyze-file", result.output)
        self.assertNotIn("example-duration-prompt", result.output)

    def test_doctor_json_output(self):
        result = self.run_cli(["doctor", "--json-output", "--strict"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(json.loads(result.output)["ok"])

    def test_doctor_text_output(self):
        result = self.run_cli(["doctor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Doctor status: OK", result.output)
        self.assertIn("PASS python_version", result.output)

    def test_example_input_round_trips_through_analyze(self):
        example = self.run_cli(["example-input"])
        self.assertEqual(example.exit_code, 0, example.output)
        payload = json.loads(example.output)
        self.assertIn("decision_policy", payload)
        src = self.write("example.json", example.output)
        result = self.run_cli(["analyze", "--input", src, "--samples", 2000])
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["experiment_name"], payload["experiment_name"])
        self.assertIsNotNone(data["recommendation"])

    def test_example_input_with_report(self):
        example = self.run_cli(["example-input"])
        src = self.write("example.json", example.output)
        report = self.dir / "r.md"
        result = self.run_cli(
            [
                "analyze",
                "--input",
                src,
                "--samples",
                2000,
                "--report",
                report,
                "--output",
                self.dir / "o.json",
            ]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        text = report.read_text(encoding="utf-8")
        self.assertIn("checkout_button_color", text)
        self.assertIn("bounce_rate", text)


class AgentWorkflowTests(TmpDirCase):
    ACTIONS = {
        "continue_collecting_data",
        "do_not_ship",
        "investigate_data_quality",
        "stop_and_rollback",
    }

    def test_successive_cumulative_snapshots(self):
        # (day, control_visitors, control_conversions, treatment_visitors, treatment_conversions)
        days = []
        cv = cc = tv = tc = 0
        for day in range(1, 8):
            cv += 2000
            tv += 2000
            cc += 80 + day
            tc += 100 + 2 * day
            days.append((day, cv, cc, tv, tc))

        seen = []
        for day, cv, cc, tv, tc in days:
            src = self.write(
                f"snap_day{day}.csv",
                f"variant,visitors,conversions\ncontrol,{cv},{cc}\ntreatment,{tv},{tc}\n",
            )
            result = self.run_cli(["analyze", "--input", src, "--method", "sequential"])
            self.assertEqual(result.exit_code, 0, f"day {day}: {result.output}")
            data = json.loads(result.output)
            self.assertEqual(data["schema_version"], "1")
            rec = data["recommendation"]
            self.assertIsNotNone(rec)
            self.assertTrue(
                rec["action"] in self.ACTIONS or rec["action"].startswith("ship_"),
                rec["action"],
            )
            self.assertGreaterEqual(rec["decision_confidence"], 0.0)
            self.assertLessEqual(rec["decision_confidence"], 1.0)
            self.assertIsInstance(rec["risk_flags"], list)
            self.assertIsInstance(rec["rationale"], str)
            self.assertIsInstance(rec["next_best_action"], str)
            for comp in data["comparisons"]:
                self.assertIn("p_value", comp)
                self.assertGreaterEqual(comp["p_value"], 0.0)
                self.assertLessEqual(comp["p_value"], 1.0)
            seen.append(data["comparisons"][0]["p_value"])

        # More cumulative evidence with a consistent lift should not make p-values grow
        # unboundedly: the final snapshot is at least as significant as the first.
        self.assertLessEqual(seen[-1], seen[0])


if __name__ == "__main__":
    unittest.main()
