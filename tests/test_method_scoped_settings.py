"""Method-scoped settings: samples, random_seed and sequential_tau only where they apply.

The CLI rejects --samples/--random-seed/--sequential-tau for methods that ignore them;
JSON input must be held to the same rule, output must not report unused values, and
[tool.ablift] project defaults must never trip the rejection.
"""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from click.testing import CliRunner

from ablift.cli import cli
from ablift.engine import analyze, parse_payload
from ablift.schemas import get_schema

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    {"name": "control", "visitors": 1000, "conversions": 40, "is_control": True},
    {"name": "v1", "visitors": 1000, "conversions": 52, "is_control": False},
]
CSV_BASIC = "variant,visitors,conversions\ncontrol,1000,40\nv1,1000,52\n"
ROW_ARPU_CSV = "variant,revenue\n" + "".join(
    f"{arm},{value}\n" for arm in ("control", "v1") for value in (0, 0, 5, 10, 12, 3, 0, 8) * 15
)


@contextmanager
def chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_payload(**fields):
    return analyze(parse_payload({"experiment_name": "t", "variants": VARIANTS, **fields}))


class EngineRejectsUnusedSettingsTests(unittest.TestCase):
    def test_samples_rejected_for_non_bayesian(self):
        for method in ("sequential", "fixed"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "samples only applies.*bayesian"):
                    run_payload(method=method, samples=1000)

    def test_random_seed_rejected_for_non_bayesian(self):
        for method in ("sequential", "fixed"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "random_seed only applies.*bayesian"):
                    run_payload(method=method, random_seed=3)

    def test_sequential_tau_rejected_for_bayesian(self):
        with self.assertRaisesRegex(ValueError, "sequential_tau only applies.*sequential"):
            run_payload(method="bayesian", sequential_tau=0.01)
        with self.assertRaisesRegex(ValueError, "sequential_tau only applies.*sequential"):
            run_payload(sequential_tau=0.01)  # method defaults to bayesian

    def test_null_values_are_not_explicit(self):
        result = run_payload(method="sequential", samples=None, random_seed=None)
        self.assertIsNone(result.analysis_settings.samples)

    def test_settings_report_null_when_unused(self):
        for method in ("sequential", "fixed"):
            with self.subTest(method=method):
                settings = run_payload(method=method).analysis_settings
                self.assertIsNone(settings.samples)
                self.assertIsNone(settings.random_seed)
        settings = run_payload(method="bayesian", samples=2000).analysis_settings
        self.assertEqual((settings.samples, settings.random_seed), (2000, 7))

    def test_output_schema_allows_null_samples_and_seed(self):
        props = get_schema("analyze-output")["$defs"]["AnalysisSettings"]["properties"]
        self.assertEqual(props["samples"]["type"], ["integer", "null"])
        self.assertEqual(props["random_seed"]["type"], ["integer", "null"])


class CliMethodScopedSettingsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def invoke(self, args, cwd):
        with chdir(cwd):
            return CliRunner().invoke(cli, ["analyze", *map(str, args)])

    def test_repo_pyproject_defaults_do_not_block_sequential(self):
        # This repo's own pyproject sets samples and random_seed under [tool.ablift].
        res = self.invoke(["--input", "examples/checkout_test.csv", "--method", "sequential"], ROOT)
        self.assertEqual(res.exit_code, 0, res.output)
        settings = json.loads(res.output)["analysis_settings"]
        self.assertIsNone(settings["samples"])
        self.assertIsNone(settings["random_seed"])
        self.assertEqual(settings["config_source"], str(ROOT / "pyproject.toml"))

    def test_pyproject_method_scoped_defaults_never_trigger_rejection(self):
        (self.dir / "pyproject.toml").write_text(
            '[tool.ablift]\nmethod = "sequential"\nsamples = 2000\nrandom_seed = 9\n'
            "sequential_tau = 0.01\n",
            encoding="utf-8",
        )
        (self.dir / "in.csv").write_text(CSV_BASIC, encoding="utf-8")
        (self.dir / "in.json").write_text(json.dumps({"variants": VARIANTS}), encoding="utf-8")
        cases = {
            "sequential": (None, None, 0.01),
            "fixed": (None, None, None),
            "bayesian": (2000, 9, None),
        }
        for src in ("in.csv", "in.json"):
            for method, (samples, seed, tau) in cases.items():
                with self.subTest(src=src, method=method):
                    res = self.invoke(["--input", src, "--method", method], self.dir)
                    self.assertEqual(res.exit_code, 0, res.output)
                    settings = json.loads(res.output)["analysis_settings"]
                    self.assertEqual(
                        (settings["samples"], settings["random_seed"], settings["sequential_tau"]),
                        (samples, seed, tau),
                    )

    def test_json_samples_with_sequential_is_a_clean_error(self):
        src = self.dir / "seq.json"
        src.write_text(
            json.dumps({"method": "sequential", "samples": 1000, "variants": VARIANTS}),
            encoding="utf-8",
        )
        res = self.invoke(["--input", src, "--no-config"], self.dir)
        self.assertEqual(res.exit_code, 1, res.output)
        self.assertIn("samples only applies", res.output)
        self.assertNotIn("Traceback", res.output)
        self.assertIsInstance(res.exception, SystemExit)

    def test_method_flag_overriding_file_method_drops_file_bayesian_settings(self):
        # Like `duration`: the file's settings were written for its own method.
        src = self.dir / "bayes.json"
        src.write_text(
            json.dumps(
                {"method": "bayesian", "samples": 1000, "random_seed": 3, "variants": VARIANTS}
            ),
            encoding="utf-8",
        )
        res = self.invoke(["--input", src, "--no-config", "--method", "sequential"], self.dir)
        self.assertEqual(res.exit_code, 0, res.output)
        settings = json.loads(res.output)["analysis_settings"]
        self.assertEqual((settings["samples"], settings["random_seed"]), (None, None))

    def test_bootstrap_keeps_random_seed_for_non_bayesian(self):
        src = self.dir / "rows.csv"
        src.write_text(ROW_ARPU_CSV, encoding="utf-8")
        res = self.invoke(
            [
                "--input",
                src,
                "--no-config",
                "--primary-metric",
                "arpu",
                "--method",
                "sequential",
                "--bootstrap-samples",
                200,
                "--random-seed",
                3,
            ],
            self.dir,
        )
        self.assertEqual(res.exit_code, 0, res.output)
        settings = json.loads(res.output)["analysis_settings"]
        self.assertIsNone(settings["samples"])
        self.assertEqual(settings["random_seed"], 3)


if __name__ == "__main__":
    unittest.main()
