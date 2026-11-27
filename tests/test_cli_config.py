"""`[tool.ablift]` project config: --no-config and analysis_settings.config_source."""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from click.testing import CliRunner

from ablift.cli import cli
from ablift.schemas import get_schema

CSV_BASIC = "variant,visitors,conversions\ncontrol,1000,40\nv1,1000,52\n"
JSON_BASIC = {
    "experiment_name": "cfg",
    "variants": [
        {"name": "control", "visitors": 1000, "conversions": 40, "is_control": True},
        {"name": "v1", "visitors": 1000, "conversions": 52, "is_control": False},
    ],
}


@contextmanager
def chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class ProjectConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name).resolve()
        (self.dir / "in.csv").write_text(CSV_BASIC, encoding="utf-8")
        (self.dir / "in.json").write_text(json.dumps(JSON_BASIC), encoding="utf-8")

    def pyproject(self, body):
        path = self.dir / "pyproject.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def analyze(self, *args):
        with chdir(self.dir):
            result = CliRunner().invoke(cli, ["analyze", *map(str, args)])
        self.assertEqual(result.exit_code, 0, result.output)
        return json.loads(result.output)

    def test_config_source_reports_applied_pyproject_path(self):
        path = self.pyproject("[tool.ablift]\nalpha = 0.1\nsamples = 2000\n")
        for src in ("in.csv", "in.json"):
            with self.subTest(src=src):
                settings = self.analyze("--input", src)["analysis_settings"]
                self.assertEqual(settings["config_source"], str(path))
                self.assertEqual(settings["alpha"], 0.1)
                self.assertEqual(settings["samples"], 2000)

    def test_config_source_null_without_pyproject(self):
        settings = self.analyze("--input", "in.csv", "--samples", 2000)["analysis_settings"]
        self.assertIsNone(settings["config_source"])

    def test_config_source_null_when_pyproject_has_no_ablift_table(self):
        self.pyproject('[project]\nname = "x"\n')
        settings = self.analyze("--input", "in.csv", "--samples", 2000)["analysis_settings"]
        self.assertIsNone(settings["config_source"])

    def test_no_config_ignores_pyproject_completely(self):
        self.pyproject('[tool.ablift]\nalpha = 0.1\nsamples = 2000\nmethod = "fixed"\n')
        for src in ("in.csv", "in.json"):
            with self.subTest(src=src):
                data = self.analyze("--input", src, "--no-config")
                settings = data["analysis_settings"]
                self.assertIsNone(settings["config_source"])
                self.assertEqual(data["method"], "bayesian")
                self.assertEqual(settings["alpha"], 0.05)
                self.assertEqual(settings["samples"], 50000)

    def test_no_config_skips_malformed_pyproject(self):
        self.pyproject("[tool.ablift\nthis is not toml")
        settings = self.analyze("--input", "in.csv", "--no-config", "--samples", 2000)[
            "analysis_settings"
        ]
        self.assertIsNone(settings["config_source"])

    def test_config_source_is_in_output_schema(self):
        settings_schema = get_schema("analyze-output")["$defs"]["AnalysisSettings"]
        self.assertIn("config_source", settings_schema["required"])
        self.assertEqual(settings_schema["properties"]["config_source"]["type"], ["string", "null"])


if __name__ == "__main__":
    unittest.main()
