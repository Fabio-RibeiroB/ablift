import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from ablift.cli import cli
from ablift.connectors import _DURATION_FIELDS
from ablift.models import (
    SCHEMA_VERSION,
    AnalysisResult,
    AnalysisSettings,
    BayesianArmSummary,
    ComparisonResult,
    GuardrailResult,
    Recommendation,
    SecondaryMetricResult,
    SrmResult,
)
from ablift.planning import DurationPlan
from ablift.schemas import JSON_SCHEMA_DRAFT, SCHEMA_NAMES, get_all_schemas, get_schema


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise AssertionError(f"Unsupported JSON Schema type {expected!r}")


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise AssertionError(f"Unsupported external ref {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def validate_json_schema(
    instance: Any,
    schema: dict[str, Any],
    *,
    root: dict[str, Any] | None = None,
    path: str = "$",
) -> list[str]:
    """Validate ``instance`` against a JSON Schema Draft 2020-12 subset, returning error strings."""
    root_schema = root if root is not None else schema
    errors: list[str] = []

    if "$ref" in schema:
        target = _resolve_ref(schema["$ref"], root_schema)
        errors.extend(validate_json_schema(instance, target, root=root_schema, path=path))

    if "anyOf" in schema:
        sub_results = [
            validate_json_schema(instance, sub, root=root_schema, path=path)
            for sub in schema["anyOf"]
        ]
        if all(sub_results):
            errors.append(f"{path}: failed all anyOf branches")

    if "oneOf" in schema:
        matches = sum(
            1
            for sub in schema["oneOf"]
            if not validate_json_schema(instance, sub, root=root_schema, path=path)
        )
        if matches != 1:
            errors.append(f"{path}: expected exactly 1 oneOf match, got {matches}")

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum {schema['enum']!r}")

    if "type" in schema:
        expected_types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_matches_type(instance, t) for t in expected_types):
            errors.append(
                f"{path}: expected type {expected_types!r}, got {type(instance).__name__} ({instance!r})"
            )
            return errors

    if isinstance(instance, str) and "minLength" in schema and len(instance) < schema["minLength"]:
        errors.append(f"{path}: string length {len(instance)} < minLength {schema['minLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {instance} >= exclusiveMaximum {schema['exclusiveMaximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: list length {len(instance)} < minItems {schema['minItems']}")
        if "items" in schema:
            for idx, item in enumerate(instance):
                errors.extend(
                    validate_json_schema(
                        item, schema["items"], root=root_schema, path=f"{path}[{idx}]"
                    )
                )

    if isinstance(instance, dict):
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errors.append(
                f"{path}: object has {len(instance)} properties < minProperties {schema['minProperties']}"
            )
        for req_key in schema.get("required", []):
            if req_key not in instance:
                errors.append(f"{path}: missing required key {req_key!r}")
        props = schema.get("properties", {})
        addl = schema.get("additionalProperties", True)
        for key, val in instance.items():
            if key in props:
                errors.extend(
                    validate_json_schema(val, props[key], root=root_schema, path=f"{path}.{key}")
                )
            elif addl is False:
                errors.append(f"{path}: unexpected additional property {key!r}")
            elif isinstance(addl, dict):
                errors.extend(
                    validate_json_schema(val, addl, root=root_schema, path=f"{path}.{key}")
                )

    return errors


class SchemaCliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_each_schema_target_emits_valid_draft_2020_12(self):
        for name in SCHEMA_NAMES:
            res = self.runner.invoke(cli, ["schema", name])
            self.assertEqual(res.exit_code, 0, res.output)
            doc = json.loads(res.output)
            self.assertEqual(doc["$schema"], JSON_SCHEMA_DRAFT)
            self.assertIn("$id", doc)
            self.assertIn(f"v{SCHEMA_VERSION}", doc["$id"])
            self.assertEqual(doc["type"], "object")

    def test_schema_all_emits_every_contract(self):
        res = self.runner.invoke(cli, ["schema", "--all"])
        self.assertEqual(res.exit_code, 0, res.output)
        bundle = json.loads(res.output)
        self.assertEqual(bundle["schema_version"], SCHEMA_VERSION)
        self.assertEqual(tuple(bundle["schemas"].keys()), SCHEMA_NAMES)
        for name in SCHEMA_NAMES:
            self.assertEqual(bundle["schemas"][name]["$schema"], JSON_SCHEMA_DRAFT)

    def test_schema_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "nested" / "analyze-output.schema.json"
            res = self.runner.invoke(cli, ["schema", "analyze-output", "--output", str(out_path)])
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertEqual(res.output, "")
            doc = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(doc["title"], "ablift analyze output")

    def test_schema_usage_errors(self):
        res_none = self.runner.invoke(cli, ["schema"])
        self.assertEqual(res_none.exit_code, 2)
        self.assertIn("--all", res_none.output)

        res_both = self.runner.invoke(cli, ["schema", "analyze-input", "--all"])
        self.assertEqual(res_both.exit_code, 2)
        self.assertIn("not both", res_both.output)

        res_bad = self.runner.invoke(cli, ["schema", "not-a-schema"])
        self.assertEqual(res_bad.exit_code, 2)

    def test_get_schema_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown schema"):
            get_schema("nope")

    def test_analyze_help_reflects_json_and_csv_overrides(self):
        res = self.runner.invoke(cli, ["analyze", "--help"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertNotIn("for CSV/XLSX input", res.output)
        self.assertIn("JSON, CSV/XLSX, and stdin", res.output)


class SchemaValidationAgainstRealPayloadsTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.schemas = get_all_schemas()["schemas"]

    def _assert_valid(self, instance: Any, schema_name: str) -> None:
        errors = validate_json_schema(instance, self.schemas[schema_name])
        self.assertEqual(errors, [], f"{schema_name} validation errors: {errors}")

    def test_example_input_and_its_analyze_output_validate(self):
        ex = self.runner.invoke(cli, ["example-input"])
        self.assertEqual(ex.exit_code, 0, ex.output)
        example_payload = json.loads(ex.output)
        self._assert_valid(example_payload, "analyze-input")

        an = self.runner.invoke(
            cli, ["analyze", "--input", "-", "--samples", "2000"], input=ex.output
        )
        self.assertEqual(an.exit_code, 0, an.output)
        self._assert_valid(json.loads(an.output), "analyze-output")

    def test_analyze_inputs_and_outputs_all_methods_and_metrics(self):
        abn_conv = {
            "experiment_name": "abn_conv",
            "method": "bayesian",
            "primary_metric": "conversion_rate",
            "expected_split": {"control": 0.5, "v1": 0.25, "v2": 0.25},
            "variants": [
                {"name": "control", "visitors": 20000, "conversions": 800, "is_control": True},
                {"name": "v1", "visitors": 10000, "conversions": 460, "is_control": False},
                {"name": "v2", "visitors": 10000, "conversions": 500, "is_control": False},
            ],
            "guardrails": [
                {
                    "name": "bounce_rate",
                    "metric": "conversion_rate",
                    "direction": "decrease",
                    "max_relative_change": 0.03,
                    "variants": [
                        {"name": "control", "conversions": 6000},
                        {"name": "v1", "conversions": 3020},
                        {"name": "v2", "conversions": 3400},
                    ],
                },
                {
                    "name": "p95_latency_ms",
                    "control": 400.0,
                    "treatment": 405.0,
                    "direction": "decrease",
                    "max_relative_change": 0.05,
                },
            ],
            "secondary_metrics": [
                {
                    "name": "revenue_per_visitor",
                    "metric": "arpu",
                    "variants": [
                        {
                            "name": "control",
                            "revenue_sum": 100000.0,
                            "revenue_sum_squares": 2000000.0,
                        },
                        {
                            "name": "v1",
                            "revenue_sum": 55000.0,
                            "revenue_sum_squares": 1100000.0,
                        },
                        {
                            "name": "v2",
                            "revenue_sum": 56000.0,
                            "revenue_sum_squares": 1120000.0,
                        },
                    ],
                }
            ],
            "decision_policy": {
                "enabled": True,
                "bayes_prob_beats_control": 0.95,
                "max_expected_loss": 0.005,
            },
            "samples": 2000,
            "random_seed": 7,
        }
        self._assert_valid(abn_conv, "analyze-input")

        # Bayesian with policy, Bayesian without policy, sequential, and fixed
        for extra_args in (
            [],
            ["--no-recommend"],
            ["--method", "sequential"],
            ["--method", "fixed"],
        ):
            payload = dict(abn_conv)
            if "--method" in extra_args and "fixed" in extra_args:
                payload.pop("decision_policy", None)
                payload.pop("samples", None)
                payload.pop("random_seed", None)
            res = self.runner.invoke(
                cli, ["analyze", "--input", "-", *extra_args], input=json.dumps(payload)
            )
            self.assertEqual(res.exit_code, 0, f"{extra_args}: {res.output}")
            self._assert_valid(json.loads(res.output), "analyze-output")

        # ARPU primary metric with expected_share across all three methods
        arpu_payload = {
            "experiment_name": "arpu_test",
            "method": "bayesian",
            "primary_metric": "arpu",
            "variants": [
                {
                    "name": "control",
                    "visitors": 2000,
                    "revenue_sum": 10000.0,
                    "revenue_sum_squares": 150000.0,
                    "is_control": True,
                    "expected_share": 0.5,
                },
                {
                    "name": "v1",
                    "visitors": 2000,
                    "revenue_sum": 11200.0,
                    "revenue_sum_squares": 175000.0,
                    "is_control": False,
                    "expected_share": 0.5,
                },
            ],
            "samples": 2000,
        }
        self._assert_valid(arpu_payload, "analyze-input")
        for method in ("bayesian", "sequential", "fixed"):
            p = dict(arpu_payload)
            p["method"] = method
            if method != "bayesian":
                p.pop("samples", None)
            res = self.runner.invoke(cli, ["analyze", "--input", "-"], input=json.dumps(p))
            self.assertEqual(res.exit_code, 0, f"{method}: {res.output}")
            self._assert_valid(json.loads(res.output), "analyze-output")

    def test_duration_inputs_and_outputs_all_methods_and_metrics(self):
        cases = [
            {
                "metric": "conversion_rate",
                "method": "sequential",
                "baseline_rate": 0.04,
                "relative_mde": 0.05,
                "daily_traffic": 50000,
                "n_variants": 3,
                "alpha": 0.05,
                "power": 0.8,
            },
            {
                "metric": "conversion_rate",
                "method": "fixed",
                "baseline_rate": 0.04,
                "relative_mde": 0.05,
                "daily_traffic": 50000,
            },
            {
                "metric": "conversion_rate",
                "method": "bayesian",
                "baseline_rate": 0.04,
                "relative_mde": 0.05,
                "daily_traffic": 100,
                "max_days": 1,
            },
            {
                "metric": "arpu",
                "method": "sequential",
                "baseline_mean": 12.5,
                "baseline_sd": 25.0,
                "relative_mde": 0.05,
                "daily_traffic": 50000,
            },
            {
                "metric": "arpu",
                "method": "fixed",
                "baseline_mean": 12.5,
                "baseline_cv": 2.0,
                "relative_mde": 0.05,
                "daily_traffic": 50000,
            },
            {
                "metric": "arpu",
                "method": "bayesian",
                "baseline_mean": 12.5,
                "baseline_sd": 25.0,
                "relative_mde": 0.1,
                "daily_traffic": 50000,
                "max_expected_loss": 0.05,
                "max_days": 2,
            },
        ]
        for req in cases:
            self._assert_valid(req, "duration-input")
            res = self.runner.invoke(cli, ["duration", "--input", "-"], input=json.dumps(req))
            self.assertEqual(res.exit_code, 0, f"{req}: {res.output}")
            self._assert_valid(json.loads(res.output), "duration-output")

    def test_schemas_reject_invalid_payloads(self):
        self.assertTrue(validate_json_schema({}, self.schemas["analyze-input"]))
        self.assertTrue(
            validate_json_schema(
                {
                    "method": "invalid",
                    "variants": [{"name": "a", "visitors": 10}, {"name": "b", "visitors": 10}],
                },
                self.schemas["analyze-input"],
            )
        )
        self.assertTrue(
            validate_json_schema(
                {"variants": [{"name": "a", "visitors": 0}, {"name": "b", "visitors": 10}]},
                self.schemas["analyze-input"],
            )
        )
        self.assertTrue(
            validate_json_schema(
                {"unknown_key": 123, "baseline_rate": 0.04},
                self.schemas["duration-input"],
            )
        )
        self.assertTrue(
            validate_json_schema(
                {"schema_version": "999"},
                self.schemas["analyze-output"],
            )
        )


class SchemaModelSyncTests(unittest.TestCase):
    def test_analyze_output_schema_stays_in_sync_with_models(self):
        out_schema = get_schema("analyze-output")
        defs = out_schema["$defs"]

        expected_top = {"schema_version", *(f.name for f in dataclasses.fields(AnalysisResult))}
        self.assertEqual(set(out_schema["properties"]), expected_top)
        self.assertEqual(set(out_schema["required"]), expected_top)

        for cls, def_name in (
            (AnalysisSettings, "AnalysisSettings"),
            (ComparisonResult, "ComparisonResult"),
            (BayesianArmSummary, "BayesianArmSummary"),
            (GuardrailResult, "GuardrailResult"),
            (SecondaryMetricResult, "SecondaryMetricResult"),
            (SrmResult, "SrmResult"),
            (Recommendation, "Recommendation"),
        ):
            model_fields = {f.name for f in dataclasses.fields(cls)}
            self.assertEqual(set(defs[def_name]["properties"]), model_fields, def_name)
            self.assertEqual(set(defs[def_name]["required"]), model_fields, def_name)

    def test_duration_schemas_stay_in_sync_with_models_and_connectors(self):
        in_schema = get_schema("duration-input")
        self.assertEqual(set(in_schema["properties"]), set(_DURATION_FIELDS))

        out_schema = get_schema("duration-output")
        expected_out = {"schema_version", *(f.name for f in dataclasses.fields(DurationPlan))}
        self.assertEqual(set(out_schema["properties"]), expected_out)
        self.assertEqual(set(out_schema["required"]), expected_out)


if __name__ == "__main__":
    unittest.main()
