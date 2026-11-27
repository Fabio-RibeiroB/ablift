from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import SCHEMA_VERSION

JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

SCHEMA_NAMES: tuple[str, ...] = (
    "analyze-input",
    "analyze-output",
    "duration-input",
    "duration-output",
    "power-output",
)


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON Schema accepting either ``schema`` or ``null``."""
    if (
        "type" in schema
        and isinstance(schema["type"], str)
        and set(schema)
        <= {
            "type",
            "description",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minLength",
            "enum",
        }
    ):
        out = dict(schema)
        out["type"] = [schema["type"], "null"]
        return out
    return {"anyOf": [schema, {"type": "null"}]}


def _variant_input_schema(*, submetric: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "name": {
            "type": "string",
            "minLength": 1,
            "description": "Variant identifier (unique across variants).",
        },
        "visitors": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Denominator units exposed to this variant. "
                + (
                    "Optional for statistical guardrails and secondary metrics; defaults to the "
                    "matching primary variant's visitors."
                    if submetric
                    else "Required positive integer."
                )
            ),
        },
        "conversions": {
            "type": "integer",
            "minimum": 0,
            "description": (
                "Binary success count (0 <= conversions <= visitors). "
                "Required when metric is 'conversion_rate'."
            ),
        },
        "revenue_sum": {
            "type": ["number", "null"],
            "description": "Sum of per-visitor revenue. Required when metric is 'arpu'.",
        },
        "revenue_sum_squares": {
            "type": ["number", "null"],
            "description": (
                "Sum of squared per-visitor revenue (for variance estimation). "
                "Required when metric is 'arpu'."
            ),
        },
        "numerator_sum": {
            "type": ["number", "null"],
            "description": "Sum of per-unit numerators. Required when metric is 'ratio'.",
        },
        "numerator_sum_squares": {
            "type": ["number", "null"],
            "description": "Sum of squared per-unit numerators. Required when metric is 'ratio'.",
        },
        "denominator_sum": {
            "type": ["number", "null"],
            "description": (
                "Sum of per-unit denominators (must be positive). Required when metric is 'ratio'."
            ),
        },
        "denominator_sum_squares": {
            "type": ["number", "null"],
            "description": "Sum of squared per-unit denominators. Required when metric is 'ratio'.",
        },
        "numerator_denominator_sum": {
            "type": ["number", "null"],
            "description": (
                "Sum of per-unit numerator * denominator cross-products. "
                "Required when metric is 'ratio'."
            ),
        },
        "is_control": {
            "type": "boolean",
            "description": (
                "Whether this variant is the control arm. "
                + (
                    "Inherited from primary variants if omitted; must not contradict primary."
                    if submetric
                    else "Exactly one variant must set is_control=true."
                )
            ),
        },
    }
    if not submetric:
        properties["expected_share"] = {
            "type": ["number", "null"],
            "exclusiveMinimum": 0,
            "description": (
                "Optional positive traffic-split weight for SRM checks. If used, must be set "
                "on every variant and cannot be combined with top-level expected_split."
            ),
        }
        properties["covariate_sum"] = {
            "type": ["number", "null"],
            "description": (
                "Optional sum of pre-experiment covariate X for CUPED variance reduction. "
                "Must be provided alongside covariate_sum_squares and outcome_covariate_sum on every variant."
            ),
        }
        properties["covariate_sum_squares"] = {
            "type": ["number", "null"],
            "description": (
                "Optional sum of squared pre-experiment covariate X^2 for CUPED variance reduction."
            ),
        }
        properties["outcome_covariate_sum"] = {
            "type": ["number", "null"],
            "description": (
                "Optional sum of outcome * pre-experiment covariate (Y * X) for CUPED variance reduction."
            ),
        }

    return {
        "type": "object",
        "properties": properties,
        "required": ["name"] if submetric else ["name", "visitors"],
        "additionalProperties": False,
    }


def _comparison_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "treatment": {"type": "string"},
            "control": {"type": "string"},
            "metric": {"type": "string", "enum": ["conversion_rate", "arpu", "ratio"]},
            "control_rate": {"type": "number"},
            "treatment_rate": {"type": "number"},
            "absolute_lift": {"type": "number"},
            "relative_lift": {"type": "number"},
            "p_value": {"type": ["number", "null"]},
            "raw_p_value": {"type": ["number", "null"]},
            "test_statistic": {"type": ["number", "null"]},
            "degrees_of_freedom": {"type": ["number", "null"]},
            "e_value": {"type": ["number", "null"]},
            "e_threshold": {"type": ["number", "null"]},
            "significant": {"type": ["boolean", "null"]},
            "probability_beats_control": {"type": ["number", "null"]},
            "expected_loss": {"type": ["number", "null"]},
            "probability_best": {"type": ["number", "null"]},
            "multi_arm_expected_loss": {"type": ["number", "null"]},
            "ci_low": {"type": ["number", "null"]},
            "ci_high": {"type": ["number", "null"]},
            "relative_ci_low": {"type": ["number", "null"]},
            "relative_ci_high": {"type": ["number", "null"]},
        },
        "required": [
            "treatment",
            "control",
            "metric",
            "control_rate",
            "treatment_rate",
            "absolute_lift",
            "relative_lift",
            "p_value",
            "raw_p_value",
            "test_statistic",
            "degrees_of_freedom",
            "e_value",
            "e_threshold",
            "significant",
            "probability_beats_control",
            "expected_loss",
            "probability_best",
            "multi_arm_expected_loss",
            "ci_low",
            "ci_high",
            "relative_ci_low",
            "relative_ci_high",
        ],
        "additionalProperties": False,
    }


def _bayesian_arm_summary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "variant": {"type": "string"},
            "is_control": {"type": "boolean"},
            "probability_best": {"type": "number", "minimum": 0, "maximum": 1},
            "multi_arm_expected_loss": {"type": "number", "minimum": 0},
        },
        "required": ["variant", "is_control", "probability_best", "multi_arm_expected_loss"],
        "additionalProperties": False,
    }


def build_analyze_input_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``ablift analyze`` JSON input payloads."""
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "$id": f"https://ablift.dev/schemas/v{SCHEMA_VERSION}/analyze-input.json",
        "title": "ablift analyze input",
        "description": (
            f"Input JSON contract for 'ablift analyze' (schema_version {SCHEMA_VERSION})."
        ),
        "type": "object",
        "$defs": {
            "VariantInput": _variant_input_schema(submetric=False),
            "SubmetricVariantInput": _variant_input_schema(submetric=True),
            "ThresholdGuardrailInput": {
                "type": "object",
                "description": (
                    "Backwards-compatible deterministic threshold guardrail comparing two point "
                    "floats (control vs treatment)."
                ),
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "control": {"type": "number"},
                    "treatment": {"type": "number"},
                    "direction": {
                        "type": "string",
                        "enum": ["decrease", "increase"],
                        "default": "decrease",
                        "description": (
                            "'decrease' means lower is better (fails if treatment increases "
                            "beyond max_relative_change); 'increase' means higher is better."
                        ),
                    },
                    "max_relative_change": {
                        "type": "number",
                        "minimum": 0,
                        "default": 0.0,
                        "description": "Maximum tolerated relative regression (e.g. 0.03 for 3%).",
                    },
                },
                "required": ["name", "control", "treatment"],
                "additionalProperties": False,
            },
            "StatisticalGuardrailInput": {
                "type": "object",
                "description": (
                    "Per-variant statistical guardrail evaluated with the active inference method "
                    "(bayesian, sequential, or fixed). Failures exclude only the failing variant(s) "
                    "in A/B/n recommendations."
                ),
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "metric": {"type": "string", "enum": ["conversion_rate", "arpu", "ratio"]},
                    "direction": {
                        "type": "string",
                        "enum": ["decrease", "increase"],
                        "default": "decrease",
                    },
                    "max_relative_change": {
                        "type": "number",
                        "minimum": 0,
                        "default": 0.0,
                    },
                    "variants": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"$ref": "#/$defs/SubmetricVariantInput"},
                    },
                },
                "required": ["name", "metric", "variants"],
                "additionalProperties": False,
            },
            "SecondaryMetricInput": {
                "type": "object",
                "description": (
                    "Informational metric evaluated alongside the primary metric. "
                    "Does not block recommendations."
                ),
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "metric": {"type": "string", "enum": ["conversion_rate", "arpu", "ratio"]},
                    "variants": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"$ref": "#/$defs/SubmetricVariantInput"},
                    },
                },
                "required": ["name", "metric", "variants"],
                "additionalProperties": False,
            },
            "DecisionPolicyInput": {
                "type": "object",
                "description": (
                    "Opt-in recommendation policy. Set enabled=true for any method to emit a "
                    "recommendation; threshold fields apply in Bayesian mode."
                ),
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "bayes_prob_beats_control": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0.95,
                    },
                    "max_expected_loss": {
                        "type": "number",
                        "minimum": 0,
                        "default": 0.001,
                    },
                },
                "additionalProperties": False,
            },
        },
        "properties": {
            "experiment_name": {
                "type": "string",
                "default": "experiment",
                "description": "Human-readable experiment identifier.",
            },
            "method": {
                "type": "string",
                "enum": ["bayesian", "sequential", "fixed"],
                "default": "bayesian",
                "description": (
                    "Inference method: 'bayesian', 'sequential' (always-valid mSPRT), or "
                    "'fixed' (single-look frequentist)."
                ),
            },
            "primary_metric": {
                "type": "string",
                "enum": ["conversion_rate", "arpu", "ratio"],
                "default": "conversion_rate",
                "description": "Primary metric model.",
            },
            "alpha": {
                "type": "number",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1,
                "default": 0.05,
                "description": "Family-wise Type I error rate (Bonferroni-adjusted for A/B/n).",
            },
            "sequential_tau": {
                "type": ["number", "null"],
                "exclusiveMinimum": 0,
                "default": None,
                "description": (
                    "mSPRT mixing parameter (prior std on absolute effect). Only applies when "
                    "method='sequential'; defaults to 10% of the control baseline."
                ),
            },
            "variants": {
                "type": "array",
                "minItems": 2,
                "items": {"$ref": "#/$defs/VariantInput"},
                "description": "Experiment arms (at least 2; exactly one with is_control=true).",
            },
            "expected_split": {
                "anyOf": [
                    {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "number", "exclusiveMinimum": 0},
                    },
                    {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": {"type": "number", "exclusiveMinimum": 0},
                    },
                    {"type": "null"},
                ],
                "default": None,
                "description": (
                    "Expected traffic allocation weights for SRM checks, either in variant order "
                    "or mapped by variant name. Weights are normalized to sum to 1.0."
                ),
            },
            "guardrails": {
                "type": "array",
                "default": [],
                "items": {
                    "oneOf": [
                        {"$ref": "#/$defs/StatisticalGuardrailInput"},
                        {"$ref": "#/$defs/ThresholdGuardrailInput"},
                    ]
                },
                "description": "Guardrail checks (per-variant statistical or legacy threshold).",
            },
            "secondary_metrics": {
                "type": "array",
                "default": [],
                "items": {"$ref": "#/$defs/SecondaryMetricInput"},
                "description": "Optional secondary metrics evaluated for full-experiment readouts.",
            },
            "decision_policy": _nullable({"$ref": "#/$defs/DecisionPolicyInput"}),
            "decision_thresholds": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "bayes_prob_beats_control": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "exclusiveMaximum": 1,
                            },
                            "max_expected_loss": {"type": "number", "minimum": 0},
                        },
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ],
                "deprecated": True,
                "description": "Legacy alias for an enabled decision_policy.",
            },
            "samples": {
                "type": "integer",
                "minimum": 1,
                "default": 50000,
                "description": (
                    "Posterior Monte Carlo sample count. Only applies when method='bayesian'; "
                    "rejected for sequential and fixed."
                ),
            },
            "random_seed": {
                "type": "integer",
                "default": 7,
                "description": (
                    "Random seed for reproducible Bayesian sampling. Only applies when "
                    "method='bayesian'; rejected for sequential and fixed."
                ),
            },
            "input_interpretation": {
                "type": "object",
                "description": "Optional metadata describing how tabular inputs were parsed.",
            },
        },
        "required": ["variants"],
        "additionalProperties": False,
    }


def build_analyze_output_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``ablift analyze`` JSON output payloads."""
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "$id": f"https://ablift.dev/schemas/v{SCHEMA_VERSION}/analyze-output.json",
        "title": "ablift analyze output",
        "description": (
            f"Output JSON contract for 'ablift analyze' (schema_version {SCHEMA_VERSION})."
        ),
        "type": "object",
        "$defs": {
            "AnalysisSettings": {
                "type": "object",
                "properties": {
                    "primary_metric": {
                        "type": "string",
                        "enum": ["conversion_rate", "arpu", "ratio"],
                    },
                    "alpha": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    "multiple_comparisons": {
                        "type": "object",
                        "properties": {
                            "correction": {"type": "string", "enum": ["none", "bonferroni"]},
                            "n_comparisons": {"type": "integer", "minimum": 1},
                            "alpha_per_comparison": {"type": "number", "exclusiveMinimum": 0},
                        },
                        "required": ["correction", "n_comparisons"],
                    },
                    "sequential_tau": {"type": ["number", "null"]},
                    "samples": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Posterior sample count; null unless method is bayesian.",
                    },
                    "random_seed": {
                        "type": ["integer", "null"],
                        "description": (
                            "Seed for Bayesian sampling or the row-level ARPU bootstrap; null "
                            "when neither ran."
                        ),
                    },
                    "priors": {"type": "object"},
                    "decision_policy": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "source": {"type": "string"},
                            "thresholds": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "bayes_prob_beats_control": {"type": "number"},
                                            "max_expected_loss": {"type": "number"},
                                        },
                                        "required": [
                                            "bayes_prob_beats_control",
                                            "max_expected_loss",
                                        ],
                                    },
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["enabled", "source", "thresholds"],
                    },
                    "input_interpretation": {"type": "object"},
                    "config_source": {
                        "type": ["string", "null"],
                        "description": (
                            "Absolute path of the pyproject.toml whose [tool.ablift] table was "
                            "applied, or null when none was found or --no-config was given."
                        ),
                    },
                },
                "required": [
                    "primary_metric",
                    "alpha",
                    "multiple_comparisons",
                    "sequential_tau",
                    "samples",
                    "random_seed",
                    "priors",
                    "decision_policy",
                    "input_interpretation",
                    "config_source",
                ],
                "additionalProperties": False,
            },
            "ComparisonResult": _comparison_result_schema(),
            "BayesianArmSummary": _bayesian_arm_summary_schema(),
            "GuardrailResult": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "direction": {"type": "string", "enum": ["decrease", "increase"]},
                    "control": {"type": "number"},
                    "treatment": {"type": "number"},
                    "relative_change": {"type": "number"},
                    "max_relative_change": {"type": "number", "minimum": 0},
                    "passed": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "variant": {"type": ["string", "null"]},
                    "mode": {"type": "string", "enum": ["threshold", "statistical"]},
                    "comparison": _nullable({"$ref": "#/$defs/ComparisonResult"}),
                },
                "required": [
                    "name",
                    "direction",
                    "control",
                    "treatment",
                    "relative_change",
                    "max_relative_change",
                    "passed",
                    "reason",
                    "variant",
                    "mode",
                    "comparison",
                ],
                "additionalProperties": False,
            },
            "SecondaryMetricResult": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "metric": {"type": "string", "enum": ["conversion_rate", "arpu", "ratio"]},
                    "comparisons": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/ComparisonResult"},
                    },
                    "bayesian_arms": _nullable(
                        {
                            "type": "array",
                            "items": {"$ref": "#/$defs/BayesianArmSummary"},
                        }
                    ),
                },
                "required": ["name", "metric", "comparisons", "bayesian_arms"],
                "additionalProperties": False,
            },
            "SrmResult": {
                "type": "object",
                "properties": {
                    "passed": {"type": "boolean"},
                    "p_value": {"type": "number", "minimum": 0, "maximum": 1},
                    "observed": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "expected": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "expected_proportions": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "passed",
                    "p_value",
                    "observed",
                    "expected",
                    "expected_proportions",
                    "reason",
                ],
                "additionalProperties": False,
            },
            "Recommendation": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "Recommended action: 'ship_<variant>', 'continue_collecting_data', "
                            "'do_not_ship', 'investigate_data_quality', or 'stop_and_rollback'."
                        ),
                    },
                    "rationale": {"type": "string"},
                    "decision_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "next_best_action": {"type": "string"},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "action",
                    "rationale",
                    "decision_confidence",
                    "next_best_action",
                    "risk_flags",
                ],
                "additionalProperties": False,
            },
        },
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "experiment_name": {"type": "string"},
            "method": {"type": "string", "enum": ["bayesian", "sequential", "fixed"]},
            "analysis_settings": {"$ref": "#/$defs/AnalysisSettings"},
            "control_variant": {"type": "string"},
            "comparisons": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/ComparisonResult"},
            },
            "guardrails_passed": {"type": "boolean"},
            "guardrails": {"type": "array", "items": {"$ref": "#/$defs/GuardrailResult"}},
            "srm": {"$ref": "#/$defs/SrmResult"},
            "risk_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Experiment and data-quality risk flags (always present, possibly empty, even when "
                    "recommendation is null)."
                ),
            },
            "recommendation": _nullable({"$ref": "#/$defs/Recommendation"}),
            "bayesian_arms": _nullable(
                {
                    "type": "array",
                    "minItems": 2,
                    "items": {"$ref": "#/$defs/BayesianArmSummary"},
                }
            ),
            "secondary_metrics": {
                "type": "array",
                "items": {"$ref": "#/$defs/SecondaryMetricResult"},
            },
        },
        "required": [
            "schema_version",
            "experiment_name",
            "method",
            "analysis_settings",
            "control_variant",
            "comparisons",
            "guardrails_passed",
            "guardrails",
            "srm",
            "risk_flags",
            "recommendation",
            "bayesian_arms",
            "secondary_metrics",
        ],
        "additionalProperties": False,
    }


def build_duration_input_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``ablift duration`` JSON/CSV input payloads."""
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "$id": f"https://ablift.dev/schemas/v{SCHEMA_VERSION}/duration-input.json",
        "title": "ablift duration input",
        "description": (
            f"Input contract for 'ablift duration --input' (JSON object or named CSV/XLSX "
            f"columns; schema_version {SCHEMA_VERSION}). CLI flags override input fields."
        ),
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": ["conversion_rate", "arpu"],
                "default": "conversion_rate",
                "description": "Metric model to plan for.",
            },
            "method": {
                "type": "string",
                "enum": ["sequential", "fixed", "bayesian"],
                "description": (
                    "Duration planning method. No default: set it here or pass --method."
                ),
            },
            "baseline_rate": {
                "type": "number",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1,
                "description": "Baseline conversion rate (required for metric='conversion_rate').",
            },
            "baseline_mean": {
                "type": "number",
                "description": "Non-zero baseline ARPU mean (required for metric='arpu').",
            },
            "baseline_sd": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": (
                    "Baseline ARPU standard deviation (for metric='arpu'; "
                    "mutually exclusive with baseline_cv)."
                ),
            },
            "baseline_cv": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": (
                    "Baseline ARPU coefficient of variation sd / |mean| (for metric='arpu'; "
                    "mutually exclusive with baseline_sd)."
                ),
            },
            "relative_mde": {
                "type": "number",
                "description": "Non-zero relative minimum detectable effect (e.g. 0.05 for +5%).",
            },
            "daily_traffic": {
                "type": "integer",
                "minimum": 1,
                "description": "Total daily traffic across all variants.",
            },
            "n_variants": {
                "type": "integer",
                "minimum": 2,
                "default": 2,
                "description": "Number of variants including control.",
            },
            "expected_split": {
                "anyOf": [
                    {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "number", "exclusiveMinimum": 0},
                    },
                    {"type": "string", "minLength": 1},
                    {"type": "null"},
                ],
                "default": None,
                "description": (
                    "Expected traffic allocation weights in variant order (control first), "
                    "either as an array of positive numbers (e.g. [0.9, 0.1] or [90, 10]) or "
                    "comma-separated weights (e.g. '0.9,0.1' in CSV/XLSX). Must have length "
                    "n_variants; weights are normalized to sum to 1.0."
                ),
            },
            "alpha": {
                "type": "number",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1,
                "default": 0.05,
                "description": "Family-wise alpha (sequential and fixed methods only).",
            },
            "power": {
                "type": "number",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1,
                "default": 0.8,
                "description": "Target statistical power (sequential and fixed methods only).",
            },
            "sequential_tau": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "mSPRT mixing parameter (sequential method only).",
            },
            "prob_threshold": {
                "type": "number",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1,
                "default": 0.95,
                "description": "Bayesian P(win) threshold (bayesian method only).",
            },
            "max_expected_loss": {
                "type": "number",
                "minimum": 0,
                "default": 0.001,
                "description": "Bayesian expected-loss threshold (bayesian method only).",
            },
            "assurance_target": {
                "type": "number",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1,
                "default": 0.8,
                "description": (
                    "Bayesian probability of reaching a decision within max_days "
                    "(bayesian method only)."
                ),
            },
            "max_days": {
                "type": "integer",
                "minimum": 1,
                "default": 60,
                "description": "Maximum search horizon in days (bayesian method only).",
            },
        },
        "additionalProperties": False,
    }


def build_duration_output_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``ablift duration`` JSON output payloads."""
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "$id": f"https://ablift.dev/schemas/v{SCHEMA_VERSION}/duration-output.json",
        "title": "ablift duration output",
        "description": (
            f"Output JSON contract for 'ablift duration' (schema_version {SCHEMA_VERSION})."
        ),
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "method": {
                "type": "string",
                "enum": ["sequential", "fixed_horizon", "bayesian"],
            },
            "estimated_days": {
                "type": ["number", "null"],
                "exclusiveMinimum": 0,
                "description": (
                    "Estimated runtime in days, or null if Bayesian assurance target was not "
                    "reached within max_days."
                ),
            },
            "n_per_variant": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": (
                    "Required sample size for the control arm (equal to every arm under an "
                    "equal split), or null if Bayesian assurance target was not reached "
                    "within max_days."
                ),
            },
            "n_by_variant": _nullable(
                {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "integer", "minimum": 1},
                    "description": (
                        "Required sample size for each variant in variant order (control first), "
                        "or null if Bayesian assurance target was not reached within max_days."
                    ),
                }
            ),
            "n_total": {
                "type": ["integer", "null"],
                "minimum": 2,
                "description": (
                    "Required total sample size across all variants (sum of n_by_variant), "
                    "or null if Bayesian assurance target was not reached within max_days."
                ),
            },
            "assumptions": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["conversion_rate", "arpu"]},
                    "relative_mde": {"type": "number"},
                    "daily_total_traffic": {"type": "integer", "minimum": 1},
                    "n_variants": {"type": "integer", "minimum": 2},
                    "expected_split": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    },
                },
                "required": ["metric", "relative_mde", "daily_total_traffic", "n_variants"],
            },
            "diagnostics": {"type": "object"},
        },
        "required": [
            "schema_version",
            "method",
            "estimated_days",
            "n_per_variant",
            "n_by_variant",
            "n_total",
            "assumptions",
            "diagnostics",
        ],
        "additionalProperties": False,
    }


def build_power_output_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``ablift power`` JSON output payloads."""
    return {
        "$schema": JSON_SCHEMA_DRAFT,
        "$id": f"https://ablift.dev/schemas/v{SCHEMA_VERSION}/power-output.json",
        "title": "ablift power output",
        "description": f"Output JSON contract for 'ablift power' (schema_version {SCHEMA_VERSION}).",
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "solved_for": {
                "type": "string",
                "enum": ["power", "relative_mde"],
                "description": "Which quantity was computed from the others.",
            },
            "method": {"type": "string", "enum": ["fixed_horizon", "sequential"]},
            "power": {"type": "number", "minimum": 0, "maximum": 1},
            "relative_mde": {"type": "number", "exclusiveMinimum": 0},
            "absolute_mde": {"type": "number", "minimum": 0},
            "n_per_variant": {
                "type": "integer",
                "minimum": 2,
                "description": (
                    "Control arm sample size (equal to every arm under an equal split)."
                ),
            },
            "n_by_variant": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "integer", "minimum": 2},
                "description": "Sample size for each variant in variant order (control first).",
            },
            "n_total": {
                "type": "integer",
                "minimum": 4,
                "description": "Total sample size across all variants (sum of n_by_variant).",
            },
            "assumptions": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["conversion_rate", "arpu"]},
                    "n_per_variant": {"type": "integer", "minimum": 2},
                    "n_total": {"type": "integer", "minimum": 4},
                    "n_variants": {"type": "integer", "minimum": 2},
                    "expected_split": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    },
                    "alpha": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    "alpha_per_comparison": {"type": "number"},
                },
                "required": ["metric", "n_per_variant", "n_total", "n_variants", "alpha"],
            },
        },
        "required": [
            "schema_version",
            "solved_for",
            "method",
            "power",
            "relative_mde",
            "absolute_mde",
            "n_per_variant",
            "n_by_variant",
            "n_total",
            "assumptions",
        ],
        "additionalProperties": False,
    }


_SCHEMA_BUILDERS = {
    "analyze-input": build_analyze_input_schema,
    "analyze-output": build_analyze_output_schema,
    "duration-input": build_duration_input_schema,
    "duration-output": build_duration_output_schema,
    "power-output": build_power_output_schema,
}


def get_schema(name: str) -> dict[str, Any]:
    """Return a fresh copy of the named JSON Schema."""
    builder = _SCHEMA_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown schema {name!r}. Choose from: {', '.join(SCHEMA_NAMES)}.")
    return deepcopy(builder())


def get_all_schemas() -> dict[str, Any]:
    """Return all schemas bundled under ``schema_version``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "schemas": {name: get_schema(name) for name in SCHEMA_NAMES},
    }
