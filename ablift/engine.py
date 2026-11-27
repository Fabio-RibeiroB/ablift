from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np

from .models import (
    AnalysisInput,
    AnalysisResult,
    AnalysisSettings,
    BayesianArmSummary,
    ComparisonResult,
    DecisionPolicy,
    DecisionThresholds,
    GuardrailInput,
    GuardrailResult,
    Recommendation,
    SecondaryMetricInput,
    SecondaryMetricResult,
    SrmResult,
    VariantInput,
)
from .stats import (
    CupedArm,
    CupedSums,
    aggregated_arpu_tail_diagnostics,
    compute_cuped_adjustment,
    delta_ratio_z_test,
    ratio_mean_and_var,
    validate_ratio_aggregates,
)

SUPPORTED_METRICS = {"conversion_rate", "arpu", "ratio"}

# Posterior draws held in memory at once (samples x arms). 20M float64 draws is about 160 MB
# per array; the default of 50,000 samples is far below this.
MAX_POSTERIOR_DRAWS = 20_000_000


def _has_cuped(variants: list[VariantInput]) -> bool:
    return any(
        v.covariate_sum is not None
        or v.covariate_sum_squares is not None
        or v.outcome_covariate_sum is not None
        for v in variants
    )


def _compute_cuped_for_input(
    inp: AnalysisInput,
) -> tuple[dict[str, CupedArm], dict[str, Any]]:
    if inp.primary_metric not in {"conversion_rate", "arpu"}:
        raise ValueError(
            "CUPED variance reduction is supported for 'conversion_rate' and 'arpu' "
            f"primary metrics, not {inp.primary_metric!r}."
        )
    names = [v.name for v in inp.variants]
    if len(set(names)) != len(names):
        raise ValueError("CUPED requires unique variant names.")
    entries: list[CupedSums] = []
    for variant in inp.variants:
        if (
            variant.covariate_sum is None
            or variant.covariate_sum_squares is None
            or variant.outcome_covariate_sum is None
        ):
            raise ValueError(
                f"CUPED requires covariate_sum, covariate_sum_squares, and "
                f"outcome_covariate_sum for every variant (incomplete for {variant.name!r})."
            )
        if inp.primary_metric == "conversion_rate":
            y_sum = float(variant.conversions)
            y_sum_sq = float(variant.conversions)
        else:
            y_sum = float(variant.revenue_sum or 0.0)
            y_sum_sq = float(variant.revenue_sum_squares or 0.0)
        entries.append(
            CupedSums(
                name=variant.name,
                is_control=variant.is_control,
                n=variant.visitors,
                y_sum=y_sum,
                y_sum_sq=y_sum_sq,
                x_sum=variant.covariate_sum,
                x_sum_sq=variant.covariate_sum_squares,
                xy_sum=variant.outcome_covariate_sum,
            )
        )
    return compute_cuped_adjustment(entries)


def _variant_ratio_mean_and_var(
    variant: VariantInput, *, label: str = "variant"
) -> tuple[float, float]:
    assert variant.numerator_sum is not None
    assert variant.numerator_sum_squares is not None
    assert variant.denominator_sum is not None
    assert variant.denominator_sum_squares is not None
    assert variant.numerator_denominator_sum is not None
    return ratio_mean_and_var(
        variant.visitors,
        float(variant.numerator_sum),
        float(variant.numerator_sum_squares),
        float(variant.denominator_sum),
        float(variant.denominator_sum_squares),
        float(variant.numerator_denominator_sum),
        label=label,
    )


def parse_payload(payload: dict) -> AnalysisInput:
    try:
        return _parse_payload(payload)
    except KeyError as exc:
        raise ValueError(f"Input is missing required field {exc}.") from exc
    except TypeError as exc:
        raise ValueError(f"Invalid input structure: {exc}") from exc


def _parse_submetric_variants(
    raw_variants: Any,
    *,
    metric: str | None,
    primary_by_name: dict[str, VariantInput],
    label: str,
) -> list[VariantInput]:
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError(f"{label} must provide a non-empty 'variants' list.")

    parsed: list[VariantInput] = []
    for row in raw_variants:
        if not isinstance(row, dict):
            raise ValueError(f"{label} variants entries must be JSON objects.")
        row_copy: dict[str, Any] = dict(row)
        name = row_copy.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} variant entry requires a non-empty string 'name'.")
        primary = primary_by_name.get(name)
        if primary is None:
            raise ValueError(
                f"{label} contains unknown variant(s): {name}. "
                f"Expected: {', '.join(primary_by_name)}."
            )
        if row_copy.get("visitors") is None:
            row_copy["visitors"] = primary.visitors
        if "is_control" not in row_copy or row_copy["is_control"] is None:
            row_copy["is_control"] = primary.is_control
        elif bool(row_copy["is_control"]) != primary.is_control:
            raise ValueError(
                f"{label} variant {name!r} has is_control={bool(row_copy['is_control'])}, "
                f"which contradicts primary variant is_control={primary.is_control}."
            )
        if metric == "conversion_rate" and row_copy.get("conversions") is None:
            raise ValueError(
                f"{label}: conversion_rate requires 'conversions' for variant {name!r}."
            )
        parsed.append(VariantInput(**row_copy))
    return parsed


_ALLOWED_PAYLOAD_KEYS = frozenset(
    {
        "experiment_name",
        "method",
        "primary_metric",
        "alpha",
        "sequential_tau",
        "variants",
        "expected_split",
        "guardrails",
        "secondary_metrics",
        "decision_policy",
        "decision_thresholds",
        "samples",
        "random_seed",
        "input_interpretation",
    }
)


DEFAULT_SAMPLES = 50000
DEFAULT_RANDOM_SEED = 7


def _uses_random_seed(method: str, input_interpretation: dict[str, Any]) -> bool:
    """Posterior sampling (bayesian) or a row-level bootstrap CI consumes the seed."""
    return method == "bayesian" or bool(input_interpretation.get("bootstrap_ci"))


def _reject_unused_sampling_settings(
    payload: dict, method: str, input_interpretation: dict[str, Any]
) -> None:
    """Mirror the CLI: explicit settings the chosen method would ignore are errors."""
    if method == "bayesian":
        return
    if payload.get("samples") is not None:
        raise ValueError(
            f"samples only applies when method is 'bayesian' (posterior sampling); "
            f"remove it for method '{method}'."
        )
    if payload.get("random_seed") is not None and not _uses_random_seed(
        method, input_interpretation
    ):
        raise ValueError(
            f"random_seed only applies when method is 'bayesian' or a bootstrap CI is "
            f"requested; remove it for method '{method}'."
        )


def _parse_payload(payload: dict) -> AnalysisInput:
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object.")
    unknown_top = sorted(set(payload) - _ALLOWED_PAYLOAD_KEYS)
    if unknown_top:
        raise ValueError(
            f"Unknown top-level field(s): {', '.join(str(k) for k in unknown_top)}. "
            f"Expected any of: {', '.join(sorted(_ALLOWED_PAYLOAD_KEYS))}."
        )
    variants = [VariantInput(**row) for row in payload["variants"]]
    if (payload.get("primary_metric") or "conversion_rate") == "conversion_rate":
        for row in payload["variants"]:
            if isinstance(row, dict) and row.get("conversions") is None:
                raise ValueError(
                    f"conversion_rate requires 'conversions' for variant {row.get('name')!r}."
                )
    primary_by_name = {v.name: v for v in variants}

    raw_guardrails = payload.get("guardrails")
    if raw_guardrails is not None and not isinstance(raw_guardrails, list):
        raise ValueError("'guardrails' must be a list.")

    guardrails: list[GuardrailInput] = []
    for row in raw_guardrails or []:
        if not isinstance(row, dict):
            raise ValueError("Each guardrail entry must be a JSON object.")
        if row.get("variants") is not None:
            row_copy = dict(row)
            g_name = row_copy.get("name") or "<unnamed>"
            g_metric = row_copy.get("metric")
            row_copy["variants"] = _parse_submetric_variants(
                row_copy["variants"],
                metric=g_metric,
                primary_by_name=primary_by_name,
                label=f"Guardrail {g_name!r}",
            )
            guardrails.append(GuardrailInput(**row_copy))
        else:
            guardrails.append(GuardrailInput(**row))

    raw_secondary = payload.get("secondary_metrics")
    if raw_secondary is not None and not isinstance(raw_secondary, list):
        raise ValueError("'secondary_metrics' must be a list.")

    secondary_metrics: list[SecondaryMetricInput] = []
    for row in raw_secondary or []:
        if not isinstance(row, dict):
            raise ValueError("Each secondary_metrics entry must be a JSON object.")
        unknown_keys = sorted(set(row) - {"name", "metric", "variants"})
        if unknown_keys:
            raise ValueError(
                f"Secondary metric entry has unknown field(s): {', '.join(unknown_keys)}."
            )
        s_name = row["name"]
        s_metric = row["metric"]
        s_variants = _parse_submetric_variants(
            row["variants"],
            metric=s_metric,
            primary_by_name=primary_by_name,
            label=f"Secondary metric {s_name!r}",
        )
        secondary_metrics.append(
            SecondaryMetricInput(name=s_name, metric=s_metric, variants=s_variants)
        )

    decision_policy_payload = payload.get("decision_policy")
    threshold_payload = payload.get("decision_thresholds")
    method = payload.get("method") or "bayesian"
    input_interpretation = payload.get("input_interpretation") or {}
    _reject_unused_sampling_settings(payload, method, input_interpretation)

    if decision_policy_payload is not None:
        if not isinstance(decision_policy_payload, dict):
            raise ValueError("'decision_policy' must be a JSON object.")
        enabled = bool(decision_policy_payload.get("enabled", True))
        if method == "bayesian" and enabled:
            prob_raw = decision_policy_payload.get("bayes_prob_beats_control")
            loss_raw = decision_policy_payload.get("max_expected_loss")
            thresholds = DecisionThresholds(
                bayes_prob_beats_control=0.95 if prob_raw is None else prob_raw,
                max_expected_loss=0.001 if loss_raw is None else loss_raw,
            )
        else:
            thresholds = None
        decision_policy = DecisionPolicy(
            enabled=enabled,
            thresholds=thresholds,
            source="decision_policy",
        )
    elif threshold_payload is not None and method == "bayesian":
        decision_policy = DecisionPolicy(
            enabled=True,
            thresholds=DecisionThresholds(**threshold_payload),
            source="decision_thresholds",
        )
    else:
        decision_policy = DecisionPolicy(enabled=False, thresholds=None, source="none")

    samples = payload.get("samples")
    random_seed = payload.get("random_seed")
    return AnalysisInput(
        experiment_name=payload.get("experiment_name") or "experiment",
        method=method,
        variants=variants,
        primary_metric=payload.get("primary_metric", "conversion_rate"),
        alpha=payload.get("alpha", 0.05),
        sequential_tau=payload.get("sequential_tau"),
        guardrails=guardrails,
        secondary_metrics=secondary_metrics,
        decision_policy=decision_policy,
        random_seed=DEFAULT_RANDOM_SEED if random_seed is None else random_seed,
        samples=DEFAULT_SAMPLES if samples is None else samples,
        input_interpretation=input_interpretation,
        expected_split=payload.get("expected_split"),
    )


def _validate_variant_metrics(variants: list[VariantInput], metric: str, label: str) -> None:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"{label} metric must be 'conversion_rate', 'arpu', or 'ratio', got {metric!r}."
        )
    for variant in variants:
        if isinstance(variant.visitors, bool) or variant.visitors <= 0:
            raise ValueError(f"{label}: visitors must be positive for {variant.name}.")
        if metric == "conversion_rate":
            if isinstance(variant.conversions, bool) or variant.conversions < 0:
                raise ValueError(f"{label}: conversions must be non-negative for {variant.name}.")
            if variant.conversions > variant.visitors:
                raise ValueError(
                    f"{label}: conversions cannot exceed visitors for {variant.name}. "
                    "This usually means the success metric is not binary per denominator unit."
                )
        elif metric == "arpu":
            if variant.revenue_sum is None or variant.revenue_sum_squares is None:
                raise ValueError(
                    f"{label}: ARPU requires revenue_sum and revenue_sum_squares for {variant.name}."
                )
            for field_name, val in (
                ("revenue_sum", variant.revenue_sum),
                ("revenue_sum_squares", variant.revenue_sum_squares),
            ):
                if (
                    isinstance(val, bool)
                    or not isinstance(val, (int, float))
                    or not math.isfinite(val)
                ):
                    raise ValueError(
                        f"{label}: {field_name} must be a finite number for {variant.name}."
                    )
        else:
            ratio_fields = (
                ("numerator_sum", variant.numerator_sum),
                ("numerator_sum_squares", variant.numerator_sum_squares),
                ("denominator_sum", variant.denominator_sum),
                ("denominator_sum_squares", variant.denominator_sum_squares),
                ("numerator_denominator_sum", variant.numerator_denominator_sum),
            )
            missing = [fname for fname, val in ratio_fields if val is None]
            if missing:
                raise ValueError(
                    f"{label}: ratio metric requires {', '.join(missing)} for {variant.name}."
                )
            assert variant.numerator_sum is not None
            assert variant.numerator_sum_squares is not None
            assert variant.denominator_sum is not None
            assert variant.denominator_sum_squares is not None
            assert variant.numerator_denominator_sum is not None
            validate_ratio_aggregates(
                variant.visitors,
                variant.numerator_sum,
                variant.numerator_sum_squares,
                variant.denominator_sum,
                variant.denominator_sum_squares,
                variant.numerator_denominator_sum,
                label=f"{label} ({variant.name})",
            )


def _resolve_submetric_variants(
    primary_variants: list[VariantInput],
    sub_variants: list[VariantInput],
    metric: str,
    label: str,
) -> tuple[list[VariantInput], VariantInput, list[VariantInput]]:
    primary_names = [v.name for v in primary_variants]
    if len(set(primary_names)) != len(primary_names):
        raise ValueError(f"{label} cannot be matched when primary variant names are not unique.")

    sub_names = [v.name for v in sub_variants]
    if len(set(sub_names)) != len(sub_names):
        raise ValueError(f"{label} contains duplicate variant names.")

    primary_set = set(primary_names)
    sub_set = set(sub_names)
    unknown = sorted(sub_set - primary_set)
    if unknown:
        raise ValueError(
            f"{label} contains unknown variant(s): {', '.join(unknown)}. "
            f"Expected: {', '.join(primary_names)}."
        )
    missing = [name for name in primary_names if name not in sub_set]
    if missing:
        raise ValueError(f"{label} is missing variant(s): {', '.join(missing)}.")

    sub_by_name = {v.name: v for v in sub_variants}
    ordered: list[VariantInput] = []
    for primary in primary_variants:
        sub = sub_by_name[primary.name]
        ordered.append(
            VariantInput(
                name=sub.name,
                visitors=sub.visitors,
                conversions=sub.conversions,
                revenue_sum=sub.revenue_sum,
                revenue_sum_squares=sub.revenue_sum_squares,
                is_control=primary.is_control,
                numerator_sum=sub.numerator_sum,
                numerator_sum_squares=sub.numerator_sum_squares,
                denominator_sum=sub.denominator_sum,
                denominator_sum_squares=sub.denominator_sum_squares,
                numerator_denominator_sum=sub.numerator_denominator_sum,
            )
        )

    _validate_variant_metrics(ordered, metric, label)
    control = [v for v in ordered if v.is_control][0]
    treatments = [v for v in ordered if not v.is_control]
    return ordered, control, treatments


def _validate_guardrail_item(inp: AnalysisInput, item: GuardrailInput) -> None:
    if not isinstance(item.name, str) or not item.name.strip():
        raise ValueError("Guardrail 'name' must be a non-empty string.")
    label = f"Guardrail {item.name!r}"
    direction = item.direction.lower().strip()
    if direction not in {"increase", "decrease"}:
        raise ValueError(f"guardrail direction must be 'increase' or 'decrease' for {item.name}.")
    if (
        isinstance(item.max_relative_change, bool)
        or not isinstance(item.max_relative_change, (int, float))
        or not math.isfinite(item.max_relative_change)
        or item.max_relative_change < 0
    ):
        raise ValueError(f"{label}: max_relative_change must be a finite non-negative number.")

    if item.variants is not None:
        if item.control is not None or item.treatment is not None:
            raise ValueError(
                f"{label}: statistical guardrail cannot specify both 'variants' and scalar "
                "'control'/'treatment'."
            )
        if item.metric is None:
            raise ValueError(
                f"{label}: statistical guardrail with 'variants' requires 'metric' "
                "('conversion_rate', 'arpu', or 'ratio')."
            )
        _resolve_submetric_variants(inp.variants, item.variants, item.metric, label)
    else:
        if item.metric is not None:
            raise ValueError(f"{label}: 'metric' requires 'variants' for a statistical guardrail.")
        if item.control is None or item.treatment is None:
            raise ValueError(
                f"{label}: threshold guardrail requires numeric 'control' and 'treatment'."
            )
        for field_name, val in (("control", item.control), ("treatment", item.treatment)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{label}: {field_name} must be a finite number.")


def validate_input(inp: AnalysisInput) -> None:
    if not inp.variants or len(inp.variants) < 2:
        raise ValueError("At least 2 variants are required.")

    controls = [v for v in inp.variants if v.is_control]
    if len(controls) != 1:
        raise ValueError("Exactly one variant must have is_control=true.")

    for variant in inp.variants:
        if variant.visitors <= 0:
            raise ValueError(f"visitors must be positive for {variant.name}.")
        if variant.conversions < 0:
            raise ValueError(f"conversions must be non-negative for {variant.name}.")
        if variant.conversions > variant.visitors:
            raise ValueError(
                f"conversions cannot exceed visitors for {variant.name}. "
                "This usually means the success metric is not binary per denominator unit "
                "(for example total clicks per visit instead of visits with at least one click)."
            )

    resolve_expected_proportions(inp.variants, inp.expected_split)

    if inp.alpha <= 0 or inp.alpha >= 1:
        raise ValueError("alpha must be in (0, 1).")

    if inp.sequential_tau is not None and inp.sequential_tau <= 0:
        raise ValueError("sequential_tau must be positive.")
    if inp.samples <= 0:
        raise ValueError("samples must be positive.")
    if inp.method == "bayesian" and inp.samples * len(inp.variants) > MAX_POSTERIOR_DRAWS:
        limit = MAX_POSTERIOR_DRAWS // len(inp.variants)
        raise ValueError(
            f"samples={inp.samples:,} across {len(inp.variants)} variants exceeds the "
            f"{MAX_POSTERIOR_DRAWS:,} posterior-draw memory limit; use at most {limit:,}. "
            "The default of 50,000 already keeps Monte Carlo error well below 1 percentage point."
        )

    if inp.method not in {"bayesian", "sequential", "fixed"}:
        raise ValueError("method must be 'bayesian', 'sequential', or 'fixed'.")
    if inp.method != "sequential" and inp.sequential_tau is not None:
        raise ValueError("sequential_tau only applies when method is 'sequential'.")
    if inp.primary_metric not in SUPPORTED_METRICS:
        raise ValueError("primary_metric must be 'conversion_rate', 'arpu', or 'ratio'.")

    if inp.method == "bayesian" and inp.decision_policy.enabled:
        if inp.decision_policy.thresholds is None:
            raise ValueError(
                "Bayesian decision policy is enabled but thresholds were not provided."
            )
        if not (0 < inp.decision_policy.thresholds.bayes_prob_beats_control < 1):
            raise ValueError("bayes_prob_beats_control must be in (0, 1).")
        if inp.decision_policy.thresholds.max_expected_loss < 0:
            raise ValueError("max_expected_loss must be non-negative.")

    if inp.primary_metric == "arpu":
        for variant in inp.variants:
            if variant.revenue_sum is None or variant.revenue_sum_squares is None:
                raise ValueError(
                    f"ARPU requires revenue_sum and revenue_sum_squares for {variant.name}."
                )
    elif inp.primary_metric == "ratio":
        _validate_variant_metrics(inp.variants, "ratio", "Primary metric")

    for item in inp.guardrails:
        _validate_guardrail_item(inp, item)

    for sec in inp.secondary_metrics:
        if not isinstance(sec.name, str) or not sec.name.strip():
            raise ValueError("Secondary metric 'name' must be a non-empty string.")
        _resolve_submetric_variants(
            inp.variants, sec.variants, sec.metric, f"Secondary metric {sec.name!r}"
        )


def analyze(inp: AnalysisInput) -> AnalysisResult:
    validate_input(inp)

    control = [v for v in inp.variants if v.is_control][0]
    treatments = [v for v in inp.variants if not v.is_control]

    cuped_arms: dict[str, CupedArm] | None = None
    cuped_diag: dict[str, Any] | None = None
    if _has_cuped(inp.variants):
        cuped_arms, cuped_diag = _compute_cuped_for_input(inp)

    guardrail_results = evaluate_guardrails(inp)
    guardrails_passed = all(g.passed for g in guardrail_results)
    srm_result = evaluate_srm(inp.variants, expected_split=inp.expected_split)
    comparisons, bayesian_arms = _run_metric_comparisons(
        inp, control, treatments, cuped_arms=cuped_arms
    )
    secondary_results = evaluate_secondary_metrics(inp)

    recommendation = recommend(
        inp,
        comparisons,
        guardrails_passed,
        srm_result,
        guardrails=guardrail_results,
    )

    input_interpretation = dict(inp.input_interpretation)
    data_risk_flags: list[str] = []
    if not srm_result.passed:
        data_risk_flags.append("srm_detected")
    if not guardrails_passed:
        global_failed = any(g.variant is None and not g.passed for g in guardrail_results)
        blocked = {g.variant for g in guardrail_results if g.variant is not None and not g.passed}
        all_blocked = len(blocked) >= len(comparisons)
        if global_failed or all_blocked or not blocked:
            data_risk_flags.append("guardrail_failure")
        else:
            data_risk_flags.append("some_variants_guardrail_failure")

    if cuped_diag is not None:
        input_interpretation["cuped"] = cuped_diag
        if cuped_diag.get("covariate_imbalanced"):
            data_risk_flags.append("covariate_imbalance")

    if inp.primary_metric == "arpu":
        tail_diag = input_interpretation.get("arpu_tail_diagnostics")
        if not isinstance(tail_diag, dict):
            tail_diag = aggregated_arpu_tail_diagnostics(
                [
                    (
                        v.name,
                        v.visitors,
                        float(v.revenue_sum or 0.0),
                        float(v.revenue_sum_squares or 0.0),
                    )
                    for v in inp.variants
                ]
            )
            input_interpretation["arpu_tail_diagnostics"] = tail_diag
        if tail_diag.get("heavy_tailed"):
            data_risk_flags.append("heavy_tailed_revenue")

    if (
        inp.method == "fixed"
        and inp.primary_metric == "conversion_rate"
        and cuped_arms is None
        and any(_is_sparse_table(control, t) for t in treatments)
    ):
        data_risk_flags.append("sparse_conversions")

    if recommendation is not None:
        for flag in data_risk_flags:
            if flag == "some_variants_guardrail_failure" and (
                "other_variant_guardrail_failure" in recommendation.risk_flags
                or "guardrail_failure" in recommendation.risk_flags
            ):
                continue
            if flag not in recommendation.risk_flags:
                recommendation.risk_flags.append(flag)
        top_risk_flags = list(recommendation.risk_flags)
    else:
        top_risk_flags = data_risk_flags

    return AnalysisResult(
        experiment_name=inp.experiment_name,
        method=inp.method,
        analysis_settings=build_analysis_settings(inp, input_interpretation=input_interpretation),
        control_variant=control.name,
        comparisons=comparisons,
        guardrails_passed=guardrails_passed,
        guardrails=guardrail_results,
        srm=srm_result,
        recommendation=recommendation,
        bayesian_arms=bayesian_arms,
        secondary_metrics=secondary_results,
        risk_flags=top_risk_flags,
    )


def _validate_weight(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number, got {value!r}.")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{label} must be a finite positive number, got {value!r}.")
    return weight


def resolve_expected_proportions(
    variants: list[VariantInput],
    expected_split: list[float] | dict[str, float] | None = None,
) -> list[float]:
    """Resolve and normalize expected traffic allocation proportions across variants."""
    if not variants:
        raise ValueError("At least 2 variants are required.")

    has_share = [v.expected_share is not None for v in variants]
    if expected_split is not None and any(has_share):
        raise ValueError(
            "Cannot specify both top-level 'expected_split' and per-variant 'expected_share'. "
            "Use only one."
        )

    weights: list[float]
    if any(has_share):
        if not all(has_share):
            missing = [v.name for v in variants if v.expected_share is None]
            raise ValueError(
                "expected_share must be set on all variants or none; "
                f"missing for: {', '.join(missing)}."
            )
        weights = [
            _validate_weight(v.expected_share, f"expected_share for variant '{v.name}'")
            for v in variants
        ]
    elif expected_split is not None:
        if isinstance(expected_split, list):
            if len(expected_split) != len(variants):
                raise ValueError(
                    f"expected_split length ({len(expected_split)}) must match "
                    f"number of variants ({len(variants)})."
                )
            weights = [
                _validate_weight(w, f"expected_split[{i}]") for i, w in enumerate(expected_split)
            ]
        elif isinstance(expected_split, dict):
            variant_names = [v.name for v in variants]
            if len(set(variant_names)) != len(variant_names):
                raise ValueError(
                    "expected_split mapping cannot be used when variant names are not unique."
                )
            variant_set = set(variant_names)
            expected_keys = set(expected_split.keys())
            unknown = sorted(str(k) for k in expected_keys - variant_set)
            if unknown:
                raise ValueError(
                    f"expected_split contains unknown variant name(s): {', '.join(unknown)}. "
                    f"Expected: {', '.join(variant_names)}."
                )
            missing = [name for name in variant_names if name not in expected_split]
            if missing:
                raise ValueError(f"expected_split is missing variant(s): {', '.join(missing)}.")
            weights = [
                _validate_weight(expected_split[v.name], f"expected_split[{v.name!r}]")
                for v in variants
            ]
        else:
            raise ValueError(
                "expected_split must be a list of positive weights or a mapping of "
                "variant names to positive weights."
            )
    else:
        k = len(variants)
        return [1.0 / k for _ in range(k)]

    total_weight = sum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("expected_split weights sum must be finite and positive.")

    proportions = [w / total_weight for w in weights]
    if any(not math.isfinite(p) or p <= 0 for p in proportions):
        raise ValueError("Normalized expected proportions must all be finite and positive.")
    return proportions


def chi2_survival(x: float, df: int) -> float:
    """Exact upper-tail probability of a chi-square distribution with integer ``df``."""
    if x <= 0:
        return 1.0
    half = x / 2.0
    if df % 2 == 0:
        terms = (i * math.log(half) - math.lgamma(i + 1) for i in range(df // 2))
        tail = 0.0
    else:
        terms = ((i - 0.5) * math.log(half) - math.lgamma(i + 0.5) for i in range(1, (df + 1) // 2))
        tail = math.erfc(math.sqrt(half))
    return min(1.0, tail + sum(math.exp(t - half) for t in terms))


def evaluate_srm(
    variants: list[VariantInput],
    expected_split: list[float] | dict[str, float] | None = None,
) -> SrmResult:
    proportions = resolve_expected_proportions(variants, expected_split)
    observed = [v.visitors for v in variants]
    total = float(sum(observed))
    k = len(observed)
    expected = [total * p for p in proportions]

    chi2 = 0.0
    for obs, exp in zip(observed, expected, strict=True):
        if exp > 0:
            chi2 += ((obs - exp) ** 2) / exp

    p_value = chi2_survival(chi2, max(k - 1, 1))
    passed = p_value >= 0.001
    reason = "No strong SRM evidence." if passed else "Potential sample ratio mismatch detected."

    return SrmResult(
        passed=passed,
        p_value=float(max(min(p_value, 1.0), 0.0)),
        observed=observed,
        expected=expected,
        expected_proportions=proportions,
        reason=reason,
    )


def _run_metric_comparisons(
    inp: AnalysisInput,
    control: VariantInput,
    treatments: list[VariantInput],
    *,
    cuped_arms: dict[str, CupedArm] | None = None,
) -> tuple[list[ComparisonResult], list[BayesianArmSummary] | None]:
    if inp.method == "bayesian":
        comparisons, bayesian_arms = _analyze_bayesian_joint(
            inp, control, treatments, cuped_arms=cuped_arms
        )
        return comparisons, bayesian_arms
    if cuped_arms is not None:
        if inp.method == "fixed":
            return analyze_fixed_cuped(inp, control, treatments, cuped_arms), None
        return analyze_sequential_cuped(inp, control, treatments, cuped_arms), None
    if inp.method == "fixed":
        if inp.primary_metric == "conversion_rate":
            return analyze_fixed_conversion(inp, control, treatments), None
        if inp.primary_metric == "ratio":
            return analyze_fixed_ratio(inp, control, treatments), None
        return analyze_fixed_arpu(inp, control, treatments), None
    if inp.primary_metric == "conversion_rate":
        return analyze_sequential_conversion(inp, control, treatments), None
    if inp.primary_metric == "ratio":
        return analyze_sequential_ratio(inp, control, treatments), None
    return analyze_sequential_arpu(inp, control, treatments), None


def _submetric_analysis_input(
    inp: AnalysisInput,
    *,
    name: str,
    metric: str,
    variants: list[VariantInput],
) -> AnalysisInput:
    return AnalysisInput(
        experiment_name=f"{inp.experiment_name}:{name}",
        method=inp.method,
        variants=variants,
        primary_metric=metric,
        alpha=inp.alpha,
        # Sub-metrics use their own scale; do not leak primary metric's sequential_tau.
        sequential_tau=None,
        decision_policy=inp.decision_policy,
        random_seed=inp.random_seed,
        samples=inp.samples,
    )


def _evaluate_statistical_guardrail(
    inp: AnalysisInput, item: GuardrailInput, direction: str
) -> list[GuardrailResult]:
    assert item.variants is not None
    assert item.metric is not None
    ordered, control, treatments = _resolve_submetric_variants(
        inp.variants, item.variants, item.metric, f"Guardrail {item.name!r}"
    )
    sub_inp = _submetric_analysis_input(inp, name=item.name, metric=item.metric, variants=ordered)

    results: list[GuardrailResult] = []
    if inp.method == "bayesian":
        control_rate, treatment_rates, control_samples, treatment_samples_list = (
            _draw_bayesian_samples(sub_inp, control, treatments)
        )
        comparisons, _ = _summarize_bayesian_draws(
            variants=ordered,
            control=control,
            treatments=treatments,
            metric=item.metric,
            control_rate=control_rate,
            treatment_rates=treatment_rates,
            control_samples=control_samples,
            treatment_samples_list=treatment_samples_list,
        )
        prob_threshold = (
            inp.decision_policy.thresholds.bayes_prob_beats_control
            if (inp.decision_policy.enabled and inp.decision_policy.thresholds is not None)
            else (1.0 - inp.alpha)
        )
        margin = (
            item.max_relative_change * abs(control_rate)
            if control_rate != 0
            else item.max_relative_change
        )
        for comp, t_samples in zip(comparisons, treatment_samples_list, strict=True):
            diff_samples = t_samples - control_samples
            if direction == "decrease":
                prob_regressed = float(np.mean(diff_samples > margin))
            else:
                prob_regressed = float(np.mean(diff_samples < -margin))
            passed = prob_regressed < prob_threshold
            rel_change = (
                (comp.treatment_rate - comp.control_rate) / abs(comp.control_rate)
                if comp.control_rate != 0
                else (comp.treatment_rate - comp.control_rate)
            )
            reason = (
                f"No statistically significant regression "
                f"(P(regressed > {item.max_relative_change * 100:.2f}%)={prob_regressed:.3f} "
                f"< {prob_threshold:.3f})"
                if passed
                else f"Statistically significant regression detected "
                f"(P(regressed > {item.max_relative_change * 100:.2f}%)={prob_regressed:.3f} "
                f">= {prob_threshold:.3f})"
            )
            results.append(
                GuardrailResult(
                    name=item.name,
                    direction=direction,
                    control=comp.control_rate,
                    treatment=comp.treatment_rate,
                    relative_change=float(rel_change),
                    max_relative_change=item.max_relative_change,
                    passed=passed,
                    reason=reason,
                    variant=comp.treatment,
                    mode="statistical",
                    comparison=comp,
                )
            )
        return results

    comparisons, _ = _run_metric_comparisons(sub_inp, control, treatments)
    for comp in comparisons:
        margin = (
            item.max_relative_change * abs(comp.control_rate)
            if comp.control_rate != 0
            else item.max_relative_change
        )
        if item.max_relative_change == 0.0:
            if direction == "decrease":
                regressed = bool(comp.significant and comp.absolute_lift > 0)
            else:
                regressed = bool(comp.significant and comp.absolute_lift < 0)
        elif direction == "decrease":
            regressed = bool(comp.ci_low is not None and comp.ci_low > margin)
        else:
            regressed = bool(comp.ci_high is not None and comp.ci_high < -margin)

        passed = not regressed
        rel_change = (
            (comp.treatment_rate - comp.control_rate) / abs(comp.control_rate)
            if comp.control_rate != 0
            else (comp.treatment_rate - comp.control_rate)
        )
        p_display = f"{comp.p_value:.6g}" if comp.p_value is not None else "n/a"
        reason = (
            f"No statistically significant regression beyond "
            f"{item.max_relative_change * 100:.2f}% (p={p_display})"
            if passed
            else f"Statistically significant regression beyond "
            f"{item.max_relative_change * 100:.2f}% (p={p_display})"
        )
        results.append(
            GuardrailResult(
                name=item.name,
                direction=direction,
                control=comp.control_rate,
                treatment=comp.treatment_rate,
                relative_change=float(rel_change),
                max_relative_change=item.max_relative_change,
                passed=passed,
                reason=reason,
                variant=comp.treatment,
                mode="statistical",
                comparison=comp,
            )
        )
    return results


def evaluate_guardrails(inp: AnalysisInput) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    for item in inp.guardrails:
        direction = item.direction.lower().strip()
        if direction not in {"increase", "decrease"}:
            raise ValueError(
                f"guardrail direction must be 'increase' or 'decrease' for {item.name}."
            )

        if item.variants is not None:
            results.extend(_evaluate_statistical_guardrail(inp, item, direction))
            continue

        assert item.control is not None and item.treatment is not None
        if item.control == 0:
            relative_change = item.treatment - item.control
        else:
            relative_change = (item.treatment - item.control) / abs(item.control)

        if direction == "decrease":
            passed = relative_change <= item.max_relative_change
            reason = (
                "Within allowed increase" if passed else "Guardrail worsened above allowed increase"
            )
        else:
            passed = relative_change >= -item.max_relative_change
            reason = (
                "Within allowed decrease" if passed else "Guardrail dropped below allowed decrease"
            )

        results.append(
            GuardrailResult(
                name=item.name,
                direction=direction,
                control=item.control,
                treatment=item.treatment,
                relative_change=float(relative_change),
                max_relative_change=item.max_relative_change,
                passed=passed,
                reason=reason,
            )
        )

    return results


def evaluate_secondary_metrics(inp: AnalysisInput) -> list[SecondaryMetricResult]:
    results: list[SecondaryMetricResult] = []
    for sec in inp.secondary_metrics:
        ordered, control, treatments = _resolve_submetric_variants(
            inp.variants, sec.variants, sec.metric, f"Secondary metric {sec.name!r}"
        )
        sub_inp = _submetric_analysis_input(inp, name=sec.name, metric=sec.metric, variants=ordered)
        comparisons, bayesian_arms = _run_metric_comparisons(sub_inp, control, treatments)
        results.append(
            SecondaryMetricResult(
                name=sec.name,
                metric=sec.metric,
                comparisons=comparisons,
                bayesian_arms=bayesian_arms,
            )
        )
    return results


def _summarize_bayesian_draws(
    *,
    variants: list[VariantInput],
    control: VariantInput,
    treatments: list[VariantInput],
    metric: str,
    control_rate: float,
    treatment_rates: list[float],
    control_samples: np.ndarray,
    treatment_samples_list: list[np.ndarray],
) -> tuple[list[ComparisonResult], list[BayesianArmSummary]]:
    """Compute pairwise and joint multi-arm Bayesian summaries from posterior draws.

    Row 0 of ``all_samples`` is ``control``; rows ``1..m`` correspond to ``treatments``.
    """
    all_samples = np.vstack([control_samples, *treatment_samples_list])
    n_arms, n_samples = all_samples.shape

    best_indices = np.argmax(all_samples, axis=0)
    prob_best = np.bincount(best_indices, minlength=n_arms) / float(n_samples)
    best_vals = all_samples[best_indices, np.arange(n_samples)]
    multi_arm_losses = np.mean(best_vals - all_samples, axis=1)

    # Preserve input variant order in bayesian_arms while matching the joint matrix rows.
    arm_index_by_id = {id(control): 0}
    for idx, treatment in enumerate(treatments, start=1):
        arm_index_by_id[id(treatment)] = idx
    ordered_variants = (
        variants if all(id(v) in arm_index_by_id for v in variants) else [control, *treatments]
    )
    bayesian_arms = [
        BayesianArmSummary(
            variant=v.name,
            is_control=v.is_control,
            probability_best=float(prob_best[arm_index_by_id[id(v)]]),
            multi_arm_expected_loss=float(multi_arm_losses[arm_index_by_id[id(v)]]),
        )
        for v in ordered_variants
    ]

    clipped_control = np.clip(control_samples, 1e-12, None)
    denom = max(control_rate, 1e-12)
    comparisons: list[ComparisonResult] = []
    for idx, (treatment, t_rate, t_samples) in enumerate(
        zip(treatments, treatment_rates, treatment_samples_list, strict=True),
        start=1,
    ):
        diff_samples = t_samples - control_samples
        rel_samples = np.divide(diff_samples, clipped_control)

        p_win = float(np.mean(t_samples > control_samples))
        expected_loss = float(np.mean(np.maximum(control_samples - t_samples, 0.0)))
        ci_low, ci_high = np.quantile(diff_samples, [0.025, 0.975])
        rel_ci_low, rel_ci_high = np.quantile(rel_samples, [0.025, 0.975])

        comparisons.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric=metric,
                control_rate=control_rate,
                treatment_rate=t_rate,
                absolute_lift=t_rate - control_rate,
                relative_lift=(t_rate - control_rate) / denom,
                probability_beats_control=p_win,
                expected_loss=expected_loss,
                probability_best=float(prob_best[idx]),
                multi_arm_expected_loss=float(multi_arm_losses[idx]),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                relative_ci_low=float(rel_ci_low),
                relative_ci_high=float(rel_ci_high),
            )
        )

    return comparisons, bayesian_arms


def _draw_bayesian_samples(
    inp: AnalysisInput,
    control: VariantInput,
    treatments: list[VariantInput],
    *,
    cuped_arms: dict[str, CupedArm] | None = None,
) -> tuple[float, list[float], np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(inp.random_seed)

    if cuped_arms is not None:
        c = cuped_arms[control.name]
        control_rate = c.mean
        control_samples = sample_mean_posterior_from_moments(c.n, c.mean, c.ss, inp.samples, rng)
        treatment_rates = []
        treatment_samples_list = []
        for t in treatments:
            arm = cuped_arms[t.name]
            treatment_rates.append(arm.mean)
            treatment_samples_list.append(
                sample_mean_posterior_from_moments(arm.n, arm.mean, arm.ss, inp.samples, rng)
            )
    elif inp.primary_metric == "conversion_rate":
        alpha_prior = 1.0
        beta_prior = 1.0
        control_samples = rng.beta(
            alpha_prior + control.conversions,
            beta_prior + (control.visitors - control.conversions),
            size=inp.samples,
        )
        treatment_samples_list = [
            rng.beta(
                alpha_prior + t.conversions,
                beta_prior + (t.visitors - t.conversions),
                size=inp.samples,
            )
            for t in treatments
        ]
        control_rate = control.conversions / control.visitors
        treatment_rates = [t.conversions / t.visitors for t in treatments]
    elif inp.primary_metric == "ratio":
        control_rate, c_var = _variant_ratio_mean_and_var(
            control, label=f"Variant {control.name!r}"
        )
        control_samples = (
            rng.normal(control_rate, math.sqrt(c_var), size=inp.samples)
            if c_var > 0.0
            else np.full(inp.samples, control_rate, dtype=float)
        )
        treatment_rates = []
        treatment_samples_list = []
        for t in treatments:
            t_rate, t_var = _variant_ratio_mean_and_var(t, label=f"Variant {t.name!r}")
            treatment_rates.append(t_rate)
            treatment_samples_list.append(
                rng.normal(t_rate, math.sqrt(t_var), size=inp.samples)
                if t_var > 0.0
                else np.full(inp.samples, t_rate, dtype=float)
            )
    else:
        control_samples = sample_mean_posterior(
            control.visitors,
            float(control.revenue_sum or 0.0),
            float(control.revenue_sum_squares or 0.0),
            inp.samples,
            rng,
        )
        treatment_samples_list = [
            sample_mean_posterior(
                t.visitors,
                float(t.revenue_sum or 0.0),
                float(t.revenue_sum_squares or 0.0),
                inp.samples,
                rng,
            )
            for t in treatments
        ]
        control_rate = float(control.revenue_sum or 0.0) / control.visitors
        treatment_rates = [float(t.revenue_sum or 0.0) / t.visitors for t in treatments]

    return control_rate, treatment_rates, control_samples, treatment_samples_list


def _analyze_bayesian_joint(
    inp: AnalysisInput,
    control: VariantInput,
    treatments: list[VariantInput],
    *,
    cuped_arms: dict[str, CupedArm] | None = None,
) -> tuple[list[ComparisonResult], list[BayesianArmSummary]]:
    control_rate, treatment_rates, control_samples, treatment_samples_list = _draw_bayesian_samples(
        inp, control, treatments, cuped_arms=cuped_arms
    )
    return _summarize_bayesian_draws(
        variants=inp.variants,
        control=control,
        treatments=treatments,
        metric=inp.primary_metric,
        control_rate=control_rate,
        treatment_rates=treatment_rates,
        control_samples=control_samples,
        treatment_samples_list=treatment_samples_list,
    )


def build_analysis_settings(
    inp: AnalysisInput, *, input_interpretation: dict[str, Any] | None = None
) -> AnalysisSettings:
    if inp.method == "bayesian" and _has_cuped(inp.variants):
        priors: dict[str, Any] = {
            "family": "cuped_normal_inverse_gamma",
            "mu0": 0.0,
            "kappa0": 1e-6,
            "alpha0": 1.0,
            "beta0": 1.0,
        }
    elif inp.method == "bayesian" and inp.primary_metric == "conversion_rate":
        priors = {
            "family": "beta_binomial",
            "alpha_prior": 1.0,
            "beta_prior": 1.0,
        }
    elif inp.method == "bayesian" and inp.primary_metric == "arpu":
        priors = {
            "family": "normal_inverse_gamma",
            "mu0": 0.0,
            "kappa0": 1e-6,
            "alpha0": 1.0,
            "beta0": 1.0,
        }
    elif inp.method == "bayesian" and inp.primary_metric == "ratio":
        priors = {"family": "delta_normal_asymptotic"}
    else:
        priors = {"family": "not_applicable"}

    if inp.decision_policy.thresholds is None:
        policy = {
            "enabled": inp.decision_policy.enabled,
            "source": inp.decision_policy.source,
            "thresholds": None,
        }
    else:
        policy = {
            "enabled": inp.decision_policy.enabled,
            "source": inp.decision_policy.source,
            "thresholds": {
                "bayes_prob_beats_control": inp.decision_policy.thresholds.bayes_prob_beats_control,
                "max_expected_loss": inp.decision_policy.thresholds.max_expected_loss,
            },
        }

    n_comparisons = sum(1 for v in inp.variants if not v.is_control)
    if inp.method in {"sequential", "fixed"}:
        multiple_comparisons = {
            "correction": "bonferroni",
            "n_comparisons": n_comparisons,
            "alpha_per_comparison": per_comparison_alpha(inp.alpha, n_comparisons),
        }
    else:
        multiple_comparisons = {"correction": "none", "n_comparisons": n_comparisons}

    interpretation = (
        input_interpretation if input_interpretation is not None else inp.input_interpretation
    )
    return AnalysisSettings(
        primary_metric=inp.primary_metric,
        alpha=inp.alpha,
        multiple_comparisons=multiple_comparisons,
        sequential_tau=inp.sequential_tau,
        samples=inp.samples if inp.method == "bayesian" else None,
        random_seed=(
            inp.random_seed if _uses_random_seed(inp.method, interpretation or {}) else None
        ),
        priors=priors,
        decision_policy=policy,
        input_interpretation=interpretation,
    )


def _default_tau(baseline: float) -> float:
    """Default mSPRT mixing parameter: 10% relative to baseline."""
    return max(abs(baseline) * 0.1, 1e-6)


def msprt_statistic(delta_hat: float, se: float, tau: float) -> float:
    """mSPRT likelihood ratio statistic (Johari et al. 2017).

    Reject H0 when this exceeds 1/alpha.
    tau is the prior standard deviation on the effect size (mixing parameter).
    """
    if se <= 0 or tau <= 0:
        return 1.0
    v = se * se
    t2 = tau * tau
    exponent = (delta_hat * delta_hat * t2) / (2.0 * v * (v + t2))
    return math.sqrt(v / (v + t2)) * math.exp(min(exponent, 700.0))


def always_valid_ci_margin(se: float, tau: float, alpha: float) -> float:
    """Half-width of always-valid confidence interval (Johari et al. 2017)."""
    if se <= 0 or tau <= 0:
        return float("inf")
    v = se * se
    t2 = tau * tau
    rho2 = t2 / v
    inner = (1.0 / alpha) * math.sqrt(1.0 + rho2)
    if inner <= 1.0:
        return float("inf")
    return se * math.sqrt((1.0 + rho2) / rho2 * 2.0 * math.log(inner))


def per_comparison_alpha(alpha: float, n_comparisons: int) -> float:
    """Bonferroni-adjusted alpha so the family-wise error rate stays at ``alpha``."""
    return alpha / max(n_comparisons, 1)


def _sequential_comparison(
    *,
    inp: AnalysisInput,
    control: VariantInput,
    treatment: VariantInput,
    metric: str,
    control_value: float,
    treatment_value: float,
    se: float,
    alpha_adj: float,
) -> ComparisonResult:
    diff = treatment_value - control_value
    tau = inp.sequential_tau if inp.sequential_tau is not None else _default_tau(control_value)
    e_val = msprt_statistic(diff, se, tau)
    margin = always_valid_ci_margin(se, tau, alpha_adj)
    # Bonferroni-adjusted always-valid p-value: p <= alpha exactly when significant.
    n_comparisons = inp.alpha / alpha_adj
    p_value = min(1.0, n_comparisons / e_val) if e_val > 0 else 1.0
    return ComparisonResult(
        treatment=treatment.name,
        control=control.name,
        metric=metric,
        control_rate=control_value,
        treatment_rate=treatment_value,
        absolute_lift=diff,
        relative_lift=diff / max(abs(control_value), 1e-12),
        p_value=float(p_value),
        e_value=float(e_val),
        e_threshold=float(1.0 / alpha_adj),
        significant=bool(e_val >= 1.0 / alpha_adj),
        # An infinite margin means the variance could not be estimated; report no interval
        # rather than +/-inf, which is not valid JSON.
        ci_low=float(diff - margin) if math.isfinite(margin) else None,
        ci_high=float(diff + margin) if math.isfinite(margin) else None,
    )


def analyze_sequential_conversion(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    alpha_adj = per_comparison_alpha(inp.alpha, len(treatments))
    control_rate = control.conversions / control.visitors
    out: list[ComparisonResult] = []
    for treatment in treatments:
        se = adjusted_difference_se(
            control.conversions,
            control.visitors,
            treatment.conversions,
            treatment.visitors,
        )
        out.append(
            _sequential_comparison(
                inp=inp,
                control=control,
                treatment=treatment,
                metric="conversion_rate",
                control_value=control_rate,
                treatment_value=treatment.conversions / treatment.visitors,
                se=se,
                alpha_adj=alpha_adj,
            )
        )
    return out


def analyze_sequential_arpu(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    alpha_adj = per_comparison_alpha(inp.alpha, len(treatments))
    c_mean, c_var = mean_and_var_from_aggregates(
        control.visitors,
        float(control.revenue_sum or 0.0),
        float(control.revenue_sum_squares or 0.0),
    )
    out: list[ComparisonResult] = []
    for treatment in treatments:
        t_mean, t_var = mean_and_var_from_aggregates(
            treatment.visitors,
            float(treatment.revenue_sum or 0.0),
            float(treatment.revenue_sum_squares or 0.0),
        )
        # With fewer than 2 users in an arm the variance cannot be estimated; se=0 makes the
        # mSPRT report "no evidence" rather than certainty.
        if control.visitors < 2 or treatment.visitors < 2:
            se = 0.0
        else:
            se = math.sqrt(max((c_var / control.visitors) + (t_var / treatment.visitors), 1e-18))
        out.append(
            _sequential_comparison(
                inp=inp,
                control=control,
                treatment=treatment,
                metric="arpu",
                control_value=c_mean,
                treatment_value=t_mean,
                se=se,
                alpha_adj=alpha_adj,
            )
        )
    return out


def analyze_sequential_ratio(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    """Always-valid mSPRT for ratio metrics using the bivariate Delta-method SE."""
    alpha_adj = per_comparison_alpha(inp.alpha, len(treatments))
    c_ratio, c_var = _variant_ratio_mean_and_var(control, label=f"Variant {control.name!r}")
    out: list[ComparisonResult] = []
    for treatment in treatments:
        t_ratio, t_var = _variant_ratio_mean_and_var(treatment, label=f"Variant {treatment.name!r}")
        se = math.sqrt(max(c_var + t_var, 1e-18))
        out.append(
            _sequential_comparison(
                inp=inp,
                control=control,
                treatment=treatment,
                metric="ratio",
                control_value=c_ratio,
                treatment_value=t_ratio,
                se=se,
                alpha_adj=alpha_adj,
            )
        )
    return out


def analyze_sequential_cuped(
    inp: AnalysisInput,
    control: VariantInput,
    treatments: list[VariantInput],
    cuped_arms: dict[str, CupedArm],
) -> list[ComparisonResult]:
    """Always-valid mSPRT using CUPED-adjusted means and variances."""
    alpha_adj = per_comparison_alpha(inp.alpha, len(treatments))
    c = cuped_arms[control.name]
    out: list[ComparisonResult] = []
    for treatment in treatments:
        t = cuped_arms[treatment.name]
        se = math.sqrt(max(c.var / c.n + t.var / t.n, 1e-18))
        out.append(
            _sequential_comparison(
                inp=inp,
                control=control,
                treatment=treatment,
                metric=inp.primary_metric,
                control_value=c.mean,
                treatment_value=t.mean,
                se=se,
                alpha_adj=alpha_adj,
            )
        )
    return out


def adjusted_difference_se(x1: int, n1: int, x2: int, n2: int) -> float:
    """Agresti-Caffo standard error for a difference in proportions.

    Adds one success and one failure to each arm before estimating the variance, so arms
    with 0% or 100% conversion (common at tiny sample sizes) never yield a zero standard
    error. For realistic sample sizes it is indistinguishable from the unpooled SE.
    """
    p1 = (x1 + 1) / (n1 + 2)
    p2 = (x2 + 1) / (n2 + 2)
    return math.sqrt(p1 * (1 - p1) / (n1 + 2) + p2 * (1 - p2) / (n2 + 2))


def two_proportion_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float]:
    p1 = x1 / n1
    p2 = x2 / n2

    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / n1 + 1 / n2), 1e-18))

    z = (p2 - p1) / se
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))

    unpooled_se = math.sqrt(max((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2), 1e-18))

    return p_value, z, unpooled_se


def mean_and_var_from_aggregates(
    n: int, value_sum: float, value_sum_squares: float
) -> tuple[float, float]:
    mean = value_sum / max(n, 1)
    if n <= 1:
        return mean, 0.0
    centered = value_sum_squares - (value_sum * value_sum) / n
    variance = max(centered / (n - 1), 0.0)
    return mean, variance


def _betacf(x: float, a: float, b: float) -> float:
    """Continued fraction for the regularized incomplete beta function (Lentz's method)."""
    max_iter = 200
    eps = 3.0e-15
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def student_t_two_sided_p(t_stat: float, df: float) -> float:
    """Two-sided p-value for Student's t distribution with ``df`` degrees of freedom."""
    if df <= 0:
        return 1.0
    if not math.isfinite(t_stat):
        return 0.0
    if df > 1e7:
        return 2.0 * (1.0 - NormalDist().cdf(abs(t_stat)))
    t2 = t_stat * t_stat
    if t2 == 0.0:
        return 1.0
    x = df / (df + t2)
    y = t2 / (df + t2)
    a = df / 2.0
    b = 0.5
    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log1p(-y) + b * math.log(y)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        p = bt * _betacf(x, a, b) / a
    else:
        p = 1.0 - bt * _betacf(y, b, a) / b
    return min(1.0, max(0.0, p))


def student_t_ppf(q: float, df: float) -> float:
    """Quantile (inverse CDF) of Student's t distribution at probability ``q``."""
    if not (0.0 < q < 1.0):
        raise ValueError(f"Probability q must be in (0, 1), got {q!r}.")
    if q == 0.5:
        return 0.0
    if q < 0.5:
        return -student_t_ppf(1.0 - q, df)
    if df > 1e7:
        return NormalDist().inv_cdf(q)
    if df == 1.0:
        return math.tan(math.pi * (q - 0.5))
    if df == 2.0:
        u = 2.0 * q - 1.0
        return math.sqrt(2.0 * u * u / (1.0 - u * u))

    target = 2.0 * (1.0 - q)
    z = NormalDist().inv_cdf(q)
    lo = max(0.0, z * 0.9)
    while student_t_two_sided_p(lo, df) < target:
        lo *= 0.5
    hi = max(z * 1.1, 1.0)
    while student_t_two_sided_p(hi, df) > target:
        hi *= 2.0

    log_norm = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    t_val = min(max(z, lo), hi)
    for _ in range(60):
        p = student_t_two_sided_p(t_val, df)
        err = p - target
        if err > 0.0:
            lo = t_val
        else:
            hi = t_val
        if abs(err) <= 1e-15 * max(target, 1e-15) or (hi - lo) <= 1e-14 * max(1.0, t_val):
            break
        log_pdf = log_norm - ((df + 1.0) / 2.0) * math.log1p((t_val * t_val) / df)
        deriv = -2.0 * math.exp(log_pdf)
        cand = t_val - err / deriv if deriv != 0.0 else 0.5 * (lo + hi)
        t_val = cand if lo < cand < hi else 0.5 * (lo + hi)
    return t_val


def welch_t_test_from_moments(
    n1: int,
    mean1: float,
    var1: float,
    n2: int,
    mean2: float,
    var2: float,
) -> tuple[float, float | None, float | None, float]:
    """Welch's two-sample t-test from sample sizes, means, and sample variances.

    Returns ``(raw_p_value, t_statistic, welch_satterthwaite_df, standard_error)``.
    """
    diff = mean2 - mean1

    if n1 < 2 or n2 < 2:
        return 1.0, None, None, 0.0

    v1 = var1 / n1
    v2 = var2 / n2
    v = v1 + v2
    if v <= 0.0:
        df = float(n1 + n2 - 2)
        if diff == 0.0:
            return 1.0, 0.0, df, 0.0
        return 0.0, None, df, 0.0

    se = math.sqrt(v)
    t_stat = diff / se
    df_denom = (v1 * v1) / (n1 - 1) + (v2 * v2) / (n2 - 1)
    df = (v * v) / df_denom
    raw_p = student_t_two_sided_p(t_stat, df)
    return raw_p, t_stat, df, se


def welch_t_test(
    n1: int,
    sum1: float,
    sum_sq1: float,
    n2: int,
    sum2: float,
    sum_sq2: float,
) -> tuple[float, float | None, float | None, float]:
    """Welch's two-sample t-test from aggregate sums and sums of squares.

    Returns ``(raw_p_value, t_statistic, welch_satterthwaite_df, standard_error)``.
    """
    mean1, var1 = mean_and_var_from_aggregates(n1, sum1, sum_sq1)
    mean2, var2 = mean_and_var_from_aggregates(n2, sum2, sum_sq2)
    return welch_t_test_from_moments(n1, mean1, var1, n2, mean2, var2)


def _wilson_interval(x: int, n: int, z: float) -> tuple[float, float]:
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return center - half, center + half


def newcombe_diff_ci(x1: int, n1: int, x2: int, n2: int, z: float) -> tuple[float, float]:
    """Newcombe (1998) method 10 hybrid score interval for ``p2 - p1``.

    Unlike the Wald interval it stays non-degenerate at 0% or 100% and keeps near-nominal
    coverage for rare events; at large counts the two agree.
    """
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = _wilson_interval(x1, n1, z)
    l2, u2 = _wilson_interval(x2, n2, z)
    diff = p2 - p1
    return (
        diff - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2),
        diff + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2),
    )


def fisher_exact_p_value(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for a 2x2 table, summing tables no likelier than observed."""
    successes = x1 + x2
    total = n1 + n2

    def log_prob(k: int) -> float:  # P(arm 2 has k successes | margins)
        return _log_comb(n2, k) + _log_comb(n1, successes - k) - _log_comb(total, successes)

    k_min, k_max = max(0, successes - n1), min(successes, n2)
    observed = log_prob(x2)
    p = sum(math.exp(lp) for k in range(k_min, k_max + 1) if (lp := log_prob(k)) <= observed + 1e-7)
    return min(1.0, p)


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _is_sparse_table(control: VariantInput, treatment: VariantInput) -> bool:
    """True when any expected 2x2 cell under the pooled null is below 5."""
    n1, n2 = control.visitors, treatment.visitors
    pooled = (control.conversions + treatment.conversions) / (n1 + n2)
    return min(n1, n2) * min(pooled, 1.0 - pooled) < 5.0


def analyze_fixed_conversion(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    """Fixed-horizon two-proportion test with Newcombe Bonferroni-adjusted CIs.

    Uses the z-test, or Fisher's exact test when any expected cell count is below 5.
    """
    n_comparisons = len(treatments)
    alpha_adj = per_comparison_alpha(inp.alpha, n_comparisons)
    z_crit = NormalDist().inv_cdf(1.0 - alpha_adj / 2.0)
    control_rate = control.conversions / control.visitors

    out: list[ComparisonResult] = []
    for treatment in treatments:
        treatment_rate = treatment.conversions / treatment.visitors
        diff = treatment_rate - control_rate
        counts = (control.conversions, control.visitors, treatment.conversions, treatment.visitors)
        z_stat: float | None
        if _is_sparse_table(control, treatment):
            raw_p = fisher_exact_p_value(*counts)
            z_stat = None
        else:
            raw_p, z_stat, _ = two_proportion_test(*counts)
        ci_low, ci_high = newcombe_diff_ci(*counts, z_crit)
        adj_p = min(1.0, raw_p * n_comparisons)

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="conversion_rate",
                control_rate=control_rate,
                treatment_rate=treatment_rate,
                absolute_lift=diff,
                relative_lift=diff / max(control_rate, 1e-12),
                p_value=float(adj_p),
                raw_p_value=float(raw_p),
                test_statistic=float(z_stat) if z_stat is not None else None,
                significant=bool(adj_p <= inp.alpha),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
            )
        )
    return out


def _fixed_welch_comparison(
    *,
    inp: AnalysisInput,
    control_name: str,
    treatment_name: str,
    metric: str,
    n1: int,
    mean1: float,
    var1: float,
    n2: int,
    mean2: float,
    var2: float,
    n_comparisons: int,
    alpha_adj: float,
) -> ComparisonResult:
    diff = mean2 - mean1
    raw_p, t_stat, df, se = welch_t_test_from_moments(n1, mean1, var1, n2, mean2, var2)
    adj_p = min(1.0, raw_p * n_comparisons)
    significant = bool(adj_p <= inp.alpha)

    ci_low: float | None
    ci_high: float | None
    if df is None:
        ci_low, ci_high = None, None
    elif se <= 0.0:
        ci_low, ci_high = float(diff), float(diff)
    else:
        t_crit = student_t_ppf(1.0 - alpha_adj / 2.0, df)
        ci_low = float(diff - t_crit * se)
        ci_high = float(diff + t_crit * se)

    return ComparisonResult(
        treatment=treatment_name,
        control=control_name,
        metric=metric,
        control_rate=mean1,
        treatment_rate=mean2,
        absolute_lift=diff,
        relative_lift=diff / max(abs(mean1), 1e-12),
        p_value=float(adj_p),
        raw_p_value=float(raw_p),
        test_statistic=float(t_stat) if t_stat is not None else None,
        degrees_of_freedom=float(df) if df is not None else None,
        significant=significant,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def analyze_fixed_arpu(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    """Fixed-horizon Welch's two-sample t-test with Bonferroni-adjusted CIs."""
    n_comparisons = len(treatments)
    alpha_adj = per_comparison_alpha(inp.alpha, n_comparisons)
    c_mean, c_var = mean_and_var_from_aggregates(
        control.visitors,
        float(control.revenue_sum or 0.0),
        float(control.revenue_sum_squares or 0.0),
    )

    out: list[ComparisonResult] = []
    for treatment in treatments:
        t_mean, t_var = mean_and_var_from_aggregates(
            treatment.visitors,
            float(treatment.revenue_sum or 0.0),
            float(treatment.revenue_sum_squares or 0.0),
        )
        out.append(
            _fixed_welch_comparison(
                inp=inp,
                control_name=control.name,
                treatment_name=treatment.name,
                metric="arpu",
                n1=control.visitors,
                mean1=c_mean,
                var1=c_var,
                n2=treatment.visitors,
                mean2=t_mean,
                var2=t_var,
                n_comparisons=n_comparisons,
                alpha_adj=alpha_adj,
            )
        )
    return out


def analyze_fixed_ratio(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    """Fixed-horizon Delta-method two-sided z-test with Bonferroni-adjusted CIs."""
    n_comparisons = len(treatments)
    alpha_adj = per_comparison_alpha(inp.alpha, n_comparisons)
    z_crit = NormalDist().inv_cdf(1.0 - alpha_adj / 2.0)
    c_ratio, c_var = _variant_ratio_mean_and_var(control, label=f"Variant {control.name!r}")

    out: list[ComparisonResult] = []
    for treatment in treatments:
        t_ratio, t_var = _variant_ratio_mean_and_var(treatment, label=f"Variant {treatment.name!r}")
        diff = t_ratio - c_ratio
        raw_p, z_stat, se = delta_ratio_z_test(c_ratio, c_var, t_ratio, t_var)
        adj_p = min(1.0, raw_p * n_comparisons)
        significant = bool(adj_p <= inp.alpha)

        if se <= 0.0:
            ci_low, ci_high = float(diff), float(diff)
        else:
            ci_low = float(diff - z_crit * se)
            ci_high = float(diff + z_crit * se)

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="ratio",
                control_rate=c_ratio,
                treatment_rate=t_ratio,
                absolute_lift=diff,
                relative_lift=diff / max(abs(c_ratio), 1e-12),
                p_value=float(adj_p),
                raw_p_value=float(raw_p),
                test_statistic=float(z_stat) if z_stat is not None else None,
                degrees_of_freedom=None,
                significant=significant,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return out


def analyze_fixed_cuped(
    inp: AnalysisInput,
    control: VariantInput,
    treatments: list[VariantInput],
    cuped_arms: dict[str, CupedArm],
) -> list[ComparisonResult]:
    """Fixed-horizon Welch's two-sample t-test on CUPED-adjusted moments."""
    n_comparisons = len(treatments)
    alpha_adj = per_comparison_alpha(inp.alpha, n_comparisons)
    c = cuped_arms[control.name]

    out: list[ComparisonResult] = []
    for treatment in treatments:
        t = cuped_arms[treatment.name]
        out.append(
            _fixed_welch_comparison(
                inp=inp,
                control_name=control.name,
                treatment_name=treatment.name,
                metric=inp.primary_metric,
                n1=c.n,
                mean1=c.mean,
                var1=c.var,
                n2=t.n,
                mean2=t.mean,
                var2=t.var,
                n_comparisons=n_comparisons,
                alpha_adj=alpha_adj,
            )
        )
    return out


def sample_mean_posterior_from_moments(
    n: int,
    mean: float | np.ndarray,
    ss: float | np.ndarray,
    num_samples: int | tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Normal-Inverse-Gamma posterior mean draws from sample size, mean, and centered SS.

    ``mean`` and ``ss`` may be arrays that broadcast against ``num_samples`` (e.g. shape
    ``(sims, 1)`` with ``num_samples=(sims, draws)``) to draw many posteriors at once.
    """
    mu0 = 0.0
    kappa0 = 1e-6
    alpha0 = 1.0
    beta0 = 1.0

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * mean) / kappa_n
    alpha_n = alpha0 + 0.5 * n
    beta_n = beta0 + 0.5 * np.maximum(ss, 0.0) + (kappa0 * n * ((mean - mu0) ** 2)) / (2 * kappa_n)

    gamma_samples = rng.gamma(shape=alpha_n, scale=1.0, size=num_samples)
    sigma2_samples = beta_n / np.clip(gamma_samples, 1e-18, None)
    mu_samples = rng.normal(mu_n, np.sqrt(np.clip(sigma2_samples / kappa_n, 1e-18, None)))
    return mu_samples


def sample_mean_posterior(
    n: int,
    value_sum: float,
    value_sum_squares: float,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x_bar = value_sum / max(n, 1)
    ss = max(value_sum_squares - n * (x_bar**2), 0.0)
    return sample_mean_posterior_from_moments(n, x_bar, ss, num_samples, rng)


def recommend(
    inp: AnalysisInput,
    comparisons: list[ComparisonResult],
    guardrails_passed: bool,
    srm: SrmResult,
    *,
    guardrails: list[GuardrailResult] | None = None,
) -> Recommendation | None:
    if not inp.decision_policy.enabled:
        return None
    if inp.method == "bayesian" and inp.decision_policy.thresholds is None:
        return None

    risk_flags: list[str] = []
    if not srm.passed:
        risk_flags.append("srm_detected")
        if not guardrails_passed:
            risk_flags.append("guardrail_failure")
        return Recommendation(
            action="investigate_data_quality",
            rationale="SRM check failed; assignment or tracking may be biased.",
            decision_confidence=float(1.0 - srm.p_value),
            next_best_action="Audit randomization, traffic filters, and event logging before shipping.",
            risk_flags=risk_flags,
        )

    if guardrails is None:
        global_guardrail_failed = not guardrails_passed
        blocked_variants: set[str] = set()
    else:
        global_guardrail_failed = any(g.variant is None and not g.passed for g in guardrails)
        blocked_variants = {g.variant for g in guardrails if g.variant is not None and not g.passed}
        if not guardrails_passed and not global_guardrail_failed and not blocked_variants:
            global_guardrail_failed = True

    if global_guardrail_failed:
        return Recommendation(
            action="do_not_ship",
            rationale="One or more guardrails failed. Keep test running or roll back.",
            # Legacy threshold guardrails are deterministic rules, not statistical tests.
            decision_confidence=1.0,
            next_best_action="Fix guardrail regressions or reduce impact before re-testing.",
            risk_flags=["guardrail_failure"],
        )

    eligible = [c for c in comparisons if c.treatment not in blocked_variants]
    excluded = [c.treatment for c in comparisons if c.treatment in blocked_variants]

    if not eligible:
        return Recommendation(
            action="do_not_ship",
            rationale=(
                f"All treatments failed guardrails ({', '.join(excluded)}). "
                "Keep test running or roll back."
            ),
            decision_confidence=1.0,
            next_best_action="Fix guardrail regressions or reduce impact before re-testing.",
            risk_flags=["guardrail_failure"],
        )

    excluded_note = (
        f" Excluded due to guardrail failure: {', '.join(excluded)}." if excluded else ""
    )

    if inp.method == "bayesian":
        assert inp.decision_policy.thresholds is not None
        prob_threshold = inp.decision_policy.thresholds.bayes_prob_beats_control
        loss_threshold = inp.decision_policy.thresholds.max_expected_loss

        qualifying_winners = [
            c
            for c in eligible
            if c.probability_beats_control is not None
            and c.expected_loss is not None
            and c.absolute_lift > 0
            and c.probability_beats_control >= prob_threshold
            and c.expected_loss <= loss_threshold
        ]
        if qualifying_winners:
            winner = min(
                qualifying_winners,
                key=lambda x: (
                    x.multi_arm_expected_loss
                    if x.multi_arm_expected_loss is not None
                    else (x.expected_loss if x.expected_loss is not None else float("inf")),
                    -x.absolute_lift,
                ),
            )
            multi_arm_detail = (
                f", P(best)={winner.probability_best:.3f}, "
                f"multi_arm_expected_loss={winner.multi_arm_expected_loss:.6f}"
                if len(comparisons) > 1
                and winner.probability_best is not None
                and winner.multi_arm_expected_loss is not None
                else ""
            )
            flags = risk_flags + (["other_variant_guardrail_failure"] if excluded else [])
            return Recommendation(
                action=f"ship_{winner.treatment}",
                rationale=(
                    f"{winner.treatment} passes Bayesian thresholds "
                    f"(P(win)={winner.probability_beats_control:.3f}, "
                    f"expected_loss={winner.expected_loss:.6f}{multi_arm_detail}).{excluded_note}"
                ),
                decision_confidence=float(winner.probability_beats_control or 0.0),
                next_best_action="Roll out gradually and monitor guardrails.",
                risk_flags=flags,
            )

        qualifying_losers = [
            c
            for c in eligible
            if c.probability_beats_control is not None
            and c.expected_loss is not None
            and c.absolute_lift < 0
            and c.probability_beats_control <= (1.0 - prob_threshold)
        ]
        if qualifying_losers and len(qualifying_losers) == len(eligible):
            if len(qualifying_losers) == 1:
                loser = qualifying_losers[0]
                rationale = (
                    f"{loser.treatment} is unlikely to beat control "
                    f"(P(win)={loser.probability_beats_control:.3f}, "
                    f"expected_loss={loser.expected_loss:.6f}).{excluded_note}"
                )
            else:
                worst = min(qualifying_losers, key=lambda x: x.absolute_lift)
                rationale = (
                    f"Every treatment is unlikely to beat control "
                    f"(worst: {worst.treatment}, "
                    f"P(win)={worst.probability_beats_control:.3f}, "
                    f"expected_loss={worst.expected_loss:.6f}).{excluded_note}"
                )
            max_p_win = max(c.probability_beats_control or 0.0 for c in qualifying_losers)
            flags = risk_flags + (["guardrail_failure"] if excluded else [])
            return Recommendation(
                action="do_not_ship",
                rationale=rationale,
                decision_confidence=float(1.0 - max_p_win),
                next_best_action="Keep the control, or redefine the metric and test design before rerunning.",
                risk_flags=flags,
            )

        leading_prob = max(
            (
                c.probability_beats_control
                for c in eligible
                if c.probability_beats_control is not None
            ),
            default=0.5,
        )
        flags = risk_flags + (["some_variants_guardrail_failure"] if excluded else [])
        return Recommendation(
            action="continue_collecting_data",
            rationale=(
                f"Bayesian confidence or expected loss thresholds were not met.{excluded_note}"
            ),
            decision_confidence=float(leading_prob),
            next_best_action="Collect more samples until decision thresholds are reached.",
            risk_flags=flags,
        )

    winners = [c for c in eligible if c.significant and c.absolute_lift > 0]
    losers = [c for c in eligible if c.significant and c.absolute_lift < 0]

    if inp.method == "fixed":
        if winners:
            winner = max(winners, key=lambda x: x.absolute_lift)
            flags = list(risk_flags)
            if losers:
                flags.append("other_variant_significantly_worse")
            if excluded:
                flags.append("other_variant_guardrail_failure")
            return Recommendation(
                action=f"ship_{winner.treatment}",
                rationale=(
                    f"Fixed-horizon significance reached for {winner.treatment} "
                    f"(adjusted p={winner.p_value:.6g} <= alpha={inp.alpha:g}). "
                    f"Valid only when evaluated once at the planned experiment end.{excluded_note}"
                ),
                decision_confidence=float(1.0 - (winner.p_value or 0.0)),
                next_best_action=(
                    "Roll out gradually and continue guardrail monitoring. "
                    "Do not reuse fixed-horizon testing for repeated peeking."
                ),
                risk_flags=flags,
            )

        if losers and len(losers) == len(eligible):
            worst = min(losers, key=lambda x: x.absolute_lift)
            flags = risk_flags + (["guardrail_failure"] if excluded else [])
            return Recommendation(
                action="do_not_ship",
                rationale=(
                    f"Fixed-horizon significant negative lift detected for every treatment "
                    f"(worst: {worst.treatment}, adjusted p={worst.p_value:.6g} "
                    f"<= alpha={inp.alpha:g}). "
                    f"Valid only when evaluated once at the planned experiment end.{excluded_note}"
                ),
                decision_confidence=float(1.0 - max(c.p_value or 0.0 for c in losers)),
                next_best_action="Keep the control and investigate adverse drivers before re-testing.",
                risk_flags=flags,
            )

        flags = list(risk_flags)
        if losers:
            flags.append("some_variants_significantly_worse")
        if excluded:
            flags.append("some_variants_guardrail_failure")

        best_p = min((c.p_value for c in eligible if c.p_value is not None), default=1.0)
        return Recommendation(
            action="do_not_ship",
            rationale=(
                f"No treatment reached fixed-horizon significance at the planned look "
                f"(min adjusted p={best_p:.6g} > alpha={inp.alpha:g}). "
                f"Fixed-horizon results are only valid for a single planned evaluation.{excluded_note}"
            ),
            decision_confidence=float(1.0 - best_p),
            next_best_action=(
                "Keep the control rather than peeking repeatedly; use --method sequential "
                "for continuous monitoring or plan a new fixed-horizon run."
            ),
            risk_flags=flags,
        )

    if winners:
        winner = max(winners, key=lambda x: x.absolute_lift)
        flags = list(risk_flags)
        if losers:
            flags.append("other_variant_significantly_worse")
        if excluded:
            flags.append("other_variant_guardrail_failure")
        return Recommendation(
            action=f"ship_{winner.treatment}",
            rationale=(
                f"mSPRT significance reached for {winner.treatment} "
                f"(e-value={winner.e_value:.3f} >= threshold={winner.e_threshold:.1f})."
                f"{excluded_note}"
            ),
            decision_confidence=float(1.0 - (winner.p_value or 0.0)),
            next_best_action="Roll out gradually and continue guardrail monitoring.",
            risk_flags=flags,
        )

    if losers and len(losers) == len(eligible):
        worst = min(losers, key=lambda x: x.absolute_lift)
        flags = risk_flags + (["guardrail_failure"] if excluded else [])
        return Recommendation(
            action="stop_and_rollback",
            rationale=(
                f"Significant negative lift detected for every treatment "
                f"(worst: {worst.treatment}, e-value={worst.e_value:.3f} "
                f">= threshold={worst.e_threshold:.1f}).{excluded_note}"
            ),
            decision_confidence=float(1.0 - max(c.p_value or 0.0 for c in losers)),
            next_best_action="Roll back treatments and investigate adverse drivers.",
            risk_flags=flags,
        )

    flags = list(risk_flags)
    if losers:
        flags.append("some_variants_significantly_worse")
    if excluded:
        flags.append("some_variants_guardrail_failure")

    best_p = min((c.p_value for c in eligible if c.p_value is not None), default=1.0)
    return Recommendation(
        action="continue_collecting_data",
        rationale=(
            f"mSPRT significance threshold not reached yet; it is safe to keep looking.{excluded_note}"
        ),
        # Strength of evidence for the leading comparison (1 - always-valid p-value), on the
        # same scale as the ship/rollback confidences. It stays below 1 - alpha here.
        decision_confidence=float(1.0 - best_p),
        next_best_action=(
            "Collect more data and re-run; the always-valid test controls the error rate "
            "no matter how often you look."
        ),
        risk_flags=flags,
    )
