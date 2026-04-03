from __future__ import annotations

from .models import AnalysisResult, ComparisonResult, GuardrailResult


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _comparison_lines(comp: ComparisonResult, method: str) -> list[str]:
    metric_label = "CR" if comp.metric == "conversion_rate" else "ARPU"
    if comp.metric == "conversion_rate":
        control_display = _fmt_pct(comp.control_rate)
        treatment_display = _fmt_pct(comp.treatment_rate)
        abs_lift_display = _fmt_pct(comp.absolute_lift)
    else:
        control_display = _fmt_float(comp.control_rate, 6)
        treatment_display = _fmt_float(comp.treatment_rate, 6)
        abs_lift_display = _fmt_float(comp.absolute_lift, 6)

    lines = [
        f"- Treatment: `{comp.treatment}` vs Control: `{comp.control}`",
        f"  - Metric: `{comp.metric}`",
        f"  - Control {metric_label}: {control_display}",
        f"  - Treatment {metric_label}: {treatment_display}",
        f"  - Absolute lift: {abs_lift_display}",
        f"  - Relative lift: {_fmt_pct(comp.relative_lift)}",
        f"  - Interval: [{_fmt_float(comp.ci_low)}, {_fmt_float(comp.ci_high)}]",
    ]

    if method == "bayesian":
        lines.extend(
            [
                f"  - P(treatment > control): {_fmt_float(comp.probability_beats_control, 4)}",
                f"  - Expected loss: {_fmt_float(comp.expected_loss, 6)}",
            ]
        )
    else:
        lines.extend(
            [
                f"  - p-value: {_fmt_float(comp.p_value, 6)}",
                f"  - Alpha spent: {_fmt_float(comp.alpha_spent, 6)}",
                f"  - Significant now: `{comp.significant}`",
            ]
        )

    return lines


def _guardrail_lines(guardrails: list[GuardrailResult]) -> list[str]:
    if not guardrails:
        return ["No guardrails provided."]

    lines: list[str] = []
    for g in guardrails:
        lines.extend(
            [
                f"- `{g.name}`: pass=`{g.passed}`",
                f"  - direction: `{g.direction}`",
                f"  - control: {g.control}",
                f"  - treatment: {g.treatment}",
                f"  - relative change: {_fmt_pct(g.relative_change)}",
                f"  - allowed: {_fmt_pct(g.max_relative_change)}",
                f"  - reason: {g.reason}",
            ]
        )
    return lines


def build_markdown_report(result: AnalysisResult) -> str:
    recommendation_action = result.recommendation.action if result.recommendation else "not_evaluated"
    recommendation_rationale = (
        result.recommendation.rationale
        if result.recommendation
        else "No decision policy was provided, so estimates are reported without an automated recommendation."
    )
    recommendation_confidence = (
        _fmt_float(result.recommendation.decision_confidence, 4)
        if result.recommendation
        else "n/a"
    )
    recommendation_next = (
        result.recommendation.next_best_action
        if result.recommendation
        else "Provide an explicit decision policy if you want an automated recommendation."
    )
    recommendation_flags = (
        ", ".join(result.recommendation.risk_flags) if result.recommendation else "none"
    )
    lines = [
        f"# A/B Decision Report: {result.experiment_name}",
        "",
        "## Summary",
        f"- Method: `{result.method}`",
        f"- Primary metric: `{result.analysis_settings.primary_metric}`",
        f"- Posterior samples: `{result.analysis_settings.samples}`",
        f"- Random seed: `{result.analysis_settings.random_seed}`",
        f"- Priors: `{result.analysis_settings.priors}`",
        f"- Decision policy: `{result.analysis_settings.decision_policy}`",
        f"- Control: `{result.control_variant}`",
        f"- Guardrails passed: `{result.guardrails_passed}`",
        f"- Recommendation: `{recommendation_action}`",
        f"- Rationale: {recommendation_rationale}",
        f"- Decision confidence: {recommendation_confidence}",
        f"- Next best action: {recommendation_next}",
        f"- Risk flags: `{recommendation_flags or 'none'}`",
        "",
        "## Data Quality",
        f"- SRM passed: `{result.srm.passed}`",
        f"- SRM p-value: {_fmt_float(result.srm.p_value, 6)}",
        f"- SRM reason: {result.srm.reason}",
        "",
        "## Variant Comparisons",
    ]

    for comp in result.comparisons:
        lines.extend(_comparison_lines(comp, result.method))

    lines.extend(["", "## Guardrails"])
    lines.extend(_guardrail_lines(result.guardrails))
    lines.append("")

    return "\n".join(lines)
