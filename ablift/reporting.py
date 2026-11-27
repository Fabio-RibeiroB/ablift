from __future__ import annotations

from .models import AnalysisResult, ComparisonResult, GuardrailResult


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _comparison_lines(comp: ComparisonResult, method: str) -> list[str]:
    if comp.metric == "conversion_rate":
        metric_label = "CR"
        control_display = _fmt_pct(comp.control_rate)
        treatment_display = _fmt_pct(comp.treatment_rate)
        abs_lift_display = _fmt_pct(comp.absolute_lift)
    else:
        metric_label = "Ratio" if comp.metric == "ratio" else "ARPU"
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
    ]
    if method == "bayesian":
        lines.append(
            f"  - 95% credible interval (absolute lift): "
            f"[{_fmt_float(comp.ci_low)}, {_fmt_float(comp.ci_high)}]"
        )
        if comp.relative_ci_low is not None and comp.relative_ci_high is not None:
            lines.append(
                f"  - 95% credible interval (relative lift): "
                f"[{_fmt_pct(comp.relative_ci_low)}, {_fmt_pct(comp.relative_ci_high)}]"
            )
    elif method == "fixed":
        lines.append(
            f"  - Confidence interval (absolute lift): "
            f"[{_fmt_float(comp.ci_low)}, {_fmt_float(comp.ci_high)}]"
        )
    else:
        lines.append(
            f"  - Always-valid confidence interval (absolute lift): "
            f"[{_fmt_float(comp.ci_low)}, {_fmt_float(comp.ci_high)}]"
        )

    if method == "bayesian":
        lines.extend(
            [
                f"  - P(treatment > control): {_fmt_float(comp.probability_beats_control, 4)}",
                f"  - Expected loss: {_fmt_float(comp.expected_loss, 6)}",
            ]
        )
        if comp.probability_best is not None:
            lines.append(f"  - P(best overall): {_fmt_float(comp.probability_best, 4)}")
        if comp.multi_arm_expected_loss is not None:
            lines.append(
                f"  - Multi-arm expected loss: {_fmt_float(comp.multi_arm_expected_loss, 6)}"
            )
    elif method == "fixed":
        if comp.degrees_of_freedom is not None:
            lines.append(
                f"  - Welch t-statistic: {_fmt_float(comp.test_statistic, 4)} "
                f"(df={_fmt_float(comp.degrees_of_freedom, 2)})"
            )
        else:
            lines.append(f"  - z-statistic: {_fmt_float(comp.test_statistic, 4)}")
        lines.extend(
            [
                f"  - Raw p-value: {_fmt_float(comp.raw_p_value, 6)}",
                f"  - Adjusted p-value: {_fmt_float(comp.p_value, 6)}",
                f"  - Significant: `{comp.significant}`",
            ]
        )
    else:
        lines.extend(
            [
                f"  - e-value: {_fmt_float(comp.e_value, 4)}",
                f"  - e-threshold: {_fmt_float(comp.e_threshold, 4)}",
                f"  - Always-valid p-value: {_fmt_float(comp.p_value, 6)}",
                f"  - Significant now: `{comp.significant}`",
            ]
        )

    return lines


def _guardrail_lines(guardrails: list[GuardrailResult], method: str) -> list[str]:
    if not guardrails:
        return ["No guardrails provided."]

    lines: list[str] = []
    for g in guardrails:
        header = (
            f"- `{g.name}` (`{g.variant}`): pass=`{g.passed}`"
            if g.variant is not None
            else f"- `{g.name}`: pass=`{g.passed}`"
        )
        entry = [header]
        if g.mode != "threshold":
            entry.append(f"  - mode: `{g.mode}`")
        entry.extend(
            [
                f"  - direction: `{g.direction}`",
                f"  - control: {g.control}",
                f"  - treatment: {g.treatment}",
                f"  - relative change: {_fmt_pct(g.relative_change)}",
                f"  - allowed: {_fmt_pct(g.max_relative_change)}",
            ]
        )
        if g.comparison is not None:
            comp = g.comparison
            entry.append(
                f"  - interval (absolute lift): [{_fmt_float(comp.ci_low)}, {_fmt_float(comp.ci_high)}]"
            )
            if method == "bayesian":
                entry.append(
                    f"  - P(treatment > control): {_fmt_float(comp.probability_beats_control, 4)}"
                )
            else:
                entry.append(f"  - p-value: {_fmt_float(comp.p_value, 6)}")
        entry.append(f"  - reason: {g.reason}")
        lines.extend(entry)
    return lines


def build_markdown_report(result: AnalysisResult) -> str:
    recommendation_action = (
        result.recommendation.action if result.recommendation else "not_evaluated"
    )
    recommendation_rationale = (
        result.recommendation.rationale
        if result.recommendation
        else "No decision policy was provided, so estimates are reported without an automated recommendation."
    )
    recommendation_confidence = (
        _fmt_float(result.recommendation.decision_confidence, 4) if result.recommendation else "n/a"
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
    ]
    settings = result.analysis_settings
    if result.method == "bayesian":
        lines.extend(
            [
                f"- Posterior samples: `{settings.samples}`",
                f"- Random seed: `{settings.random_seed}`",
                f"- Priors: `{settings.priors}`",
                f"- Decision policy: `{settings.decision_policy}`",
            ]
        )
    elif result.method == "fixed":
        mc = settings.multiple_comparisons
        lines.extend(
            [
                f"- Alpha (family-wise): `{settings.alpha}`",
                f"- Multiple comparisons: `{mc.get('correction')}` across "
                f"{mc.get('n_comparisons')} comparison(s), "
                f"alpha per comparison `{mc.get('alpha_per_comparison')}`",
                "- Horizon assumption: Fixed-horizon (valid for a single planned look only; "
                "do not peek repeatedly)",
            ]
        )
    else:
        mc = settings.multiple_comparisons
        lines.extend(
            [
                f"- Alpha (family-wise): `{settings.alpha}`",
                f"- Multiple comparisons: `{mc.get('correction')}` across "
                f"{mc.get('n_comparisons')} comparison(s), "
                f"alpha per comparison `{mc.get('alpha_per_comparison')}`",
                f"- mSPRT tau: `{settings.sequential_tau or 'default (10% of control baseline)'}`",
            ]
        )
    lines += [
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
        f"- SRM expected proportions: {', '.join(_fmt_pct(p) for p in result.srm.expected_proportions)}",
        f"- SRM reason: {result.srm.reason}",
    ]

    interp = settings.input_interpretation
    winsor = interp.get("winsorization")
    if isinstance(winsor, dict):
        lines.append(
            f"- Winsorization: pooled q=`{winsor.get('quantile')}`, "
            f"cap={_fmt_float(winsor.get('cap_value'), 4)}, "
            f"capped={winsor.get('capped_count')} ({_fmt_pct(float(winsor.get('capped_rate', 0.0)))}), "
            f"variance reduction={_fmt_pct(float(winsor.get('variance_reduction_ratio', 0.0)))}"
        )
    tail_diag = interp.get("arpu_tail_diagnostics")
    if isinstance(tail_diag, dict) and tail_diag.get("heavy_tailed"):
        if tail_diag.get("source") == "row_level":
            lines.append(
                f"- ARPU tail warning (`heavy_tailed_revenue`): "
                f"skewness={_fmt_float(tail_diag.get('max_arm_skewness'), 2)}, "
                f"max/|mean|={_fmt_float(tail_diag.get('max_to_mean_ratio'), 2)}, "
                f"Kohavi min n={_fmt_float(tail_diag.get('kohavi_min_n'), 0)} "
                f"(min arm n={tail_diag.get('min_arm_n')})"
            )
        else:
            lines.append(
                f"- ARPU tail warning (`heavy_tailed_revenue`): "
                f"max CV={_fmt_float(tail_diag.get('max_cv'), 2)} (> 3.0)"
            )
    cuped = interp.get("cuped")
    if isinstance(cuped, dict) and cuped.get("enabled"):
        lines.append(
            f"- CUPED variance reduction: theta={_fmt_float(cuped.get('theta'), 4)}, "
            f"R^2={_fmt_float(cuped.get('r_squared'), 4)}, "
            f"variance reduction={_fmt_float(cuped.get('variance_reduction_pct'), 2)}%, "
            f"covariate balance p={_fmt_float(cuped.get('covariate_balance_p_value'), 6)}"
        )
        if cuped.get("covariate_imbalanced"):
            lines.append(
                "- CUPED covariate balance warning (`covariate_imbalance`): "
                "pre-experiment covariate differs significantly across arms (p < 0.001); "
                "verify randomization and ensure the covariate was measured strictly pre-assignment."
            )

    lines.extend(["", "## Variant Comparisons"])

    bootstrap_ci = interp.get("bootstrap_ci")
    for comp in result.comparisons:
        lines.extend(_comparison_lines(comp, result.method))
        if isinstance(bootstrap_ci, dict) and isinstance(bootstrap_ci.get(comp.treatment), dict):
            bci = bootstrap_ci[comp.treatment]
            lines.append(
                f"  - Bootstrap CI (absolute lift, {bci.get('n_bootstrap')} resamples): "
                f"[{_fmt_float(bci.get('ci_low'))}, {_fmt_float(bci.get('ci_high'))}]"
            )

    if result.method == "bayesian" and result.bayesian_arms:
        lines.extend(["", "## Joint Posterior Summary"])
        for arm in result.bayesian_arms:
            role = "control" if arm.is_control else "treatment"
            lines.append(
                f"- `{arm.variant}` ({role}): "
                f"P(best)={_fmt_float(arm.probability_best, 4)}, "
                f"multi-arm expected loss={_fmt_float(arm.multi_arm_expected_loss, 6)}"
            )

    lines.extend(["", "## Guardrails"])
    lines.extend(_guardrail_lines(result.guardrails, result.method))

    if result.secondary_metrics:
        lines.extend(["", "## Secondary Metrics"])
        for sec in result.secondary_metrics:
            lines.extend(["", f"### {sec.name} (`{sec.metric}`)"])
            for comp in sec.comparisons:
                lines.extend(_comparison_lines(comp, result.method))

    lines.append("")

    return "\n".join(lines)
