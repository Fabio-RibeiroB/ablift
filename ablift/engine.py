from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .models import (
    AnalysisInput,
    AnalysisResult,
    AnalysisSettings,
    ComparisonResult,
    DecisionPolicy,
    DecisionThresholds,
    GuardrailInput,
    GuardrailResult,
    Recommendation,
    SrmResult,
    VariantInput,
)


def parse_payload(payload: dict) -> AnalysisInput:
    try:
        return _parse_payload(payload)
    except KeyError as exc:
        raise ValueError(f"Input is missing required field {exc}.") from exc
    except TypeError as exc:
        raise ValueError(f"Invalid input structure: {exc}") from exc


def _parse_payload(payload: dict) -> AnalysisInput:
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object.")
    variants = [VariantInput(**row) for row in payload["variants"]]
    guardrails = [GuardrailInput(**row) for row in payload.get("guardrails") or []]
    decision_policy_payload = payload.get("decision_policy")
    threshold_payload = payload.get("decision_thresholds")
    method = payload.get("method") or "bayesian"

    if decision_policy_payload is not None:
        enabled = bool(decision_policy_payload.get("enabled", True))
        merged_thresholds = {
            "bayes_prob_beats_control": decision_policy_payload.get(
                "bayes_prob_beats_control", 0.95
            ),
            "max_expected_loss": decision_policy_payload.get("max_expected_loss", 0.001),
        }
        thresholds = DecisionThresholds(**merged_thresholds) if enabled else None
        decision_policy = DecisionPolicy(
            enabled=enabled,
            thresholds=thresholds,
            source="decision_policy",
        )
    elif threshold_payload is not None:
        decision_policy = DecisionPolicy(
            enabled=True,
            thresholds=DecisionThresholds(**threshold_payload),
            source="decision_thresholds",
        )
    elif method == "bayesian":
        decision_policy = DecisionPolicy(enabled=False, thresholds=None, source="none")
    else:
        decision_policy = DecisionPolicy(
            enabled=True, thresholds=None, source="implicit_method_defaults"
        )

    return AnalysisInput(
        experiment_name=payload.get("experiment_name") or "experiment",
        method=method,
        variants=variants,
        primary_metric=payload.get("primary_metric", "conversion_rate"),
        alpha=payload.get("alpha", 0.05),
        sequential_tau=payload.get("sequential_tau"),
        guardrails=guardrails,
        decision_policy=decision_policy,
        random_seed=payload.get("random_seed", 7),
        samples=payload.get("samples", 50000),
        input_interpretation=payload.get("input_interpretation", {}),
    )


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

    if inp.alpha <= 0 or inp.alpha >= 1:
        raise ValueError("alpha must be in (0, 1).")

    if inp.sequential_tau is not None and inp.sequential_tau <= 0:
        raise ValueError("sequential_tau must be positive.")
    if inp.samples <= 0:
        raise ValueError("samples must be positive.")

    if inp.method not in {"bayesian", "sequential"}:
        raise ValueError("method must be 'bayesian' or 'sequential'.")
    if inp.primary_metric not in {"conversion_rate", "arpu"}:
        raise ValueError("primary_metric must be 'conversion_rate' or 'arpu'.")

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


def analyze(inp: AnalysisInput) -> AnalysisResult:
    validate_input(inp)

    control = [v for v in inp.variants if v.is_control][0]
    treatments = [v for v in inp.variants if not v.is_control]

    guardrail_results = evaluate_guardrails(inp)
    guardrails_passed = all(g.passed for g in guardrail_results)
    srm_result = evaluate_srm(inp.variants)

    if inp.method == "bayesian":
        if inp.primary_metric == "conversion_rate":
            comparisons = analyze_bayesian_conversion(inp, control, treatments)
        else:
            comparisons = analyze_bayesian_arpu(inp, control, treatments)
    else:
        if inp.primary_metric == "conversion_rate":
            comparisons = analyze_sequential_conversion(inp, control, treatments)
        else:
            comparisons = analyze_sequential_arpu(inp, control, treatments)

    recommendation = recommend(inp, comparisons, guardrails_passed, srm_result)

    return AnalysisResult(
        experiment_name=inp.experiment_name,
        method=inp.method,
        analysis_settings=build_analysis_settings(inp),
        control_variant=control.name,
        comparisons=comparisons,
        guardrails_passed=guardrails_passed,
        guardrails=guardrail_results,
        srm=srm_result,
        recommendation=recommendation,
    )


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


def evaluate_srm(variants: list[VariantInput]) -> SrmResult:
    observed = [v.visitors for v in variants]
    total = float(sum(observed))
    k = len(observed)
    expected = [total / k for _ in range(k)]

    chi2 = 0.0
    for obs, exp in zip(observed, expected, strict=False):
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
        reason=reason,
    )


def evaluate_guardrails(inp: AnalysisInput) -> list[GuardrailResult]:
    results: list[GuardrailResult] = []

    for item in inp.guardrails:
        if item.control == 0:
            relative_change = item.treatment - item.control
        else:
            relative_change = (item.treatment - item.control) / abs(item.control)

        direction = item.direction.lower().strip()
        if direction not in {"increase", "decrease"}:
            raise ValueError(
                f"guardrail direction must be 'increase' or 'decrease' for {item.name}."
            )

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


def analyze_bayesian_conversion(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    rng = np.random.default_rng(inp.random_seed)

    alpha_prior = 1.0
    beta_prior = 1.0

    c_a = alpha_prior + control.conversions
    c_b = beta_prior + (control.visitors - control.conversions)
    control_samples = rng.beta(c_a, c_b, size=inp.samples)

    out: list[ComparisonResult] = []
    for treatment in treatments:
        t_a = alpha_prior + treatment.conversions
        t_b = beta_prior + (treatment.visitors - treatment.conversions)
        treatment_samples = rng.beta(t_a, t_b, size=inp.samples)

        diff_samples = treatment_samples - control_samples
        rel_samples = np.divide(
            diff_samples,
            np.clip(control_samples, 1e-12, None),
        )

        p_win = float(np.mean(treatment_samples > control_samples))
        expected_loss = float(np.mean(np.maximum(control_samples - treatment_samples, 0.0)))

        control_rate = control.conversions / control.visitors
        treatment_rate = treatment.conversions / treatment.visitors

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="conversion_rate",
                control_rate=control_rate,
                treatment_rate=treatment_rate,
                absolute_lift=treatment_rate - control_rate,
                relative_lift=(treatment_rate - control_rate) / max(control_rate, 1e-12),
                probability_beats_control=p_win,
                expected_loss=expected_loss,
                ci_low=float(np.quantile(diff_samples, 0.025)),
                ci_high=float(np.quantile(diff_samples, 0.975)),
                relative_ci_low=float(np.quantile(rel_samples, 0.025)),
                relative_ci_high=float(np.quantile(rel_samples, 0.975)),
            )
        )

    return out


def build_analysis_settings(inp: AnalysisInput) -> AnalysisSettings:
    if inp.method == "bayesian" and inp.primary_metric == "conversion_rate":
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
    if inp.method == "sequential":
        multiple_comparisons = {
            "correction": "bonferroni",
            "n_comparisons": n_comparisons,
            "alpha_per_comparison": per_comparison_alpha(inp.alpha, n_comparisons),
        }
    else:
        multiple_comparisons = {"correction": "none", "n_comparisons": n_comparisons}

    return AnalysisSettings(
        primary_metric=inp.primary_metric,
        alpha=inp.alpha,
        multiple_comparisons=multiple_comparisons,
        sequential_tau=inp.sequential_tau,
        samples=inp.samples,
        random_seed=inp.random_seed,
        priors=priors,
        decision_policy=policy,
        input_interpretation=inp.input_interpretation,
    )


def analyze_bayesian_arpu(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    rng = np.random.default_rng(inp.random_seed)

    control_samples = sample_mean_posterior(
        control.visitors,
        float(control.revenue_sum or 0.0),
        float(control.revenue_sum_squares or 0.0),
        inp.samples,
        rng,
    )

    out: list[ComparisonResult] = []
    for treatment in treatments:
        treatment_samples = sample_mean_posterior(
            treatment.visitors,
            float(treatment.revenue_sum or 0.0),
            float(treatment.revenue_sum_squares or 0.0),
            inp.samples,
            rng,
        )

        diff_samples = treatment_samples - control_samples
        rel_samples = np.divide(
            diff_samples,
            np.clip(control_samples, 1e-12, None),
        )

        p_win = float(np.mean(treatment_samples > control_samples))
        expected_loss = float(np.mean(np.maximum(control_samples - treatment_samples, 0.0)))

        control_arpu = float(control.revenue_sum or 0.0) / control.visitors
        treatment_arpu = float(treatment.revenue_sum or 0.0) / treatment.visitors

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="arpu",
                control_rate=control_arpu,
                treatment_rate=treatment_arpu,
                absolute_lift=treatment_arpu - control_arpu,
                relative_lift=(treatment_arpu - control_arpu) / max(control_arpu, 1e-12),
                probability_beats_control=p_win,
                expected_loss=expected_loss,
                ci_low=float(np.quantile(diff_samples, 0.025)),
                ci_high=float(np.quantile(diff_samples, 0.975)),
                relative_ci_low=float(np.quantile(rel_samples, 0.025)),
                relative_ci_high=float(np.quantile(rel_samples, 0.975)),
            )
        )

    return out


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
    p_value = min(1.0, 1.0 / e_val) if e_val > 0 else 1.0
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


def sample_mean_posterior(
    n: int,
    value_sum: float,
    value_sum_squares: float,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # Normal-Inverse-Gamma weak prior over unknown mean/variance using aggregate stats.
    mu0 = 0.0
    kappa0 = 1e-6
    alpha0 = 1.0
    beta0 = 1.0

    x_bar = value_sum / max(n, 1)
    ss = max(value_sum_squares - n * (x_bar**2), 0.0)

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    alpha_n = alpha0 + 0.5 * n
    beta_n = beta0 + 0.5 * ss + (kappa0 * n * ((x_bar - mu0) ** 2)) / (2 * kappa_n)

    gamma_samples = rng.gamma(shape=alpha_n, scale=1.0, size=num_samples)
    sigma2_samples = beta_n / np.clip(gamma_samples, 1e-18, None)
    mu_samples = rng.normal(mu_n, np.sqrt(np.clip(sigma2_samples / kappa_n, 1e-18, None)))
    return mu_samples


def recommend(
    inp: AnalysisInput,
    comparisons: list[ComparisonResult],
    guardrails_passed: bool,
    srm: SrmResult,
) -> Recommendation | None:
    risk_flags: list[str] = []
    if not srm.passed:
        risk_flags.append("srm_detected")
    if not guardrails_passed:
        risk_flags.append("guardrail_failure")

    if not srm.passed:
        return Recommendation(
            action="investigate_data_quality",
            rationale="SRM check failed; assignment or tracking may be biased.",
            decision_confidence=float(1.0 - srm.p_value),
            next_best_action="Audit randomization, traffic filters, and event logging before shipping.",
            risk_flags=risk_flags,
        )

    if not guardrails_passed:
        return Recommendation(
            action="do_not_ship",
            rationale="One or more guardrails failed. Keep test running or roll back.",
            # Guardrails are deterministic threshold rules, not statistical tests.
            decision_confidence=1.0,
            next_best_action="Fix guardrail regressions or reduce impact before re-testing.",
            risk_flags=risk_flags,
        )

    best = max(comparisons, key=lambda x: x.absolute_lift)

    if inp.method == "bayesian":
        if not inp.decision_policy.enabled or inp.decision_policy.thresholds is None:
            return None

        prob_threshold = inp.decision_policy.thresholds.bayes_prob_beats_control
        loss_threshold = inp.decision_policy.thresholds.max_expected_loss

        if (
            best.probability_beats_control is not None
            and best.expected_loss is not None
            and best.absolute_lift > 0
            and best.probability_beats_control >= prob_threshold
            and best.expected_loss <= loss_threshold
        ):
            return Recommendation(
                action=f"ship_{best.treatment}",
                rationale=(
                    f"{best.treatment} passes Bayesian thresholds "
                    f"(P(win)={best.probability_beats_control:.3f}, "
                    f"expected_loss={best.expected_loss:.6f})."
                ),
                decision_confidence=float(best.probability_beats_control),
                next_best_action="Roll out gradually and monitor guardrails.",
                risk_flags=risk_flags,
            )

        if (
            best.probability_beats_control is not None
            and best.expected_loss is not None
            and best.absolute_lift < 0
            and best.probability_beats_control <= (1.0 - prob_threshold)
        ):
            return Recommendation(
                action="do_not_ship",
                rationale=(
                    f"{best.treatment} is unlikely to beat control "
                    f"(P(win)={best.probability_beats_control:.3f}, "
                    f"expected_loss={best.expected_loss:.6f})."
                ),
                decision_confidence=float(1.0 - best.probability_beats_control),
                next_best_action="Keep the control, or redefine the metric and test design before rerunning.",
                risk_flags=risk_flags,
            )

        return Recommendation(
            action="continue_collecting_data",
            rationale="Bayesian confidence or expected loss thresholds were not met.",
            decision_confidence=float(best.probability_beats_control or 0.5),
            next_best_action="Collect more samples until decision thresholds are reached.",
            risk_flags=risk_flags,
        )

    winners = [c for c in comparisons if c.significant and c.absolute_lift > 0]
    losers = [c for c in comparisons if c.significant and c.absolute_lift < 0]

    if winners:
        winner = max(winners, key=lambda x: x.absolute_lift)
        flags = risk_flags + (["other_variant_significantly_worse"] if losers else [])
        return Recommendation(
            action=f"ship_{winner.treatment}",
            rationale=(
                f"mSPRT significance reached for {winner.treatment} "
                f"(e-value={winner.e_value:.3f} >= threshold={winner.e_threshold:.1f})."
            ),
            decision_confidence=float(1.0 - (winner.p_value or 0.0)),
            next_best_action="Roll out gradually and continue guardrail monitoring.",
            risk_flags=flags,
        )

    if losers and len(losers) == len(comparisons):
        worst = min(losers, key=lambda x: x.absolute_lift)
        return Recommendation(
            action="stop_and_rollback",
            rationale=(
                f"Significant negative lift detected for every treatment "
                f"(worst: {worst.treatment}, e-value={worst.e_value:.3f} "
                f">= threshold={worst.e_threshold:.1f})."
            ),
            decision_confidence=float(1.0 - max(c.p_value or 0.0 for c in losers)),
            next_best_action="Roll back treatments and investigate adverse drivers.",
            risk_flags=risk_flags,
        )

    if losers:
        risk_flags = risk_flags + ["some_variants_significantly_worse"]

    best_p = min((c.p_value for c in comparisons if c.p_value is not None), default=1.0)
    return Recommendation(
        action="continue_collecting_data",
        rationale="mSPRT significance threshold not reached yet; it is safe to keep looking.",
        decision_confidence=float(best_p),
        next_best_action=(
            "Collect more data and re-run; the always-valid test controls the error rate "
            "no matter how often you look."
        ),
        risk_flags=risk_flags,
    )
