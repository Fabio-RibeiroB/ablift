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
    variants = [VariantInput(**row) for row in payload["variants"]]
    guardrails = [GuardrailInput(**row) for row in payload.get("guardrails", [])]
    decision_policy_payload = payload.get("decision_policy")
    threshold_payload = payload.get("decision_thresholds")
    method = payload["method"]

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
        experiment_name=payload["experiment_name"],
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


def evaluate_srm(variants: list[VariantInput]) -> SrmResult:
    observed = [v.visitors for v in variants]
    total = float(sum(observed))
    k = len(observed)
    expected = [total / k for _ in range(k)]

    chi2 = 0.0
    for obs, exp in zip(observed, expected, strict=False):
        if exp > 0:
            chi2 += ((obs - exp) ** 2) / exp

    # Wilson-Hilferty approximation for chi-square upper-tail p-value.
    df = max(k - 1, 1)
    x = max(chi2 / df, 1e-12)
    z = ((x ** (1 / 3)) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    p_value = 1 - NormalDist().cdf(z)
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
                ci_low=float(np.quantile(rel_samples, 0.025)),
                ci_high=float(np.quantile(rel_samples, 0.975)),
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

    return AnalysisSettings(
        primary_metric=inp.primary_metric,
        alpha=inp.alpha,
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
                ci_low=float(np.quantile(rel_samples, 0.025)),
                ci_high=float(np.quantile(rel_samples, 0.975)),
            )
        )

    return out


def _default_tau(baseline: float) -> float:
    """Default mSPRT mixing parameter: 10% relative to baseline."""
    return max(abs(baseline) * 0.1, 1e-6)


def msrpt_statistic(delta_hat: float, se: float, tau: float) -> float:
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


def analyze_sequential_conversion(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    e_threshold = 1.0 / inp.alpha
    control_rate = control.conversions / control.visitors

    out: list[ComparisonResult] = []
    for treatment in treatments:
        _, _, se = two_proportion_test(
            control.conversions,
            control.visitors,
            treatment.conversions,
            treatment.visitors,
        )
        treatment_rate = treatment.conversions / treatment.visitors
        diff = treatment_rate - control_rate
        tau = inp.sequential_tau if inp.sequential_tau is not None else _default_tau(control_rate)

        e_val = msrpt_statistic(diff, se, tau)
        margin = always_valid_ci_margin(se, tau, inp.alpha)

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="conversion_rate",
                control_rate=control_rate,
                treatment_rate=treatment_rate,
                absolute_lift=diff,
                relative_lift=diff / max(control_rate, 1e-12),
                e_value=float(e_val),
                e_threshold=float(e_threshold),
                significant=bool(e_val >= e_threshold),
                ci_low=float(diff - margin),
                ci_high=float(diff + margin),
            )
        )

    return out


def analyze_sequential_arpu(
    inp: AnalysisInput, control: VariantInput, treatments: list[VariantInput]
) -> list[ComparisonResult]:
    e_threshold = 1.0 / inp.alpha
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
        se = math.sqrt(max((c_var / control.visitors) + (t_var / treatment.visitors), 1e-18))
        diff = t_mean - c_mean
        tau = inp.sequential_tau if inp.sequential_tau is not None else _default_tau(c_mean)

        e_val = msrpt_statistic(diff, se, tau)
        margin = always_valid_ci_margin(se, tau, inp.alpha)

        out.append(
            ComparisonResult(
                treatment=treatment.name,
                control=control.name,
                metric="arpu",
                control_rate=c_mean,
                treatment_rate=t_mean,
                absolute_lift=diff,
                relative_lift=diff / max(c_mean, 1e-12),
                e_value=float(e_val),
                e_threshold=float(e_threshold),
                significant=bool(e_val >= e_threshold),
                ci_low=float(diff - margin),
                ci_high=float(diff + margin),
            )
        )

    return out


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
            decision_confidence=0.99,
            next_best_action="Audit randomization, traffic filters, and event logging before shipping.",
            risk_flags=risk_flags,
        )

    if not guardrails_passed:
        return Recommendation(
            action="do_not_ship",
            rationale="One or more guardrails failed. Keep test running or roll back.",
            decision_confidence=0.95,
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

    if best.significant and best.absolute_lift > 0:
        return Recommendation(
            action=f"ship_{best.treatment}",
            rationale=(
                f"mSPRT significance reached for {best.treatment} "
                f"(e-value={best.e_value:.3f} >= threshold={best.e_threshold:.1f})."
            ),
            decision_confidence=0.95,
            next_best_action="Roll out gradually and continue guardrail monitoring.",
            risk_flags=risk_flags,
        )

    if best.significant and best.absolute_lift < 0:
        return Recommendation(
            action="stop_and_rollback",
            rationale=(
                f"Significant negative lift detected (e-value={best.e_value:.3f} "
                f">= threshold={best.e_threshold:.1f})."
            ),
            decision_confidence=0.95,
            next_best_action="Rollback treatment and investigate adverse drivers.",
            risk_flags=risk_flags,
        )

    return Recommendation(
        action="continue_collecting_data",
        rationale="mSPRT significance threshold not reached yet — safe to keep looking.",
        decision_confidence=0.5,
        next_best_action="Collect more data; the always-valid test controls error rate at any sample size.",
        risk_flags=risk_flags,
    )
