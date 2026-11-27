from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from .engine import sample_mean_posterior_from_moments


@dataclass
class DurationPlan:
    method: str
    estimated_days: float | None
    n_per_variant: int | None
    assumptions: dict
    diagnostics: dict


def _z(p: float) -> float:
    return NormalDist().inv_cdf(p)


def _effect(baseline_rate: float, relative_mde: float) -> tuple[float, float, float]:
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1).")
    if relative_mde == 0:
        raise ValueError("relative_mde must be non-zero.")
    p1 = baseline_rate
    p2 = min(max(baseline_rate * (1 + relative_mde), 1e-6), 1 - 1e-6)
    return p1, p2, abs(p2 - p1)


def _arpu_effect(
    baseline_mean: float,
    relative_mde: float,
    baseline_sd: float | None = None,
    baseline_cv: float | None = None,
) -> tuple[float, float, float, float]:
    if not math.isfinite(baseline_mean) or baseline_mean == 0:
        raise ValueError("baseline_mean must be non-zero.")
    if not math.isfinite(relative_mde) or relative_mde == 0:
        raise ValueError("relative_mde must be non-zero.")
    if baseline_sd is not None and baseline_cv is not None:
        raise ValueError("Specify either baseline_sd or baseline_cv, not both.")
    if baseline_sd is None and baseline_cv is None:
        raise ValueError("Specify either baseline_sd or baseline_cv.")
    if baseline_sd is not None:
        if not math.isfinite(baseline_sd) or baseline_sd <= 0:
            raise ValueError("baseline_sd must be positive.")
        sd = float(baseline_sd)
    else:
        assert baseline_cv is not None
        if not math.isfinite(baseline_cv) or baseline_cv <= 0:
            raise ValueError("baseline_cv must be positive.")
        sd = abs(baseline_mean) * float(baseline_cv)

    mu1 = float(baseline_mean)
    mu2 = mu1 * (1.0 + relative_mde)
    delta = abs(mu1 * relative_mde)
    return mu1, mu2, delta, sd


def _validate_common(daily_total_traffic: int, n_variants: int, alpha: float, power: float):
    if daily_total_traffic <= 0:
        raise ValueError("daily_total_traffic must be positive.")
    if n_variants < 2:
        raise ValueError("n_variants must be at least 2 (control plus one treatment).")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1).")


def _msprt_power_for_variance(v: float, delta: float, tau: float, alpha_adj: float) -> float:
    """Probability the mSPRT has rejected when the difference estimator has variance ``v``.

    Evaluated at the final sample size only, which lower-bounds the probability of having
    crossed at any earlier look, so the resulting duration is slightly conservative.
    """
    t2 = tau * tau
    # Lambda(delta_hat) >= 1/alpha  <=>  |delta_hat| >= c
    c = math.sqrt(2 * v * (v + t2) / t2 * (math.log(1 / alpha_adj) + 0.5 * math.log((v + t2) / v)))
    sd = math.sqrt(v)
    nd = NormalDist()
    return (1 - nd.cdf((c - delta) / sd)) + nd.cdf((-c - delta) / sd)


def _search_msprt_n(
    unit_variance: float, delta: float, tau: float, alpha_adj: float, power: float
) -> int:
    lo, hi = 2, 2
    while _msprt_power_for_variance(unit_variance / hi, delta, tau, alpha_adj) < power:
        hi *= 2
        if hi > 10**12:
            raise ValueError("Effect is too small to detect with any realistic sample size.")
    while lo < hi:
        mid = (lo + hi) // 2
        if _msprt_power_for_variance(unit_variance / mid, delta, tau, alpha_adj) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fixed_horizon_duration_conversion(
    baseline_rate: float,
    relative_mde: float,
    daily_total_traffic: int,
    n_variants: int = 2,
    alpha: float = 0.05,
    power: float = 0.8,
) -> DurationPlan:
    """Classic two-sided two-proportion sample size, Bonferroni-adjusted for A/B/n.

    Only valid if you look at the data once, at the planned end date.
    """
    _validate_common(daily_total_traffic, n_variants, alpha, power)
    p1, p2, delta = _effect(baseline_rate, relative_mde)
    alpha_adj = alpha / (n_variants - 1)

    p_bar = 0.5 * (p1 + p2)
    z_alpha = _z(1 - alpha_adj / 2)
    z_beta = _z(power)
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n_per_variant = int(math.ceil(numerator / (delta**2)))
    daily_per_variant = daily_total_traffic / n_variants

    return DurationPlan(
        method="fixed_horizon",
        estimated_days=n_per_variant / daily_per_variant,
        n_per_variant=n_per_variant,
        assumptions={
            "metric": "conversion_rate",
            "baseline_rate": baseline_rate,
            "relative_mde": relative_mde,
            "alpha": alpha,
            "power": power,
            "daily_total_traffic": daily_total_traffic,
            "n_variants": n_variants,
        },
        diagnostics={
            "p_control": p1,
            "p_treatment": p2,
            "absolute_delta": delta,
            "alpha_per_comparison": alpha_adj,
        },
    )


def sequential_duration_conversion(
    baseline_rate: float,
    relative_mde: float,
    daily_total_traffic: int,
    n_variants: int = 2,
    alpha: float = 0.05,
    power: float = 0.8,
    sequential_tau: float | None = None,
) -> DurationPlan:
    """Sample size for the mSPRT used by ``ablift analyze --method sequential``.

    Finds the smallest per-variant sample size at which the always-valid test reaches the
    requested power. You may look at the results as often as you like before then.
    """
    _validate_common(daily_total_traffic, n_variants, alpha, power)
    p1, p2, delta = _effect(baseline_rate, relative_mde)
    alpha_adj = alpha / (n_variants - 1)
    tau = sequential_tau if sequential_tau is not None else max(abs(p1) * 0.1, 1e-6)
    if tau <= 0:
        raise ValueError("sequential_tau must be positive.")

    unit_variance = p1 * (1 - p1) + p2 * (1 - p2)
    n_per_variant = _search_msprt_n(unit_variance, p2 - p1, tau, alpha_adj, power)

    fixed = fixed_horizon_duration_conversion(
        baseline_rate, relative_mde, daily_total_traffic, n_variants, alpha, power
    )
    daily_per_variant = daily_total_traffic / n_variants
    return DurationPlan(
        method="sequential",
        estimated_days=n_per_variant / daily_per_variant,
        n_per_variant=n_per_variant,
        assumptions={
            "metric": "conversion_rate",
            "baseline_rate": baseline_rate,
            "relative_mde": relative_mde,
            "alpha": alpha,
            "power": power,
            "daily_total_traffic": daily_total_traffic,
            "n_variants": n_variants,
            "sequential_tau": tau,
        },
        diagnostics={
            "p_control": p1,
            "p_treatment": p2,
            "absolute_delta": delta,
            "alpha_per_comparison": alpha_adj,
            "fixed_horizon_n_per_variant": fixed.n_per_variant,
            "sequential_inflation": n_per_variant / max(fixed.n_per_variant or 1, 1),
            "note": (
                "Upper bound: the test may stop earlier if the true effect is larger than the MDE."
            ),
        },
    )


def _first_day_reaching(
    target: float, days: list[int], assurance_at: Callable[[int], float]
) -> tuple[int, float] | None:
    """Earliest day whose simulated assurance reaches ``target``, or None within ``days``.

    Assurance rises with sample size, so this doubles the index until the target is met and
    then bisects, needing O(log len(days)) simulations instead of one per day.
    """
    cache: dict[int, float] = {}

    def at(i: int) -> float:
        if i not in cache:
            cache[i] = assurance_at(days[i])
        return cache[i]

    if not days:
        return None
    lo, hi = -1, 0
    while at(hi) < target:
        if hi == len(days) - 1:
            return None
        lo, hi = hi, min(2 * hi + 1, len(days) - 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if at(mid) >= target:
            hi = mid
        else:
            lo = mid
    return days[hi], at(hi)


def _decision_rate(
    c_draws: np.ndarray,
    t_draws: np.ndarray,
    t_beats_c: np.ndarray,
    prob_threshold: float,
    max_expected_loss: float,
) -> float:
    """Share of simulated experiments (rows) that meet the Bayesian ship thresholds."""
    p_win = (t_draws > c_draws).mean(axis=1)
    expected_loss = np.maximum(c_draws - t_draws, 0.0).mean(axis=1)
    return float(
        np.mean((p_win >= prob_threshold) & (expected_loss <= max_expected_loss) & t_beats_c)
    )


def bayesian_duration_conversion(
    baseline_rate: float,
    relative_mde: float,
    daily_total_traffic: int,
    n_variants: int = 2,
    prob_threshold: float = 0.95,
    max_expected_loss: float = 0.001,
    assurance_target: float = 0.8,
    max_days: int = 60,
    step_days: int = 1,
    sims: int = 300,
    posterior_draws: int = 2000,
    seed: int = 7,
) -> DurationPlan:
    rng = np.random.default_rng(seed)
    p_control, p_treatment, _ = _effect(baseline_rate, relative_mde)

    if daily_total_traffic <= 0:
        raise ValueError("daily_total_traffic must be positive.")
    if n_variants < 2:
        raise ValueError("n_variants must be at least 2 (control plus one treatment).")

    daily_per_variant = daily_total_traffic / n_variants
    candidates = list(range(max(step_days, 1), max_days + 1, max(step_days, 1)))

    def n_on(day: int) -> int:
        return int(max(1, round(day * daily_per_variant)))

    def assurance_at(day: int) -> float:
        n = n_on(day)
        c = rng.binomial(n, p_control, size=(sims, 1))
        t = rng.binomial(n, p_treatment, size=(sims, 1))
        c_draws = rng.beta(1 + c, 1 + (n - c), size=(sims, posterior_draws))
        t_draws = rng.beta(1 + t, 1 + (n - t), size=(sims, posterior_draws))
        return _decision_rate(c_draws, t_draws, (t > c)[:, 0], prob_threshold, max_expected_loss)

    found = _first_day_reaching(assurance_target, candidates, assurance_at)
    selected_day = float(found[0]) if found else None
    selected_assurance = found[1] if found else None
    selected_n = n_on(found[0]) if found else None

    return DurationPlan(
        method="bayesian",
        estimated_days=selected_day,
        n_per_variant=selected_n,
        assumptions={
            "metric": "conversion_rate",
            "baseline_rate": baseline_rate,
            "relative_mde": relative_mde,
            "daily_total_traffic": daily_total_traffic,
            "n_variants": n_variants,
            "prob_threshold": prob_threshold,
            "max_expected_loss": max_expected_loss,
            "assurance_target": assurance_target,
            "max_days": max_days,
            "step_days": step_days,
            "sims": sims,
            "posterior_draws": posterior_draws,
        },
        diagnostics={
            "assurance_at_selected_day": selected_assurance,
            "p_control": p_control,
            "p_treatment": p_treatment,
        },
    )


def fixed_horizon_duration_arpu(
    baseline_mean: float,
    relative_mde: float,
    daily_total_traffic: int,
    baseline_sd: float | None = None,
    baseline_cv: float | None = None,
    n_variants: int = 2,
    alpha: float = 0.05,
    power: float = 0.8,
) -> DurationPlan:
    """Two-sided two-sample z sample size for ARPU, Bonferroni-adjusted for A/B/n.

    Only valid if you look at the data once, at the planned end date.
    """
    _validate_common(daily_total_traffic, n_variants, alpha, power)
    mu1, mu2, delta, sd = _arpu_effect(baseline_mean, relative_mde, baseline_sd, baseline_cv)
    alpha_adj = alpha / (n_variants - 1)

    z_alpha = _z(1 - alpha_adj / 2)
    z_beta = _z(power)
    n_per_variant = int(math.ceil(2.0 * (sd**2) * ((z_alpha + z_beta) ** 2) / (delta**2)))
    daily_per_variant = daily_total_traffic / n_variants

    assumptions: dict[str, Any] = {
        "metric": "arpu",
        "baseline_mean": mu1,
        "baseline_sd": sd,
        "relative_mde": relative_mde,
        "alpha": alpha,
        "power": power,
        "daily_total_traffic": daily_total_traffic,
        "n_variants": n_variants,
    }
    if baseline_cv is not None:
        assumptions["baseline_cv"] = baseline_cv

    return DurationPlan(
        method="fixed_horizon",
        estimated_days=n_per_variant / daily_per_variant,
        n_per_variant=n_per_variant,
        assumptions=assumptions,
        diagnostics={
            "mean_control": mu1,
            "mean_treatment": mu2,
            "baseline_sd": sd,
            "absolute_delta": delta,
            "alpha_per_comparison": alpha_adj,
        },
    )


def sequential_duration_arpu(
    baseline_mean: float,
    relative_mde: float,
    daily_total_traffic: int,
    baseline_sd: float | None = None,
    baseline_cv: float | None = None,
    n_variants: int = 2,
    alpha: float = 0.05,
    power: float = 0.8,
    sequential_tau: float | None = None,
) -> DurationPlan:
    """Sample size for the ARPU mSPRT used by ``ablift analyze --method sequential``.

    Finds the smallest per-variant sample size at which the always-valid test reaches the
    requested power. You may look at the results as often as you like before then.
    """
    _validate_common(daily_total_traffic, n_variants, alpha, power)
    mu1, mu2, delta, sd = _arpu_effect(baseline_mean, relative_mde, baseline_sd, baseline_cv)
    alpha_adj = alpha / (n_variants - 1)
    tau = sequential_tau if sequential_tau is not None else max(abs(mu1) * 0.1, 1e-6)
    if tau <= 0:
        raise ValueError("sequential_tau must be positive.")

    n_per_variant = _search_msprt_n(2.0 * (sd**2), mu2 - mu1, tau, alpha_adj, power)

    fixed = fixed_horizon_duration_arpu(
        baseline_mean,
        relative_mde,
        daily_total_traffic,
        baseline_sd=baseline_sd,
        baseline_cv=baseline_cv,
        n_variants=n_variants,
        alpha=alpha,
        power=power,
    )
    daily_per_variant = daily_total_traffic / n_variants

    assumptions: dict[str, Any] = {
        "metric": "arpu",
        "baseline_mean": mu1,
        "baseline_sd": sd,
        "relative_mde": relative_mde,
        "alpha": alpha,
        "power": power,
        "daily_total_traffic": daily_total_traffic,
        "n_variants": n_variants,
        "sequential_tau": tau,
    }
    if baseline_cv is not None:
        assumptions["baseline_cv"] = baseline_cv

    return DurationPlan(
        method="sequential",
        estimated_days=n_per_variant / daily_per_variant,
        n_per_variant=n_per_variant,
        assumptions=assumptions,
        diagnostics={
            "mean_control": mu1,
            "mean_treatment": mu2,
            "baseline_sd": sd,
            "absolute_delta": delta,
            "alpha_per_comparison": alpha_adj,
            "fixed_horizon_n_per_variant": fixed.n_per_variant,
            "sequential_inflation": n_per_variant / max(fixed.n_per_variant or 1, 1),
            "note": (
                "Upper bound: the test may stop earlier if the true effect is larger than the MDE."
            ),
        },
    )


def bayesian_duration_arpu(
    baseline_mean: float,
    relative_mde: float,
    daily_total_traffic: int,
    baseline_sd: float | None = None,
    baseline_cv: float | None = None,
    n_variants: int = 2,
    prob_threshold: float = 0.95,
    max_expected_loss: float = 0.001,
    assurance_target: float = 0.8,
    max_days: int = 60,
    step_days: int = 1,
    sims: int = 300,
    posterior_draws: int = 2000,
    seed: int = 7,
) -> DurationPlan:
    """Simulation-based Bayesian assurance for ARPU using the Normal-Inverse-Gamma posterior."""
    rng = np.random.default_rng(seed)
    mu_control, mu_treatment, _, sd = _arpu_effect(
        baseline_mean, relative_mde, baseline_sd, baseline_cv
    )

    if daily_total_traffic <= 0:
        raise ValueError("daily_total_traffic must be positive.")
    if n_variants < 2:
        raise ValueError("n_variants must be at least 2 (control plus one treatment).")

    daily_per_variant = daily_total_traffic / n_variants
    candidates = list(range(max(step_days, 1), max_days + 1, max(step_days, 1)))

    var = sd * sd

    def n_on(day: int) -> int:
        return int(max(1, round(day * daily_per_variant)))

    def assurance_at(day: int) -> float:
        n = n_on(day)
        # Exact sufficient statistics for normal data in O(1) regardless of n:
        # x_bar ~ N(mu, sd^2 / n) independently of centered SS ~ sd^2 * chi2(n - 1).
        mean_se = sd / math.sqrt(n)
        c_mean = rng.normal(mu_control, mean_se, size=(sims, 1))
        t_mean = rng.normal(mu_treatment, mean_se, size=(sims, 1))
        c_ss = var * rng.chisquare(n - 1, size=(sims, 1)) if n > 1 else np.zeros((sims, 1))
        t_ss = var * rng.chisquare(n - 1, size=(sims, 1)) if n > 1 else np.zeros((sims, 1))
        shape = (sims, posterior_draws)
        c_draws = sample_mean_posterior_from_moments(n, c_mean, c_ss, shape, rng)
        t_draws = sample_mean_posterior_from_moments(n, t_mean, t_ss, shape, rng)
        return _decision_rate(
            c_draws, t_draws, (t_mean > c_mean)[:, 0], prob_threshold, max_expected_loss
        )

    found = _first_day_reaching(assurance_target, candidates, assurance_at)
    selected_day = float(found[0]) if found else None
    selected_assurance = found[1] if found else None
    selected_n = n_on(found[0]) if found else None

    assumptions: dict[str, Any] = {
        "metric": "arpu",
        "baseline_mean": mu_control,
        "baseline_sd": sd,
        "relative_mde": relative_mde,
        "daily_total_traffic": daily_total_traffic,
        "n_variants": n_variants,
        "prob_threshold": prob_threshold,
        "max_expected_loss": max_expected_loss,
        "assurance_target": assurance_target,
        "max_days": max_days,
        "step_days": step_days,
        "sims": sims,
        "posterior_draws": posterior_draws,
    }
    if baseline_cv is not None:
        assumptions["baseline_cv"] = baseline_cv

    return DurationPlan(
        method="bayesian",
        estimated_days=selected_day,
        n_per_variant=selected_n,
        assumptions=assumptions,
        diagnostics={
            "assurance_at_selected_day": selected_assurance,
            "mean_control": mu_control,
            "mean_treatment": mu_treatment,
            "baseline_sd": sd,
        },
    )
