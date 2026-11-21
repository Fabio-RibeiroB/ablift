from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


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


def _validate_common(daily_total_traffic: int, n_variants: int, alpha: float, power: float):
    if daily_total_traffic <= 0:
        raise ValueError("daily_total_traffic must be positive.")
    if n_variants < 2:
        raise ValueError("n_variants must be at least 2 (control plus one treatment).")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1).")


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


def _msprt_power(n: int, p1: float, p2: float, tau: float, alpha_adj: float) -> float:
    """Probability the mSPRT has rejected by the time each arm has ``n`` units.

    Evaluated at the final sample size only, which lower-bounds the probability of having
    crossed at any earlier look, so the resulting duration is slightly conservative.
    """
    v = (p1 * (1 - p1) + p2 * (1 - p2)) / n
    t2 = tau * tau
    # Lambda(delta_hat) >= 1/alpha  <=>  |delta_hat| >= c
    c = math.sqrt(2 * v * (v + t2) / t2 * (math.log(1 / alpha_adj) + 0.5 * math.log((v + t2) / v)))
    delta = p2 - p1
    sd = math.sqrt(v)
    nd = NormalDist()
    return (1 - nd.cdf((c - delta) / sd)) + nd.cdf((-c - delta) / sd)


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

    lo, hi = 2, 2
    while _msprt_power(hi, p1, p2, tau, alpha_adj) < power:
        hi *= 2
        if hi > 10**12:
            raise ValueError("Effect is too small to detect with any realistic sample size.")
    while lo < hi:
        mid = (lo + hi) // 2
        if _msprt_power(mid, p1, p2, tau, alpha_adj) >= power:
            hi = mid
        else:
            lo = mid + 1
    n_per_variant = lo

    fixed = fixed_horizon_duration_conversion(
        baseline_rate, relative_mde, daily_total_traffic, n_variants, alpha, power
    )
    daily_per_variant = daily_total_traffic / n_variants
    return DurationPlan(
        method="sequential",
        estimated_days=n_per_variant / daily_per_variant,
        n_per_variant=n_per_variant,
        assumptions={
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

    selected_day = None
    selected_assurance = None
    selected_n = None

    for day in candidates:
        n = int(max(1, round(day * daily_per_variant)))
        wins = 0

        for _ in range(sims):
            c = rng.binomial(n, p_control)
            t = rng.binomial(n, p_treatment)

            c_samps = rng.beta(1 + c, 1 + (n - c), size=posterior_draws)
            t_samps = rng.beta(1 + t, 1 + (n - t), size=posterior_draws)

            p_win = float(np.mean(t_samps > c_samps))
            expected_loss = float(np.mean(np.maximum(c_samps - t_samps, 0.0)))

            if p_win >= prob_threshold and expected_loss <= max_expected_loss and t > c:
                wins += 1

        assurance = wins / sims
        if assurance >= assurance_target:
            selected_day = float(day)
            selected_assurance = assurance
            selected_n = n
            break

    return DurationPlan(
        method="bayesian",
        estimated_days=selected_day,
        n_per_variant=selected_n,
        assumptions={
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
