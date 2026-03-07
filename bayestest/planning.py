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


def frequentist_duration_conversion(
    baseline_rate: float,
    relative_mde: float,
    daily_total_traffic: int,
    n_variants: int = 2,
    alpha: float = 0.05,
    power: float = 0.8,
    max_looks: int = 1,
) -> DurationPlan:
    p1 = baseline_rate
    p2 = baseline_rate * (1 + relative_mde)
    p2 = min(max(p2, 1e-6), 1 - 1e-6)

    delta = abs(p2 - p1)
    if delta <= 0:
        raise ValueError("relative_mde must imply non-zero effect size.")

    p_bar = 0.5 * (p1 + p2)
    z_alpha = _z(1 - alpha / 2)
    z_beta = _z(power)

    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n_per_variant = int(math.ceil(numerator / (delta**2)))

    # Approximate inflation for group-sequential monitoring.
    if max_looks > 1:
        inflation = 1.0 + 0.02 * math.sqrt(max_looks)
        n_per_variant = int(math.ceil(n_per_variant * inflation))
    else:
        inflation = 1.0

    daily_per_variant = daily_total_traffic / max(n_variants, 1)
    if daily_per_variant <= 0:
        raise ValueError("daily_total_traffic and n_variants imply zero traffic per variant.")

    days = n_per_variant / daily_per_variant

    return DurationPlan(
        method="frequentist_sequential" if max_looks > 1 else "frequentist_fixed_horizon",
        estimated_days=days,
        n_per_variant=n_per_variant,
        assumptions={
            "baseline_rate": baseline_rate,
            "relative_mde": relative_mde,
            "alpha": alpha,
            "power": power,
            "daily_total_traffic": daily_total_traffic,
            "n_variants": n_variants,
            "max_looks": max_looks,
        },
        diagnostics={
            "p1": p1,
            "p2": p2,
            "absolute_delta": delta,
            "sequential_inflation": inflation,
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
    p_control = baseline_rate
    p_treatment = min(max(baseline_rate * (1 + relative_mde), 1e-6), 1 - 1e-6)

    if daily_total_traffic <= 0:
        raise ValueError("daily_total_traffic must be positive.")

    daily_per_variant = daily_total_traffic / max(n_variants, 1)
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
