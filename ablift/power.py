"""Power analysis for a fixed sample size: achieved power, or the minimum detectable effect.

``ablift duration`` answers "how many users do I need?". This module answers the other two
questions with the same formulas, so the three always agree:

- given ``n_per_variant`` and ``relative_mde`` -> the power of the test;
- given ``n_per_variant`` and a target ``power`` -> the smallest detectable relative lift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from .planning import _arpu_effect, _effect, _msprt_power_for_variance, _z

_ND = NormalDist()


@dataclass
class PowerResult:
    solved_for: str  # "power" or "relative_mde"
    method: str  # "fixed_horizon" or "sequential"
    power: float
    relative_mde: float
    absolute_mde: float
    n_per_variant: int
    assumptions: dict[str, Any]


def _fixed_power_conversion(n: int, p1: float, p2: float, alpha_adj: float) -> float:
    """Power of the two-sided two-proportion z-test (both rejection tails)."""
    p_bar = 0.5 * (p1 + p2)
    null_term = _z(1 - alpha_adj / 2) * math.sqrt(2 * p_bar * (1 - p_bar))
    shift = abs(p2 - p1) * math.sqrt(n)
    spread = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return _ND.cdf((shift - null_term) / spread) + _ND.cdf((-shift - null_term) / spread)


def _fixed_power_means(n: int, delta: float, sd: float, alpha_adj: float) -> float:
    """Power of the two-sided two-sample z-test for means with equal arm variances."""
    shift = abs(delta) * math.sqrt(n / (2.0 * sd * sd))
    z_alpha = _z(1 - alpha_adj / 2)
    return _ND.cdf(shift - z_alpha) + _ND.cdf(-shift - z_alpha)


def _validate(n_per_variant: int, n_variants: int, alpha: float, power: float | None) -> None:
    if isinstance(n_per_variant, bool) or n_per_variant < 2:
        raise ValueError("n_per_variant must be at least 2.")
    if n_variants < 2:
        raise ValueError("n_variants must be at least 2 (control plus one treatment).")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if power is not None and not 0 < power < 1:
        raise ValueError("power must be in (0, 1).")


def power_analysis(
    *,
    metric: str,
    method: str,
    n_per_variant: int,
    relative_mde: float | None = None,
    power: float | None = None,
    baseline_rate: float | None = None,
    baseline_mean: float | None = None,
    baseline_sd: float | None = None,
    baseline_cv: float | None = None,
    n_variants: int = 2,
    alpha: float = 0.05,
    sequential_tau: float | None = None,
) -> PowerResult:
    """Solve for power (when ``relative_mde`` is given) or for the MDE (when ``power`` is)."""
    if (relative_mde is None) == (power is None):
        raise ValueError("Give exactly one of relative_mde (to get power) or power (to get MDE).")
    if method not in {"fixed", "sequential"}:
        raise ValueError("method must be 'fixed' or 'sequential'.")
    if metric not in {"conversion_rate", "arpu"}:
        raise ValueError("metric must be 'conversion_rate' or 'arpu'.")
    _validate(n_per_variant, n_variants, alpha, power)
    alpha_adj = alpha / (n_variants - 1)

    if metric == "conversion_rate":
        if baseline_rate is None:
            raise ValueError("baseline_rate is required for conversion_rate.")
        baseline = baseline_rate
        tau = sequential_tau if sequential_tau is not None else max(baseline_rate * 0.1, 1e-6)

        def effect(mde: float) -> tuple[float, float]:
            p1, p2, delta = _effect(baseline_rate, mde)
            if method == "fixed":
                return _fixed_power_conversion(n_per_variant, p1, p2, alpha_adj), delta
            v = (p1 * (1 - p1) + p2 * (1 - p2)) / n_per_variant
            return _msprt_power_for_variance(v, p2 - p1, tau, alpha_adj), delta

        # A lift cannot push the treatment rate to 100%.
        max_mde = (1 - 1e-6) / baseline_rate - 1
    else:
        if baseline_mean is None:
            raise ValueError("baseline_mean is required for arpu.")
        baseline = baseline_mean
        tau = sequential_tau if sequential_tau is not None else max(abs(baseline_mean) * 0.1, 1e-6)

        def effect(mde: float) -> tuple[float, float]:
            _, _, delta, sd = _arpu_effect(baseline_mean, mde, baseline_sd, baseline_cv)
            if method == "fixed":
                return _fixed_power_means(n_per_variant, delta, sd, alpha_adj), delta
            v = 2.0 * sd * sd / n_per_variant
            return _msprt_power_for_variance(v, delta, tau, alpha_adj), delta

        max_mde = 1e6

    if tau <= 0:
        raise ValueError("sequential_tau must be positive.")

    if relative_mde is not None:
        achieved, delta = effect(relative_mde)
        solved_for, mde, target = "power", relative_mde, achieved
    else:
        assert power is not None
        if effect(max_mde)[0] < power:
            raise ValueError(
                f"No lift up to {max_mde:+.0%} reaches power {power:.0%} with "
                f"{n_per_variant:,} users per variant; increase the sample size."
            )
        lo, hi = 0.0, max_mde
        for _ in range(200):  # power rises with the lift; bisect to float precision
            mid = 0.5 * (lo + hi)
            if mid in (lo, hi):
                break
            if effect(mid)[0] >= power:
                hi = mid
            else:
                lo = mid
        mde = hi
        achieved, delta = effect(mde)
        solved_for, target = "relative_mde", achieved

    assumptions: dict[str, Any] = {
        "metric": metric,
        "n_per_variant": n_per_variant,
        "n_variants": n_variants,
        "alpha": alpha,
        "alpha_per_comparison": alpha_adj,
    }
    if metric == "conversion_rate":
        assumptions["baseline_rate"] = baseline
    else:
        assumptions["baseline_mean"] = baseline
        assumptions["baseline_sd"] = _arpu_effect(baseline, 1.0, baseline_sd, baseline_cv)[3]
    if method == "sequential":
        assumptions["sequential_tau"] = tau
    if power is not None:
        assumptions["target_power"] = power

    return PowerResult(
        solved_for=solved_for,
        method="fixed_horizon" if method == "fixed" else "sequential",
        power=float(target),
        relative_mde=float(mde),
        absolute_mde=float(delta),
        n_per_variant=n_per_variant,
        assumptions=assumptions,
    )
