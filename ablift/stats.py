from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np


def validate_ratio_aggregates(
    n: int,
    num_sum: float,
    num_sum_sq: float,
    den_sum: float,
    den_sum_sq: float,
    num_den_sum: float,
    *,
    label: str = "variant",
) -> tuple[float, float, float]:
    """Validate ratio sufficient statistics and return centered sums ``(s_yy, s_xx, s_xy)``.

    Enforces ``n >= 2``, ``den_sum > 0``, finite numeric inputs, non-negative second moments,
    and the Cauchy-Schwarz inequality ``s_xy**2 <= s_xx * s_yy`` (up to floating-point tolerance).
    """
    if isinstance(n, bool) or n < 2:
        raise ValueError(f"{label}: ratio metric requires at least 2 visitors, got {n!r}.")

    fields = (
        ("numerator_sum", num_sum),
        ("numerator_sum_squares", num_sum_sq),
        ("denominator_sum", den_sum),
        ("denominator_sum_squares", den_sum_sq),
        ("numerator_denominator_sum", num_den_sum),
    )
    for field_name, value in fields:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label}: {field_name} must be a finite number.")

    if den_sum <= 0:
        raise ValueError(f"{label}: denominator_sum must be positive for ratio metrics.")

    s_yy_raw = num_sum_sq - (num_sum * num_sum) / n
    s_xx_raw = den_sum_sq - (den_sum * den_sum) / n
    s_xy = num_den_sum - (num_sum * den_sum) / n

    tol_y = 1e-9 * max(1.0, abs(num_sum_sq), (num_sum * num_sum) / n)
    tol_x = 1e-9 * max(1.0, abs(den_sum_sq), (den_sum * den_sum) / n)
    if s_yy_raw < -tol_y:
        raise ValueError(
            f"{label}: numerator_sum_squares ({num_sum_sq}) is inconsistent with "
            f"numerator_sum ({num_sum}) and visitors ({n})."
        )
    if s_xx_raw < -tol_x:
        raise ValueError(
            f"{label}: denominator_sum_squares ({den_sum_sq}) is inconsistent with "
            f"denominator_sum ({den_sum}) and visitors ({n})."
        )

    s_yy = max(s_yy_raw, 0.0)
    s_xx = max(s_xx_raw, 0.0)

    cs_tol = 1e-7 * max(1.0, s_xx * s_yy, abs(num_den_sum) ** 2 / n)
    if s_xy * s_xy > s_xx * s_yy + cs_tol:
        raise ValueError(
            f"{label}: numerator_denominator_sum ({num_den_sum}) violates Cauchy-Schwarz "
            "bounds for the given numerator and denominator sums."
        )

    return s_yy, s_xx, s_xy


def ratio_mean_and_var(
    n: int,
    num_sum: float,
    num_sum_sq: float,
    den_sum: float,
    den_sum_sq: float,
    num_den_sum: float,
    *,
    label: str = "variant",
) -> tuple[float, float]:
    """Ratio estimate ``sum(Y) / sum(X)`` and its first-order bivariate Delta-method variance.

    Implements Deng, Knoblich & Lu (2018):
        Var(Y_bar / X_bar) = (s_Y^2 - 2 * R * s_XY + R^2 * s_X^2) / (n * X_bar^2)
    where s_Y^2, s_X^2, s_XY are unbiased sample variances/covariance across ``n`` units.
    """
    s_yy, s_xx, s_xy = validate_ratio_aggregates(
        n,
        num_sum,
        num_sum_sq,
        den_sum,
        den_sum_sq,
        num_den_sum,
        label=label,
    )
    ratio = num_sum / den_sum
    x_bar = den_sum / n
    dof = n - 1
    var_y = s_yy / dof
    var_x = s_xx / dof
    cov_xy = s_xy / dof

    delta_var = (var_y - 2.0 * ratio * cov_xy + (ratio * ratio) * var_x) / (n * x_bar * x_bar)
    return float(ratio), float(max(delta_var, 0.0))


def delta_ratio_z_test(
    control_ratio: float,
    control_var: float,
    treatment_ratio: float,
    treatment_var: float,
) -> tuple[float, float | None, float]:
    """Two-sided asymptotic z-test for a difference of two independent ratio estimators.

    Returns ``(raw_p_value, z_statistic, standard_error)``.
    """
    diff = treatment_ratio - control_ratio
    total_var = max(control_var + treatment_var, 0.0)
    if total_var <= 0.0:
        if diff == 0.0:
            return 1.0, 0.0, 0.0
        return 0.0, None, 0.0
    se = math.sqrt(total_var)
    z_stat = diff / se
    raw_p = 2.0 * (1.0 - NormalDist().cdf(abs(z_stat)))
    return float(raw_p), float(z_stat), float(se)


def winsorize_pooled(
    groups: dict[str, np.ndarray], quantile: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Cap row-level ARPU outcomes at the pooled ``quantile`` across all variants.

    Pooling prevents any treatment arm from shifting its own upper cap (post-treatment
    selection bias). Returns ``(capped_groups, diagnostics)``.
    """
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(quantile)
        or not (0.5 < float(quantile) < 1.0)
    ):
        raise ValueError(
            f"winsorize quantile must be a finite number strictly between 0.5 and 1.0, got {quantile!r}."
        )
    q = float(quantile)
    non_empty = [arr for arr in groups.values() if arr.size > 0]
    if not non_empty:
        raise ValueError("Cannot winsorize empty variant data.")

    pooled = np.concatenate(non_empty)
    cap = float(np.quantile(pooled, q))
    capped_groups: dict[str, np.ndarray] = {}
    per_variant_capped: dict[str, int] = {}
    total_capped = 0
    for name, arr in groups.items():
        capped = np.minimum(arr, cap)
        n_capped = int(np.sum(arr > cap))
        capped_groups[name] = capped
        per_variant_capped[name] = n_capped
        total_capped += n_capped

    pooled_capped = np.concatenate([arr for arr in capped_groups.values() if arr.size > 0])
    raw_var = float(np.var(pooled, ddof=1)) if pooled.size >= 2 else 0.0
    capped_var = float(np.var(pooled_capped, ddof=1)) if pooled_capped.size >= 2 else 0.0
    var_reduction = max(0.0, 1.0 - capped_var / raw_var) if raw_var > 0.0 else 0.0

    diagnostics: dict[str, Any] = {
        "quantile": q,
        "cap_value": cap,
        "capped_count": total_capped,
        "capped_rate": float(total_capped / pooled.size),
        "per_variant_capped": per_variant_capped,
        "variance_reduction_ratio": float(var_reduction),
    }
    return capped_groups, diagnostics


def _sample_skewness_and_kurtosis(values: np.ndarray) -> tuple[float, float]:
    """Return Fisher-Pearson skewness g1 and excess kurtosis g2 (0.0 when degenerate)."""
    if values.size < 3:
        return 0.0, 0.0
    centered = values - float(np.mean(values))
    m2 = float(np.mean(centered * centered))
    if m2 <= 1e-24:
        return 0.0, 0.0
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = m3 / (m2**1.5)
    excess_kurtosis = (m4 / (m2 * m2)) - 3.0
    return float(skew), float(excess_kurtosis)


def row_level_tail_diagnostics(groups: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute skewness, max/mean ratio, and Kohavi CLT adequacy diagnostics for row-level ARPU.

    Flags ``heavy_tailed=True`` when any arm (or pooled data) has ``|skewness| > 2.0`` and
    sample size below Kohavi's CLT threshold ``355 * skewness^2``, or when ``max(|y|) / |mean| > 20``.
    """
    non_empty = {name: arr for name, arr in groups.items() if arr.size > 0}
    if not non_empty:
        return {
            "source": "row_level",
            "heavy_tailed": False,
            "pooled_skewness": 0.0,
            "pooled_excess_kurtosis": 0.0,
            "max_arm_skewness": 0.0,
            "max_to_mean_ratio": 0.0,
            "kohavi_min_n": 0.0,
            "min_arm_n": 0,
        }

    pooled = np.concatenate(list(non_empty.values()))
    pooled_skew, pooled_kurt = _sample_skewness_and_kurtosis(pooled)

    max_abs_skew = abs(pooled_skew)
    max_to_mean = 0.0
    heavy_by_clt = abs(pooled_skew) > 2.0 and pooled.size < 355.0 * (pooled_skew**2)
    min_arm_n = min(int(arr.size) for arr in non_empty.values())

    for arr in non_empty.values():
        arm_skew, _ = _sample_skewness_and_kurtosis(arr)
        max_abs_skew = max(max_abs_skew, abs(arm_skew))
        if abs(arm_skew) > 2.0 and arr.size < 355.0 * (arm_skew**2):
            heavy_by_clt = True
        mean_abs = abs(float(np.mean(arr)))
        if mean_abs > 1e-12:
            ratio = float(np.max(np.abs(arr))) / mean_abs
            max_to_mean = max(max_to_mean, ratio)

    heavy_tailed = bool(heavy_by_clt or max_to_mean > 20.0)
    return {
        "source": "row_level",
        "heavy_tailed": heavy_tailed,
        "pooled_skewness": float(pooled_skew),
        "pooled_excess_kurtosis": float(pooled_kurt),
        "max_arm_skewness": float(max_abs_skew),
        "max_to_mean_ratio": float(max_to_mean),
        "kohavi_min_n": float(355.0 * (max_abs_skew**2)),
        "min_arm_n": min_arm_n,
    }


def aggregated_arpu_tail_diagnostics(
    variants: list[tuple[str, int, float, float]],
) -> dict[str, Any]:
    """Estimate heavy-tail risk from aggregated ARPU moments ``(name, n, sum, sum_sq)``.

    Flags ``heavy_tailed=True`` when any arm has sample coefficient of variation ``sd / |mean| > 3.0``.
    """
    per_variant_cv: dict[str, float] = {}
    for name, n, rev_sum, rev_sum_sq in variants:
        if n < 2:
            per_variant_cv[name] = 0.0
            continue
        mean = rev_sum / n
        if abs(mean) <= 1e-12:
            per_variant_cv[name] = 0.0
            continue
        centered = max(rev_sum_sq - (rev_sum * rev_sum) / n, 0.0)
        sd = math.sqrt(centered / (n - 1))
        per_variant_cv[name] = float(sd / abs(mean))

    max_cv = max(per_variant_cv.values(), default=0.0)
    return {
        "source": "aggregated",
        "heavy_tailed": bool(max_cv > 3.0),
        "max_cv": float(max_cv),
        "per_variant_cv": per_variant_cv,
    }


def bootstrap_mean_diff_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    n_bootstrap: int,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Chunked vectorized percentile bootstrap CI for ``mean(treatment) - mean(control)``."""
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError(f"bootstrap_samples must be an integer >= 100, got {n_bootstrap!r}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    if control.size < 2 or treatment.size < 2:
        raise ValueError("Bootstrap CI requires at least 2 observations per arm.")

    rng = np.random.default_rng(seed)
    n_c = control.size
    n_t = treatment.size
    diffs = np.empty(n_bootstrap, dtype=float)
    # Keep peak temporary allocation bounded (~8 MB max per chunk) even on large row tables.
    max_elems = 1_000_000
    chunk_size = max(1, min(n_bootstrap, max_elems // max(n_c, n_t)))

    offset = 0
    while offset < n_bootstrap:
        step = min(chunk_size, n_bootstrap - offset)
        c_idx = rng.integers(0, n_c, size=(step, n_c))
        t_idx = rng.integers(0, n_t, size=(step, n_t))
        diffs[offset : offset + step] = treatment[t_idx].mean(axis=1) - control[c_idx].mean(axis=1)
        offset += step

    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)
