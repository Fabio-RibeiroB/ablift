from __future__ import annotations

import math
from statistics import NormalDist


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
