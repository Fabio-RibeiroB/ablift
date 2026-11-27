"""Tests for unequal traffic splits (--expected-split) in duration planning and power analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
from click.testing import CliRunner
from openpyxl import Workbook
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest
from test_schema import validate_json_schema

from ablift.cli import cli
from ablift.engine import msprt_statistic, two_proportion_test
from ablift.planning import (
    bayesian_duration_arpu,
    bayesian_duration_conversion,
    fixed_horizon_duration_arpu,
    fixed_horizon_duration_conversion,
    sequential_duration_arpu,
    sequential_duration_conversion,
)
from ablift.power import power_analysis
from ablift.schemas import get_schema


def _independent_fixed_two_proportion_n(
    p1: float,
    p2: float,
    split: list[float],
    alpha: float = 0.05,
    power: float = 0.8,
) -> tuple[list[int], int]:
    """Independent textbook unequal-allocation two-proportion z sample size formula."""
    total_w = sum(split)
    props = [w / total_w for w in split]
    p_c = props[0]
    nd = NormalDist()
    alpha_adj = alpha / (len(props) - 1)
    z_a = nd.inv_cdf(1.0 - alpha_adj / 2.0)
    z_b = nd.inv_cdf(power)
    delta = abs(p2 - p1)

    n_total_cont = 0.0
    for p_j in props[1:]:
        p_bar = (p_c * p1 + p_j * p2) / (p_c + p_j)
        term_null = z_a * math.sqrt(p_bar * (1.0 - p_bar) * (1.0 / p_c + 1.0 / p_j))
        term_alt = z_b * math.sqrt(p1 * (1.0 - p1) / p_c + p2 * (1.0 - p2) / p_j)
        n_total_cont = max(n_total_cont, ((term_null + term_alt) ** 2) / (delta**2))

    n_by_variant = [math.ceil(n_total_cont * p_i) for p_i in props]
    return n_by_variant, sum(n_by_variant)


def test_fixed_conversion_90_10_ratio_and_independent_formula() -> None:
    """90/10 needs ~2.78x total traffic vs 50/50 and matches independent z formula."""
    # Equal-variance regime around p=0.5 makes 1/0.9 + 1/0.1 = 11.111 vs 4.0 -> 2.778x exact.
    equal_mid = fixed_horizon_duration_conversion(0.5, 0.04, 50_000)
    split_mid = fixed_horizon_duration_conversion(0.5, 0.04, 50_000, expected_split=[0.9, 0.1])
    assert equal_mid.n_total is not None and split_mid.n_total is not None
    assert split_mid.n_total / equal_mid.n_total == pytest.approx(25 / 9, rel=0.01)

    # Practical conversion rate (4% -> 4.2%): check against independent closed-form formula.
    for baseline, mde, split in (
        (0.04, 0.05, [0.9, 0.1]),
        (0.04, 0.05, [90, 10]),
        (0.08, -0.10, [0.8, 0.2]),
        (0.05, 0.10, [0.8, 0.1, 0.1]),
        (0.05, 0.10, [0.6, 0.3, 0.1]),
    ):
        plan = fixed_horizon_duration_conversion(
            baseline_rate=baseline,
            relative_mde=mde,
            daily_total_traffic=50_000,
            n_variants=len(split),
            expected_split=split,
        )
        p2 = baseline * (1.0 + mde)
        exp_by_var, exp_total = _independent_fixed_two_proportion_n(baseline, p2, split)
        assert plan.n_by_variant == exp_by_var
        assert plan.n_total == exp_total
        assert plan.n_per_variant == exp_by_var[0]
        assert plan.estimated_days == pytest.approx(exp_total / 50_000)

    # Typical 4% -> 4.2% 90/10 vs 50/50 ratio is also ~2.78x (slightly above due to p2 > p1).
    equal_4pct = fixed_horizon_duration_conversion(0.04, 0.05, 50_000)
    split_4pct = fixed_horizon_duration_conversion(0.04, 0.05, 50_000, expected_split=[0.9, 0.1])
    assert equal_4pct.n_total is not None and split_4pct.n_total is not None
    assert split_4pct.n_total / equal_4pct.n_total == pytest.approx(2.78, rel=0.02)


def test_multi_arm_asymmetric_split_sizes_for_worst_arm() -> None:
    """With [0.6, 0.3, 0.1], every treatment (including the 10% arm) reaches target power."""
    plan = fixed_horizon_duration_conversion(
        0.04, 0.10, 50_000, n_variants=3, expected_split=[0.6, 0.3, 0.1]
    )
    assert plan.n_by_variant is not None
    assert plan.n_total is not None
    # Control gets 60%, v1 gets 30%, v2 gets 10%.
    assert plan.n_by_variant[0] > plan.n_by_variant[1] > plan.n_by_variant[2]
    assert plan.n_by_variant[0] / plan.n_total == pytest.approx(0.6, abs=1e-3)
    assert plan.n_by_variant[2] / plan.n_total == pytest.approx(0.1, abs=1e-3)

    # Worst-arm power check via power_analysis confirms >= 80% for the 10% arm.
    achieved = power_analysis(
        metric="conversion_rate",
        method="fixed",
        n_total=plan.n_total,
        n_variants=3,
        expected_split=[0.6, 0.3, 0.1],
        baseline_rate=0.04,
        relative_mde=0.10,
    )
    # The closed-form plan lands on the target to ~1e-5; the exact power and
    # statistics.NormalDist differ at that level across Python versions.
    assert achieved.power >= 0.8 - 1e-4


@pytest.mark.parametrize(
    ("mean", "sd", "mde", "split", "alpha", "power"),
    [
        (12.5, 25.0, 0.05, [0.9, 0.1], 0.05, 0.8),
        (12.5, 25.0, 0.05, [90, 10], 0.05, 0.8),
        (20.0, 40.0, 0.10, [0.8, 0.2], 0.01, 0.9),
        (15.0, 30.0, -0.08, [0.8, 0.1, 0.1], 0.05, 0.8),
        (10.0, 20.0, 0.10, [0.6, 0.3, 0.1], 0.05, 0.85),
    ],
)
def test_arpu_fixed_matches_statsmodels_normal_ind_power(
    mean: float, sd: float, mde: float, split: list[float], alpha: float, power: float
) -> None:
    """statsmodels NormalIndPower().solve_power(..., ratio=...) agrees with ARPU fixed n per arm."""
    plan = fixed_horizon_duration_arpu(
        baseline_mean=mean,
        baseline_sd=sd,
        relative_mde=mde,
        daily_total_traffic=50_000,
        n_variants=len(split),
        alpha=alpha,
        power=power,
        expected_split=split,
    )
    assert plan.n_by_variant is not None
    assert plan.n_total is not None
    assert plan.n_per_variant == plan.n_by_variant[0]
    assert plan.n_total == sum(plan.n_by_variant)
    assert plan.estimated_days == pytest.approx(plan.n_total / 50_000)

    total_w = sum(split)
    props = [w / total_w for w in split]
    ratio_worst = min(props[1:]) / props[0]
    delta = abs(mean * mde)
    alpha_adj = alpha / (len(split) - 1)
    n_control_sm = float(
        NormalIndPower().solve_power(
            effect_size=delta / sd,
            alpha=alpha_adj,
            power=power,
            ratio=ratio_worst,
            alternative="two-sided",
        )
    )
    n_worst_sm = n_control_sm * ratio_worst

    # Every arm gets at least the statsmodels requirement; topping up the total so the
    # smallest arm is covered can add a few users per arm, never more than one per variant.
    slack = len(split) + 1
    assert math.floor(n_control_sm) <= plan.n_by_variant[0] <= math.ceil(n_control_sm) + slack
    worst = min(plan.n_by_variant[1:])
    assert math.floor(n_worst_sm) <= worst <= math.ceil(n_worst_sm) + slack

    # Check exact 25/9 = 2.778x scaling for 90/10 vs 50/50.
    if split == [0.9, 0.1]:
        equal = fixed_horizon_duration_arpu(
            baseline_mean=mean,
            baseline_sd=sd,
            relative_mde=mde,
            daily_total_traffic=50_000,
            alpha=alpha,
            power=power,
        )
        assert equal.n_total is not None
        assert plan.n_total / equal.n_total == pytest.approx(25 / 9, rel=1e-3)


def test_sequential_conversion_with_split_achieves_target_power() -> None:
    """Sequential plan with a 90/10 split achieves target power in Monte Carlo simulation."""
    baseline, mde, power = 0.05, 0.2, 0.8
    plan = sequential_duration_conversion(
        baseline, mde, 10_000, power=power, expected_split=[0.9, 0.1]
    )
    assert plan.n_by_variant is not None
    assert plan.n_total is not None
    n_c, n_t = plan.n_by_variant
    tau = plan.assumptions["sequential_tau"]
    assert plan.assumptions["expected_split"] == [0.9, 0.1]

    rng = np.random.default_rng(1)
    sims = 2000
    c = rng.binomial(n_c, baseline, size=sims)
    t = rng.binomial(n_t, baseline * (1 + mde), size=sims)
    hits = 0
    for ci, ti in zip(c, t, strict=True):
        _, _, se = two_proportion_test(int(ci), n_c, int(ti), n_t)
        hits += msprt_statistic(ti / n_t - ci / n_c, se, tau) >= 20
    assert hits / sims == pytest.approx(power, abs=0.04)

    equal = sequential_duration_conversion(baseline, mde, 10_000, power=power)
    assert equal.n_total is not None
    assert plan.n_total / equal.n_total > 2.5
    assert plan.n_per_variant == n_c
    assert plan.n_per_variant > plan.diagnostics["fixed_horizon_n_per_variant"]


def test_sequential_arpu_with_split_achieves_target_power() -> None:
    """Sequential ARPU plan with an 80/20 split achieves target power in Monte Carlo simulation."""
    baseline_mean, baseline_sd, mde, power = 10.0, 20.0, 0.15, 0.8
    plan = sequential_duration_arpu(
        baseline_mean=baseline_mean,
        baseline_sd=baseline_sd,
        relative_mde=mde,
        daily_total_traffic=10_000,
        power=power,
        expected_split=[0.8, 0.2],
    )
    assert plan.n_by_variant is not None
    n_c, n_t = plan.n_by_variant
    tau = plan.assumptions["sequential_tau"]

    rng = np.random.default_rng(2)
    sims = 2000
    mu_t = baseline_mean * (1.0 + mde)
    c_means = rng.normal(baseline_mean, baseline_sd / math.sqrt(n_c), size=sims)
    t_means = rng.normal(mu_t, baseline_sd / math.sqrt(n_t), size=sims)
    c_vars = (baseline_sd**2) * rng.chisquare(n_c - 1, size=sims) / (n_c - 1)
    t_vars = (baseline_sd**2) * rng.chisquare(n_t - 1, size=sims) / (n_t - 1)

    hits = 0
    for cm, tm, cv_i, tv_i in zip(c_means, t_means, c_vars, t_vars, strict=True):
        se = math.sqrt(float(cv_i) / n_c + float(tv_i) / n_t)
        hits += msprt_statistic(float(tm - cm), se, tau) >= 20
    assert hits / sims == pytest.approx(power, abs=0.04)


def test_bayesian_with_split_simulates_unequal_arms() -> None:
    """Bayesian conversion and ARPU plans simulate unequal arm sizes and take longer at 90/10."""
    eq_conv = bayesian_duration_conversion(
        baseline_rate=0.05,
        relative_mde=0.15,
        daily_total_traffic=40_000,
        max_days=60,
        sims=200,
        posterior_draws=1000,
        seed=11,
    )
    sp_conv = bayesian_duration_conversion(
        baseline_rate=0.05,
        relative_mde=0.15,
        daily_total_traffic=40_000,
        max_days=60,
        sims=200,
        posterior_draws=1000,
        seed=11,
        expected_split=[0.9, 0.1],
    )
    assert eq_conv.estimated_days is not None and sp_conv.estimated_days is not None
    assert sp_conv.estimated_days > eq_conv.estimated_days
    assert sp_conv.n_by_variant is not None and sp_conv.n_total is not None
    assert sp_conv.n_per_variant == sp_conv.n_by_variant[0]
    assert sp_conv.n_total == sum(sp_conv.n_by_variant)
    assert sp_conv.n_by_variant[0] / sp_conv.n_total == pytest.approx(0.9, abs=1e-3)
    assert sp_conv.n_by_variant[1] / sp_conv.n_total == pytest.approx(0.1, abs=1e-3)
    assert sp_conv.diagnostics["assurance_at_selected_day"] >= 0.8

    eq_arpu = bayesian_duration_arpu(
        baseline_mean=10.0,
        baseline_sd=15.0,
        relative_mde=0.10,
        daily_total_traffic=1_000,
        max_expected_loss=0.05,
        max_days=60,
        sims=150,
        posterior_draws=800,
        seed=11,
    )
    sp_arpu = bayesian_duration_arpu(
        baseline_mean=10.0,
        baseline_sd=15.0,
        relative_mde=0.10,
        daily_total_traffic=1_000,
        max_expected_loss=0.05,
        max_days=60,
        sims=150,
        posterior_draws=800,
        seed=11,
        expected_split=[0.9, 0.1],
    )
    assert eq_arpu.estimated_days is not None and sp_arpu.estimated_days is not None
    assert sp_arpu.estimated_days > eq_arpu.estimated_days
    assert sp_arpu.n_by_variant is not None and sp_arpu.n_total is not None
    assert sp_arpu.n_by_variant[0] / sp_arpu.n_total == pytest.approx(0.9, abs=1e-3)

    # Unreachable Bayesian plan returns None across all sample-size fields.
    none_plan = bayesian_duration_conversion(
        baseline_rate=0.04,
        relative_mde=0.01,
        daily_total_traffic=500,
        max_days=2,
        sims=30,
        posterior_draws=200,
        expected_split=[0.9, 0.1],
    )
    assert none_plan.estimated_days is None
    assert none_plan.n_per_variant is None
    assert none_plan.n_by_variant is None
    assert none_plan.n_total is None


def test_no_split_output_is_unchanged() -> None:
    """Omitting expected_split keeps existing values and matches an explicit equal split."""
    fixed_conv = fixed_horizon_duration_conversion(0.04, 0.05, 50_000, n_variants=2)
    fixed_conv_eq = fixed_horizon_duration_conversion(
        0.04, 0.05, 50_000, n_variants=2, expected_split=[0.5, 0.5]
    )
    assert "expected_split" not in fixed_conv.assumptions
    assert fixed_conv_eq.assumptions["expected_split"] == [0.5, 0.5]
    assert fixed_conv.n_per_variant == fixed_conv_eq.n_per_variant
    assert fixed_conv.n_by_variant == [fixed_conv.n_per_variant, fixed_conv.n_per_variant]
    assert fixed_conv.n_total == fixed_conv.n_per_variant * 2
    assert fixed_conv.estimated_days == pytest.approx(fixed_conv_eq.estimated_days)

    seq_conv = sequential_duration_conversion(0.04, 0.05, 50_000, n_variants=3)
    seq_conv_eq = sequential_duration_conversion(
        0.04, 0.05, 50_000, n_variants=3, expected_split=[1, 1, 1]
    )
    assert "expected_split" not in seq_conv.assumptions
    assert seq_conv.n_per_variant == seq_conv_eq.n_per_variant
    assert seq_conv.n_by_variant == seq_conv_eq.n_by_variant
    assert seq_conv.n_total == seq_conv_eq.n_total

    fixed_arpu = fixed_horizon_duration_arpu(12.5, 0.05, 50_000, baseline_sd=25.0)
    fixed_arpu_eq = fixed_horizon_duration_arpu(
        12.5, 0.05, 50_000, baseline_sd=25.0, expected_split=[50, 50]
    )
    assert "expected_split" not in fixed_arpu.assumptions
    assert fixed_arpu.n_per_variant == fixed_arpu_eq.n_per_variant
    assert fixed_arpu.n_total == fixed_arpu_eq.n_total

    seq_arpu = sequential_duration_arpu(12.5, 0.05, 50_000, baseline_sd=25.0)
    seq_arpu_eq = sequential_duration_arpu(
        12.5, 0.05, 50_000, baseline_sd=25.0, expected_split=[0.5, 0.5]
    )
    assert "expected_split" not in seq_arpu.assumptions
    assert seq_arpu.n_per_variant == seq_arpu_eq.n_per_variant

    bayes_conv = bayesian_duration_conversion(
        0.04, 0.08, 60_000, max_days=20, sims=80, posterior_draws=400, seed=7
    )
    bayes_conv_eq = bayesian_duration_conversion(
        0.04,
        0.08,
        60_000,
        max_days=20,
        sims=80,
        posterior_draws=400,
        seed=7,
        expected_split=[0.5, 0.5],
    )
    assert "expected_split" not in bayes_conv.assumptions
    assert bayes_conv.estimated_days == bayes_conv_eq.estimated_days
    assert bayes_conv.n_per_variant == bayes_conv_eq.n_per_variant
    assert (
        bayes_conv.diagnostics["assurance_at_selected_day"]
        == bayes_conv_eq.diagnostics["assurance_at_selected_day"]
    )

    # Power without expected_split matches --n-total with equal split.
    pwr_per = power_analysis(
        metric="conversion_rate",
        method="fixed",
        n_per_variant=10_000,
        relative_mde=0.1,
        baseline_rate=0.08,
    )
    pwr_tot = power_analysis(
        metric="conversion_rate",
        method="fixed",
        n_total=20_000,
        relative_mde=0.1,
        baseline_rate=0.08,
    )
    assert "expected_split" not in pwr_per.assumptions
    assert pwr_per.power == pytest.approx(pwr_tot.power, abs=1e-12)
    assert pwr_per.n_by_variant == [10_000, 10_000]
    assert pwr_per.n_total == 20_000


def test_power_with_split_matches_statsmodels_and_simulation() -> None:
    """Power with unequal split matches statsmodels for ARPU and simulated z-test for conversion."""
    res_arpu = power_analysis(
        metric="arpu",
        method="fixed",
        n_total=10_000,
        expected_split=[0.9, 0.1],
        relative_mde=0.1,
        baseline_mean=100.0,
        baseline_sd=100.0,
    )
    assert res_arpu.n_by_variant == [9_000, 1_000]
    assert res_arpu.n_per_variant == 9_000
    assert res_arpu.n_total == 10_000
    expected = NormalIndPower().power(effect_size=0.1, nobs1=9_000, alpha=0.05, ratio=1 / 9)
    assert res_arpu.power == pytest.approx(expected, abs=1e-9)

    # Conversion power with 90/10 split vs simulated proportions_ztest.
    res_conv = power_analysis(
        metric="conversion_rate",
        method="fixed",
        n_total=20_000,
        expected_split=[0.9, 0.1],
        relative_mde=0.2,
        baseline_rate=0.08,
    )
    n_c, n_t = res_conv.n_by_variant
    assert (n_c, n_t) == (18_000, 2_000)
    rng = np.random.default_rng(17)
    sims = 3_000
    x_c = rng.binomial(n_c, 0.08, size=sims)
    x_t = rng.binomial(n_t, 0.08 * 1.2, size=sims)
    rejections = sum(
        proportions_ztest([b, a], [n_t, n_c])[1] <= 0.05 for a, b in zip(x_c, x_t, strict=True)
    )
    assert res_conv.power == pytest.approx(rejections / sims, abs=0.025)


@pytest.mark.parametrize("split", [[0.9, 0.1], [0.7, 0.2, 0.1]])
@pytest.mark.parametrize(
    ("plan_fn", "kwargs", "method", "metric"),
    [
        (fixed_horizon_duration_conversion, {"baseline_rate": 0.08}, "fixed", "conversion_rate"),
        (sequential_duration_conversion, {"baseline_rate": 0.04}, "sequential", "conversion_rate"),
        (
            fixed_horizon_duration_arpu,
            {"baseline_mean": 12.5, "baseline_sd": 25.0},
            "fixed",
            "arpu",
        ),
        (
            sequential_duration_arpu,
            {"baseline_mean": 12.5, "baseline_sd": 25.0},
            "sequential",
            "arpu",
        ),
    ],
)
def test_duration_and_power_round_trip_with_split(
    split: list[float], plan_fn, kwargs: dict, method: str, metric: str
) -> None:
    """duration's n_total with expected_split round-trips cleanly through power."""
    plan = plan_fn(
        relative_mde=0.1,
        daily_total_traffic=20_000,
        n_variants=len(split),
        expected_split=split,
        **kwargs,
    )
    assert plan.n_total is not None
    achieved = power_analysis(
        metric=metric,
        method=method,
        n_total=plan.n_total,
        n_variants=len(split),
        expected_split=split,
        relative_mde=0.1,
        **kwargs,
    )
    # power splits n_total exactly as duration did, so every arm gets its planned size and
    # reaches the target (to the ~1e-5 gap between the closed form and exact power).
    assert achieved.n_by_variant == plan.n_by_variant
    assert achieved.power >= 0.8 - 1e-4

    mde_res = power_analysis(
        metric=metric,
        method=method,
        n_total=plan.n_total,
        n_variants=len(split),
        expected_split=split,
        power=0.8,
        **kwargs,
    )
    assert mde_res.relative_mde == pytest.approx(0.1, rel=0.01)
    assert mde_res.power == pytest.approx(0.8, abs=1e-6)


@pytest.mark.parametrize(
    "bad_split",
    [
        "",
        "0.9",
        "0.8,0.1,0.1",
        "0.9,0",
        "0.9,-0.1",
        "0.9,abc",
        "0.9,,0.1",
        "nan,0.1",
        "inf,0.1",
        "control=0.9,treatment=0.1",
        "1e308,1e308",
        "1e300,1e-300",
    ],
)
def test_cli_bad_splits_exit_code_2(bad_split: str) -> None:
    """Bad --expected-split values are clean usage errors (exit code 2) for duration and power."""
    runner = CliRunner()
    dur = runner.invoke(
        cli,
        [
            "duration",
            "--method",
            "fixed",
            "--baseline-rate",
            "0.04",
            "--relative-mde",
            "0.05",
            "--daily-traffic",
            "50000",
            "--expected-split",
            bad_split,
        ],
    )
    assert dur.exit_code == 2, f"duration({bad_split!r}): {dur.output}"

    pwr = runner.invoke(
        cli,
        [
            "power",
            "--baseline-rate",
            "0.08",
            "--n-total",
            "20000",
            "--relative-mde",
            "0.1",
            "--expected-split",
            bad_split,
        ],
    )
    assert pwr.exit_code == 2, f"power({bad_split!r}): {pwr.output}"


def test_cli_power_flag_usage_errors() -> None:
    """Power rejects --n-per-variant with --expected-split, both size flags, or neither."""
    runner = CliRunner()
    for args in (
        ["power", "--baseline-rate", "0.08", "--relative-mde", "0.1"],
        [
            "power",
            "--baseline-rate",
            "0.08",
            "--n-per-variant",
            "10000",
            "--n-total",
            "20000",
            "--relative-mde",
            "0.1",
        ],
        [
            "power",
            "--baseline-rate",
            "0.08",
            "--n-per-variant",
            "10000",
            "--expected-split",
            "0.9,0.1",
            "--relative-mde",
            "0.1",
        ],
    ):
        res = runner.invoke(cli, args)
        assert res.exit_code == 2, f"{args}: {res.output}"


def test_cli_duration_input_files_with_expected_split(tmp_path: Path) -> None:
    """Duration accepts expected_split in JSON, CSV, and XLSX inputs and lets CLI override."""
    runner = CliRunner()
    in_schema = get_schema("duration-input")
    out_schema = get_schema("duration-output")

    # JSON input with array expected_split
    json_payload = {
        "metric": "conversion_rate",
        "method": "fixed",
        "baseline_rate": 0.04,
        "relative_mde": 0.05,
        "daily_traffic": 50_000,
        "n_variants": 3,
        "expected_split": [80, 10, 10],
    }
    assert validate_json_schema(json_payload, in_schema) == []
    json_res = runner.invoke(cli, ["duration", "--input", "-"], input=json.dumps(json_payload))
    assert json_res.exit_code == 0, json_res.output
    json_out = json.loads(json_res.output)
    assert validate_json_schema(json_out, out_schema) == []
    assert json_out["assumptions"]["expected_split"] == [0.8, 0.1, 0.1]

    # CSV input with quoted comma-separated expected_split
    csv_path = tmp_path / "split.csv"
    csv_path.write_text(
        "method,baseline_rate,relative_mde,daily_traffic,n_variants,expected_split\n"
        'fixed,0.04,0.05,50000,3,"80,10,10"\n',
        encoding="utf-8",
    )
    csv_res = runner.invoke(cli, ["duration", "--input", str(csv_path)])
    assert csv_res.exit_code == 0, csv_res.output
    csv_out = json.loads(csv_res.output)
    assert csv_out == json_out

    # XLSX input with comma-separated expected_split
    xlsx_path = tmp_path / "split.xlsx"
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(
        ["method", "baseline_rate", "relative_mde", "daily_traffic", "n_variants", "expected_split"]
    )
    ws.append(["fixed", 0.04, 0.05, 50_000, 3, "80,10,10"])
    wb.save(xlsx_path)
    xlsx_res = runner.invoke(cli, ["duration", "--input", str(xlsx_path)])
    assert xlsx_res.exit_code == 0, xlsx_res.output
    assert json.loads(xlsx_res.output) == json_out

    # CLI --expected-split overrides input file's split
    override_res = runner.invoke(
        cli,
        ["duration", "--input", str(csv_path), "--n-variants", "2", "--expected-split", "0.9,0.1"],
    )
    assert override_res.exit_code == 0, override_res.output
    override_out = json.loads(override_res.output)
    assert override_out["assumptions"]["expected_split"] == [0.9, 0.1]
    assert len(override_out["n_by_variant"]) == 2

    # Bad expected_split in JSON / CSV input exits with code 2
    bad_json = json.dumps(
        {
            "baseline_rate": 0.04,
            "relative_mde": 0.05,
            "daily_traffic": 50_000,
            "expected_split": [0.9, -0.1],
        }
    )
    bad_res = runner.invoke(cli, ["duration", "--input", "-"], input=bad_json)
    assert bad_res.exit_code == 2, bad_res.output

    wrong_len_json = json.dumps(
        {
            "baseline_rate": 0.04,
            "relative_mde": 0.05,
            "daily_traffic": 50_000,
            "n_variants": 2,
            "expected_split": [0.8, 0.1, 0.1],
        }
    )
    wrong_len_res = runner.invoke(cli, ["duration", "--input", "-"], input=wrong_len_json)
    assert wrong_len_res.exit_code == 2, wrong_len_res.output


def test_cli_outputs_validate_against_schemas() -> None:
    """Duration and power outputs with unequal splits validate against their JSON Schemas."""
    runner = CliRunner()
    dur_schema = get_schema("duration-output")
    pwr_schema = get_schema("power-output")

    dur_cases = [
        [
            "duration",
            "--method",
            "fixed",
            "--baseline-rate",
            "0.04",
            "--relative-mde",
            "0.05",
            "--daily-traffic",
            "50000",
            "--expected-split",
            "0.9,0.1",
        ],
        [
            "duration",
            "--method",
            "sequential",
            "--baseline-rate",
            "0.04",
            "--relative-mde",
            "0.05",
            "--daily-traffic",
            "50000",
            "--n-variants",
            "3",
            "--expected-split",
            "80,10,10",
        ],
        [
            "duration",
            "--method",
            "bayesian",
            "--baseline-rate",
            "0.05",
            "--relative-mde",
            "0.15",
            "--daily-traffic",
            "40000",
            "--max-days",
            "10",
            "--expected-split",
            "0.9,0.1",
        ],
        [
            "duration",
            "--metric",
            "arpu",
            "--method",
            "fixed",
            "--baseline-mean",
            "12.5",
            "--baseline-sd",
            "25.0",
            "--relative-mde",
            "0.05",
            "--daily-traffic",
            "50000",
            "--expected-split",
            "0.9,0.1",
        ],
        [
            "duration",
            "--metric",
            "arpu",
            "--method",
            "sequential",
            "--baseline-mean",
            "12.5",
            "--baseline-cv",
            "2.0",
            "--relative-mde",
            "0.05",
            "--daily-traffic",
            "50000",
            "--expected-split",
            "0.9,0.1",
        ],
        [
            "duration",
            "--metric",
            "arpu",
            "--method",
            "bayesian",
            "--baseline-mean",
            "12.5",
            "--baseline-sd",
            "25.0",
            "--relative-mde",
            "0.1",
            "--daily-traffic",
            "50000",
            "--max-expected-loss",
            "0.05",
            "--max-days",
            "3",
            "--expected-split",
            "0.9,0.1",
        ],
    ]
    for args in dur_cases:
        res = runner.invoke(cli, args)
        assert res.exit_code == 0, f"{args}: {res.output}"
        out = json.loads(res.output)
        assert validate_json_schema(out, dur_schema) == [], f"{args}: {out}"

    pwr_cases = [
        [
            "power",
            "--baseline-rate",
            "0.08",
            "--n-total",
            "20000",
            "--expected-split",
            "0.9,0.1",
            "--relative-mde",
            "0.1",
        ],
        [
            "power",
            "--baseline-rate",
            "0.08",
            "--n-total",
            "100000",
            "--n-variants",
            "3",
            "--expected-split",
            "80,10,10",
        ],
        [
            "power",
            "--metric",
            "arpu",
            "--method",
            "sequential",
            "--baseline-mean",
            "12.5",
            "--baseline-sd",
            "25",
            "--n-total",
            "100000",
            "--expected-split",
            "0.9,0.1",
            "--relative-mde",
            "0.1",
        ],
    ]
    for args in pwr_cases:
        res = runner.invoke(cli, args)
        assert res.exit_code == 0, f"{args}: {res.output}"
        out = json.loads(res.output)
        assert validate_json_schema(out, pwr_schema) == [], f"{args}: {out}"


def test_round_trip_regression_80_10_10_keeps_extra_users_on_small_arms() -> None:
    """Review of #62: power re-split 32506 as [26005, 3251, 3250] and reported 0.79995."""
    plan = fixed_horizon_duration_conversion(
        0.08, 0.2, 50_000, n_variants=3, expected_split=[80, 10, 10]
    )
    achieved = power_analysis(
        metric="conversion_rate",
        method="fixed",
        n_total=plan.n_total,
        n_variants=3,
        expected_split=[80, 10, 10],
        baseline_rate=0.08,
        relative_mde=0.2,
    )
    assert achieved.n_by_variant == plan.n_by_variant
    assert achieved.power >= 0.8 - 1e-4


@pytest.mark.parametrize("seed", range(40))
def test_round_trip_sweep_never_underpowers(seed: int) -> None:
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 5))
    split = [float(w) for w in rng.uniform(0.05, 1.0, size=k)]
    baseline = float(rng.uniform(0.02, 0.3))
    mde = float(rng.uniform(0.05, 0.4))
    plan = fixed_horizon_duration_conversion(
        baseline, mde, 10_000, n_variants=k, expected_split=split
    )
    achieved = power_analysis(
        metric="conversion_rate",
        method="fixed",
        n_total=plan.n_total,
        n_variants=k,
        expected_split=split,
        baseline_rate=baseline,
        relative_mde=mde,
    )
    assert achieved.n_by_variant == plan.n_by_variant
    assert achieved.power >= 0.8 - 1e-4


def test_bayesian_conversion_sizes_for_smallest_treatment_arm() -> None:
    """With three arms, the smaller treatment share must drive the duration."""
    kwargs = dict(baseline_rate=0.05, relative_mde=0.2, daily_total_traffic=5_000, n_variants=3)
    uneven = bayesian_duration_conversion(**kwargs, expected_split=[0.6, 0.3, 0.1], max_days=120)
    even = bayesian_duration_conversion(**kwargs, expected_split=[0.6, 0.2, 0.2], max_days=120)
    assert uneven.estimated_days is not None and even.estimated_days is not None
    assert uneven.estimated_days > even.estimated_days


def test_bayesian_arpu_treatment_noise_uses_treatment_n() -> None:
    """Shrinking only the treatment share must lengthen the ARPU Bayesian plan.

    Holding the control share at 50% isolates the treatment arm's own sample size.
    """
    kwargs = dict(
        baseline_mean=50.0,
        baseline_sd=100.0,
        relative_mde=0.1,
        daily_total_traffic=10_000,
        n_variants=3,
        max_expected_loss=1.0,
        max_days=120,
    )
    balanced = bayesian_duration_arpu(**kwargs, expected_split=[0.5, 0.25, 0.25])
    starved = bayesian_duration_arpu(**kwargs, expected_split=[0.5, 0.45, 0.05])
    assert balanced.estimated_days is not None and starved.estimated_days is not None
    assert starved.estimated_days >= 2 * balanced.estimated_days


def _reference_arpu_assurance(
    mean: float, mde: float, sd: float, n_c: int, n_t: int, prob: float, max_loss: float
) -> float:
    """Independent assurance: simulate arm means, then use normal-posterior closed forms.

    At these sample sizes the NIG posterior of each mean is ~N(x_bar, sd^2/n), so
    P(win) = Phi(d/se) and E[max(theta_C - theta_T, 0)] = se*phi(d/se) - d*Phi(-d/se).
    """
    from scipy.stats import norm

    rng = np.random.default_rng(0)
    sims = 200_000
    d = rng.normal(mean * (1 + mde), sd / math.sqrt(n_t), sims) - rng.normal(
        mean, sd / math.sqrt(n_c), sims
    )
    se = sd * math.sqrt(1 / n_c + 1 / n_t)
    z = d / se
    loss = se * norm.pdf(z) - d * norm.cdf(-z)
    return float(np.mean((norm.cdf(z) >= prob) & (loss <= max_loss) & (d > 0)))


def test_bayesian_arpu_assurance_matches_independent_reference_with_split() -> None:
    plan = bayesian_duration_arpu(
        baseline_mean=50.0,
        baseline_sd=100.0,
        relative_mde=0.1,
        daily_total_traffic=10_000,
        n_variants=2,
        expected_split=[0.9, 0.1],
        max_expected_loss=1.0,
        max_days=200,
    )
    assert plan.n_by_variant is not None
    n_c, n_t = plan.n_by_variant
    reference = _reference_arpu_assurance(50.0, 0.1, 100.0, n_c, n_t, 0.95, 1.0)
    # 300 simulations per day give a Monte Carlo SE of about 0.023.
    assert plan.diagnostics["assurance_at_selected_day"] == pytest.approx(reference, abs=0.07)
