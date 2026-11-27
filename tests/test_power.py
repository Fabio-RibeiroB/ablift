"""`ablift power`: achieved power and minimum detectable effect for a fixed sample size."""

import json

import numpy as np
import pytest
from click.testing import CliRunner
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest
from test_schema import validate_json_schema

from ablift.cli import cli
from ablift.planning import (
    fixed_horizon_duration_arpu,
    fixed_horizon_duration_conversion,
    sequential_duration_arpu,
    sequential_duration_conversion,
)
from ablift.power import power_analysis
from ablift.schemas import get_schema


def run(*args: str):
    if "--method" not in args:
        args = ("--method", "fixed", *args)
    return CliRunner().invoke(cli, ["power", *args])


@pytest.mark.parametrize("n", [500, 5_000, 50_000])
@pytest.mark.parametrize("d", [0.02, 0.1, 0.3])
def test_means_power_matches_statsmodels(n, d):
    # Cohen's d = delta / sd with baseline 100, sd 100: relative_mde == d.
    res = power_analysis(
        metric="arpu",
        method="fixed",
        n_per_variant=n,
        relative_mde=d,
        baseline_mean=100.0,
        baseline_sd=100.0,
    )
    expected = NormalIndPower().power(effect_size=d, nobs1=n, alpha=0.05, ratio=1.0)
    assert res.power == pytest.approx(expected, abs=1e-9)


def test_conversion_power_matches_simulated_z_test():
    n, p1, mde = 4_000, 0.08, 0.2
    res = power_analysis(
        metric="conversion_rate",
        method="fixed",
        n_per_variant=n,
        relative_mde=mde,
        baseline_rate=p1,
    )
    rng = np.random.default_rng(11)
    sims = 4_000
    x1 = rng.binomial(n, p1, size=sims)
    x2 = rng.binomial(n, p1 * (1 + mde), size=sims)
    rejections = sum(
        proportions_ztest([b, a], [n, n])[1] <= 0.05 for a, b in zip(x1, x2, strict=True)
    )
    assert res.power == pytest.approx(rejections / sims, abs=0.025)


@pytest.mark.parametrize(
    ("plan", "kwargs", "method", "metric"),
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
def test_round_trips_with_duration(plan, kwargs, method, metric):
    """duration's n gives back ~80% power, and power's MDE at that n is the planned MDE."""
    n = plan(relative_mde=0.1, daily_total_traffic=10_000, **kwargs).n_per_variant
    achieved = power_analysis(
        metric=metric, method=method, n_per_variant=n, relative_mde=0.1, **kwargs
    )
    assert achieved.power == pytest.approx(0.8, abs=0.003)
    mde = power_analysis(metric=metric, method=method, n_per_variant=n, power=0.8, **kwargs)
    assert mde.relative_mde == pytest.approx(0.1, rel=0.01)
    assert mde.power == pytest.approx(0.8, abs=1e-6)


def test_bonferroni_lowers_power_for_more_variants():
    two, four = (
        power_analysis(
            metric="conversion_rate",
            method="fixed",
            n_per_variant=10_000,
            relative_mde=0.1,
            baseline_rate=0.08,
            n_variants=k,
        ).power
        for k in (2, 4)
    )
    assert four < two


def test_sequential_needs_bigger_effects_than_fixed():
    fixed, seq = (
        power_analysis(
            metric="conversion_rate",
            method=m,
            n_per_variant=20_000,
            power=0.8,
            baseline_rate=0.05,
        ).relative_mde
        for m in ("fixed", "sequential")
    )
    assert seq > fixed


def test_cli_power_and_mde_outputs_match_schema():
    schema = get_schema("power-output")
    for args, solved in (
        (["--baseline-rate", "0.08", "--n-per-variant", "10000", "--relative-mde", "0.1"], "power"),
        (["--baseline-rate", "0.08", "--n-per-variant", "10000"], "relative_mde"),
        (
            [
                "--metric",
                "arpu",
                "--baseline-mean",
                "12.5",
                "--baseline-cv",
                "2",
                "--n-per-variant",
                "20000",
                "--method",
                "sequential",
            ],
            "relative_mde",
        ),
    ):
        res = run(*args)
        assert res.exit_code == 0, res.output
        out = json.loads(res.output)
        assert out["solved_for"] == solved
        validate_json_schema(out, schema)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            [
                "--baseline-rate",
                "0.08",
                "--n-per-variant",
                "100",
                "--relative-mde",
                "0.1",
                "--power",
                "0.9",
            ],
            "not both",
        ),
        (
            ["--baseline-rate", "0.08", "--n-per-variant", "100", "--sequential-tau", "0.01"],
            "only applies to --method sequential",
        ),
        (
            ["--baseline-mean", "10", "--baseline-sd", "5", "--n-per-variant", "100"],
            "Not used by --metric conversion_rate",
        ),
        (
            ["--metric", "arpu", "--baseline-mean", "10", "--n-per-variant", "100"],
            "one of --baseline-sd or --baseline-cv",
        ),
        (["--n-per-variant", "100"], "Missing --baseline-rate"),
    ],
)
def test_cli_usage_errors(args, message):
    res = run(*args)
    assert res.exit_code == 2
    assert message in res.output


def test_unreachable_power_is_a_clean_error():
    res = run("--baseline-rate", "0.5", "--n-per-variant", "3")
    assert res.exit_code == 1
    assert "increase the sample size" in res.output
