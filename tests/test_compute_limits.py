"""Expensive computations fail fast with clean errors instead of exhausting memory or time."""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from ablift import cli as cli_module
from ablift.cli import cli
from ablift.engine import (
    MAX_POSTERIOR_DRAWS,
    analyze,
    parse_payload,
    sample_mean_posterior_from_moments,
)
from ablift.planning import _first_day_reaching, bayesian_duration_conversion

CHECKOUT = "examples/checkout_test.csv"  # 3 variants


def invoke(*args: str):
    return CliRunner().invoke(cli, list(args))


def test_bayesian_samples_over_memory_budget_is_a_clean_error():
    too_many = MAX_POSTERIOR_DRAWS // 3 + 1
    res = invoke("analyze", "--input", CHECKOUT, "--samples", str(too_many))
    assert res.exit_code == 1
    assert "posterior-draw memory limit" in res.output
    assert f"{MAX_POSTERIOR_DRAWS // 3:,}" in res.output
    assert res.exception is None or isinstance(res.exception, SystemExit)


def test_samples_limit_only_applies_to_bayesian():
    # samples is unused outside Bayesian mode, so a huge value must not be rejected there.
    payload = {
        "method": "sequential",
        "samples": 10**9,
        "variants": [
            {"name": "control", "visitors": 30000, "conversions": 1200, "is_control": True},
            {"name": "t", "visitors": 30000, "conversions": 1410},
        ],
    }
    assert analyze(parse_payload(payload)).comparisons[0].significant is True


def test_bootstrap_over_budget_is_a_usage_error(tmp_path):
    rng = np.random.default_rng(1)
    rows = ["variant,revenue"] + [
        f"{v},{x:.2f}" for v in ("a", "b") for x in rng.exponential(10.0, size=5000)
    ]
    path = tmp_path / "rows.csv"
    path.write_text("\n".join(rows) + "\n")
    res = invoke("analyze", "--input", str(path), "--bootstrap-samples", "1000000")
    assert res.exit_code == 2
    assert "use at most 100,000" in res.output


def test_out_of_memory_becomes_a_clean_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise MemoryError

    monkeypatch.setattr(cli_module, "run_analysis", boom)
    res = invoke("analyze", "--input", CHECKOUT)
    assert res.exit_code == 1
    assert "Ran out of memory" in res.output
    assert "Traceback" not in res.output


@pytest.mark.parametrize("target_day", [1, 2, 7, 64, 200, 365])
def test_day_search_matches_linear_scan_with_few_evaluations(target_day):
    days = list(range(1, 366))
    calls = []

    def assurance(day: int) -> float:
        calls.append(day)
        return 0.9 if day >= target_day else 0.1

    assert _first_day_reaching(0.8, days, assurance) == (target_day, 0.9)
    assert len(calls) <= 2 * 9 + 2  # O(log2 365), not 365


def test_day_search_returns_none_when_target_never_reached():
    assert _first_day_reaching(0.8, list(range(1, 31)), lambda _d: 0.5) is None
    assert _first_day_reaching(0.8, [], lambda _d: 1.0) is None


def test_vectorised_nig_draws_match_scalar_distribution():
    rng = np.random.default_rng(0)
    means = np.array([[10.0], [20.0]])
    ss = np.array([[400.0], [900.0]])
    draws = sample_mean_posterior_from_moments(100, means, ss, (2, 200_000), rng)
    scalar = sample_mean_posterior_from_moments(100, 10.0, 400.0, 200_000, rng)
    assert draws.shape == (2, 200_000)
    assert draws[0].mean() == pytest.approx(scalar.mean(), abs=0.01)
    assert draws[0].std() == pytest.approx(scalar.std(), rel=0.02)
    assert draws[1].mean() == pytest.approx(20.0, abs=0.01)


def test_long_bayesian_duration_search_is_fast_and_reports_none():
    plan = bayesian_duration_conversion(0.04, 0.01, 1000, max_days=365)
    assert plan.estimated_days is None
    assert json.dumps(plan.diagnostics)
