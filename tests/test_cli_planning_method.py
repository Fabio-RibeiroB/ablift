"""`duration` and `power` need an explicit method so planning never silently switches framework."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ablift.cli import cli

DURATION_ARGS = ["--baseline-rate", "0.04", "--relative-mde", "0.08", "--daily-traffic", "60000"]
POWER_ARGS = ["--baseline-rate", "0.04", "--n-per-variant", "50000", "--relative-mde", "0.1"]


def test_duration_without_method_is_a_usage_error():
    result = CliRunner().invoke(cli, ["duration", *DURATION_ARGS])
    assert result.exit_code == 2
    assert "Missing --method" in result.output
    for choice in ("sequential", "fixed", "bayesian"):
        assert f"'{choice}'" in result.output
    assert "check results as they come in" in result.output


def test_power_without_method_is_a_usage_error():
    result = CliRunner().invoke(cli, ["power", *POWER_ARGS])
    assert result.exit_code == 2
    assert "Missing --method" in result.output
    assert "'sequential'" in result.output and "'fixed'" in result.output
    assert "'bayesian'" not in result.output
    assert "check results as they come in" in result.output


@pytest.mark.parametrize("method", ["sequential", "fixed", "bayesian"])
def test_duration_method_from_json_input_counts_as_explicit(tmp_path: Path, method: str):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "method": method,
                "baseline_rate": 0.04,
                "relative_mde": 0.08,
                "daily_traffic": 60000,
            }
        )
    )
    result = CliRunner().invoke(cli, ["duration", "--input", str(path)])
    assert result.exit_code == 0, result.output
    expected = "fixed_horizon" if method == "fixed" else method
    assert json.loads(result.output)["method"] == expected


def test_duration_method_from_stdin_csv_counts_as_explicit():
    csv = "method,baseline_rate,relative_mde,daily_traffic\nfixed,0.04,0.08,60000\n"
    result = CliRunner().invoke(cli, ["duration", "--input", "-"], input=csv)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["method"] == "fixed_horizon"


def test_duration_input_without_method_is_a_usage_error():
    payload = '{"baseline_rate": 0.04, "relative_mde": 0.08, "daily_traffic": 60000}'
    result = CliRunner().invoke(cli, ["duration", "--input", "-"], input=payload)
    assert result.exit_code == 2
    assert "Missing --method" in result.output


@pytest.mark.parametrize("command", ["duration", "power"])
def test_help_no_longer_advertises_a_default_method(command: str):
    result = CliRunner().invoke(cli, [command, "--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "default: sequential" not in text
    assert "default: fixed" not in text
