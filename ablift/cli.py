from __future__ import annotations

import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import click

from .connectors import build_duration_request_from_rows, build_payload_from_rows, read_table
from .engine import analyze as run_analysis
from .engine import parse_payload
from .models import SCHEMA_VERSION
from .planning import (
    bayesian_duration_conversion,
    fixed_horizon_duration_conversion,
    sequential_duration_conversion,
)
from .reporting import build_markdown_report

ANALYZE_EPILOG = """\b
Examples:
  ablift analyze --input experiment.json
  ablift analyze --input experiment.csv
  ablift analyze --input experiment.xlsx --sheet Results
  ablift analyze --input experiment.csv --enable-recommendation --prob-threshold 0.9 --max-expected-loss 0.005
  ablift analyze --input experiment.json --report report.md
"""


def _write_text(path: str, content: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def _emit_analysis(result, output_path: str | None, report_path: str | None) -> None:
    result_json = json.dumps(result.to_dict(), indent=2)
    if output_path:
        _write_text(output_path, result_json + "\n")
    else:
        click.echo(result_json)
    if report_path:
        _write_text(report_path, build_markdown_report(result))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def _load_project_config() -> dict[str, Any]:
    pyproject_path = Path.cwd() / "pyproject.toml"
    if not pyproject_path.exists():
        return {}
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return data.get("tool", {}).get("ablift", {})


def _load_analysis_payload(
    *,
    input_path: str,
    sheet: str | None,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    config_payload = _load_project_config()
    suffix = Path(input_path).suffix.lower()
    if suffix == ".json":
        input_payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        if not isinstance(input_payload, dict):
            raise ValueError("JSON input must be an object with a 'variants' list.")
        payload = _deep_merge(config_payload, _deep_merge(input_payload, defaults))
        payload.setdefault("input_interpretation", {"source_type": "json"})
        return payload

    if suffix not in {".csv", ".xlsx", ".xlsm"}:
        raise click.ClickException("Unsupported analysis input. Use .json, .csv, .xlsx, or .xlsm.")

    rows = read_table(input_path, sheet=sheet)
    table_defaults = _deep_merge(config_payload, defaults)
    return build_payload_from_rows(rows, defaults=table_defaults)


def example_payload() -> dict:
    return {
        "experiment_name": "checkout_button_color",
        "method": "bayesian",
        "primary_metric": "conversion_rate",
        "alpha": 0.05,
        "variants": [
            {
                "name": "control",
                "visitors": 100000,
                "conversions": 4000,
                "is_control": True,
            },
            {
                "name": "treatment_a",
                "visitors": 100000,
                "conversions": 4200,
                "is_control": False,
            },
        ],
        "guardrails": [
            {
                "name": "bounce_rate",
                "control": 0.36,
                "treatment": 0.365,
                "direction": "decrease",
                "max_relative_change": 0.03,
            },
            {
                "name": "p95_latency_ms",
                "control": 420.0,
                "treatment": 430.0,
                "direction": "decrease",
                "max_relative_change": 0.05,
            },
        ],
        "decision_policy": {
            "enabled": True,
            "bayes_prob_beats_control": 0.95,
            "max_expected_loss": 0.001,
        },
        "samples": 50000,
        "random_seed": 7,
    }


def run_doctor() -> dict:
    checks = []

    py_ok = sys.version_info >= (3, 11)
    checks.append(
        {
            "name": "python_version",
            "passed": py_ok,
            "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "required": ">=3.11",
        }
    )

    for dep, required in [("numpy", ">=1.22"), ("openpyxl", ">=3.1.0")]:
        try:
            version = importlib.metadata.version(dep)
            checks.append(
                {
                    "name": f"dependency_{dep}",
                    "passed": True,
                    "detail": version,
                    "required": required,
                }
            )
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                {
                    "name": f"dependency_{dep}",
                    "passed": False,
                    "detail": "not installed",
                    "required": required,
                }
            )

    all_passed = all(c["passed"] for c in checks)
    return {"ok": all_passed, "checks": checks}


@click.group()
@click.version_option(package_name="ablift", prog_name="ablift")
def cli() -> None:
    """Agent-friendly CLI for Bayesian and sequential (mSPRT) A/B/n experiment decisions."""


@cli.command(epilog=ANALYZE_EPILOG)
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input .json, .csv, .xlsx, or .xlsm.",
)
@click.option(
    "--sheet", required=False, default=None, help="Excel sheet name for .xlsx/.xlsm input."
)
@click.option(
    "--experiment-name",
    required=False,
    default=None,
    help="Optional experiment name override for CSV/XLSX input.",
)
@click.option(
    "--method",
    type=click.Choice(["bayesian", "sequential"]),
    default=None,
    help="Optional method override for CSV/XLSX input.",
)
@click.option(
    "--primary-metric",
    type=click.Choice(["conversion_rate", "arpu"]),
    default=None,
    help="Optional metric override for CSV/XLSX input.",
)
@click.option(
    "--enable-recommendation/--no-recommendation",
    default=None,
    help="Enable or disable automated recommendation output.",
)
@click.option(
    "--prob-threshold",
    type=float,
    default=None,
    help="Bayesian probability-to-win threshold for recommendations.",
)
@click.option(
    "--max-expected-loss",
    type=float,
    default=None,
    help="Bayesian expected-loss threshold for recommendations.",
)
@click.option("--alpha", type=float, default=None, help="Optional alpha override.")
@click.option(
    "--sequential-tau",
    type=float,
    default=None,
    help="mSPRT mixing parameter (prior std on effect size). Defaults to 10% of control baseline.",
)
@click.option(
    "--samples", type=int, default=None, help="Optional Bayesian posterior sample count override."
)
@click.option(
    "--random-seed", type=int, default=None, help="Optional Bayesian random seed override."
)
@click.option(
    "--output",
    "output_path",
    required=False,
    type=click.Path(),
    help="Path to output JSON. If omitted, writes to stdout.",
)
@click.option(
    "--report",
    "report_path",
    required=False,
    type=click.Path(),
    help="Optional path to markdown report output.",
)
def analyze(
    input_path: str,
    sheet: str | None,
    experiment_name: str | None,
    method: str | None,
    primary_metric: str | None,
    enable_recommendation: bool | None,
    prob_threshold: float | None,
    max_expected_loss: float | None,
    alpha: float | None,
    sequential_tau: float | None,
    samples: int | None,
    random_seed: int | None,
    output_path: str | None,
    report_path: str | None,
) -> None:
    """Analyze JSON, CSV, or XLSX experiment input."""
    decision_policy = None
    if (
        enable_recommendation is not None
        or prob_threshold is not None
        or max_expected_loss is not None
    ):
        decision_policy = {
            "enabled": True if enable_recommendation is None else enable_recommendation,
            "bayes_prob_beats_control": prob_threshold,
            "max_expected_loss": max_expected_loss,
        }

    defaults = {
        "experiment_name": experiment_name,
        "method": method,
        "primary_metric": primary_metric,
        "decision_policy": decision_policy,
        "alpha": alpha,
        "sequential_tau": sequential_tau,
        "samples": samples,
        "random_seed": random_seed,
    }
    try:
        input_payload = _load_analysis_payload(
            input_path=input_path,
            sheet=sheet,
            defaults=defaults,
        )
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Could not read {input_path}: {exc}") from exc
    if Path(input_path).suffix.lower() in {".csv", ".xlsx", ".xlsm"} and input_payload.get(
        "experiment_name"
    ) in {None, "", "table_input_experiment"}:
        input_payload["experiment_name"] = Path(input_path).stem
    try:
        _emit_analysis(run_analysis(parse_payload(input_payload)), output_path, report_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


DURATION_EPILOG = """\b
Examples:
  ablift duration --method sequential --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000
  ablift duration --method fixed --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3
  ablift duration --method bayesian --baseline-rate 0.04 --relative-mde 0.08 --daily-traffic 60000
  ablift duration --input assumptions.csv
"""


_BAYESIAN_ONLY = ("prob_threshold", "max_expected_loss", "assurance_target", "max_days")


@cli.command(epilog=DURATION_EPILOG)
@click.option(
    "--method",
    type=click.Choice(["sequential", "fixed", "bayesian"]),
    default=None,
    help="Planning model. 'sequential' matches 'analyze --method sequential' (mSPRT). "
    "[default: sequential]",
)
@click.option(
    "--baseline-rate",
    type=float,
    default=None,
    help="Baseline conversion rate, decimal (e.g. 0.04).",
)
@click.option(
    "--relative-mde", type=float, default=None, help="Relative MDE, decimal (e.g. 0.05 for +5%)."
)
@click.option(
    "--daily-traffic", type=int, default=None, help="Total daily traffic across all variants."
)
@click.option(
    "--n-variants", type=int, default=None, help="Variants including control. [default: 2]"
)
@click.option("--alpha", type=float, default=None, help="Family-wise alpha. [default: 0.05]")
@click.option("--power", type=float, default=None, help="Target power. [default: 0.8]")
@click.option(
    "--sequential-tau",
    type=float,
    default=None,
    help="mSPRT mixing parameter. Defaults to 10% of the baseline rate.",
)
@click.option(
    "--prob-threshold", type=float, default=None, help="Bayesian P(win) threshold. [default: 0.95]"
)
@click.option(
    "--max-expected-loss",
    type=float,
    default=None,
    help="Bayesian expected-loss threshold. [default: 0.001]",
)
@click.option(
    "--assurance-target",
    type=float,
    default=None,
    help="Bayesian probability of reaching a decision. [default: 0.8]",
)
@click.option(
    "--max-days", type=int, default=None, help="Bayesian search horizon in days. [default: 60]"
)
@click.option("--output", "output_path", required=False, type=click.Path())
@click.option(
    "--input",
    "input_path",
    required=False,
    type=click.Path(exists=True),
    help="CSV/XLSX with named columns (baseline_rate, relative_mde, daily_traffic, ...). "
    "CLI flags override file values.",
)
@click.option("--sheet", required=False, default=None, help="Excel sheet name for duration input.")
def duration(
    method: str | None,
    baseline_rate: float | None,
    relative_mde: float | None,
    daily_traffic: int | None,
    n_variants: int | None,
    alpha: float | None,
    power: float | None,
    sequential_tau: float | None,
    prob_threshold: float | None,
    max_expected_loss: float | None,
    assurance_target: float | None,
    max_days: int | None,
    output_path: str | None,
    input_path: str | None,
    sheet: str | None,
) -> None:
    """Estimate how long a conversion-rate test needs to run."""
    flags = {
        "method": method,
        "baseline_rate": baseline_rate,
        "relative_mde": relative_mde,
        "daily_traffic": daily_traffic,
        "n_variants": n_variants,
        "alpha": alpha,
        "power": power,
        "sequential_tau": sequential_tau,
        "prob_threshold": prob_threshold,
        "max_expected_loss": max_expected_loss,
        "assurance_target": assurance_target,
        "max_days": max_days,
    }
    req: dict[str, Any] = {
        "method": "sequential",
        "n_variants": 2,
        "alpha": 0.05,
        "power": 0.8,
        "prob_threshold": 0.95,
        "max_expected_loss": 0.001,
        "assurance_target": 0.8,
        "max_days": 60,
    }
    try:
        file_keys: set[str] = set()
        if input_path:
            file_req = build_duration_request_from_rows(read_table(input_path, sheet=sheet))
            # When --method overrides the file's method, the file's other settings were written
            # for a different planner, so they are not counted as explicitly requested.
            overridden = method is not None and file_req.get("method", method) != method
            file_keys = set() if overridden else set(file_req)
            req.update(file_req)
        req.update({k: v for k, v in flags.items() if v is not None})

        missing = [
            f"--{k.replace('_', '-')}"
            for k in ("baseline_rate", "relative_mde", "daily_traffic")
            if req.get(k) is None
        ]
        if missing:
            raise click.UsageError(f"Missing required assumptions: {', '.join(missing)}.")

        # Reject settings the chosen planner would silently ignore.
        explicit = {k for k, v in flags.items() if v is not None} | file_keys
        not_applicable = {
            "bayesian": ("alpha", "power", "sequential_tau"),
            "fixed": ("sequential_tau", *_BAYESIAN_ONLY),
            "sequential": _BAYESIAN_ONLY,
        }.get(req["method"], ())
        ignored = [f"--{k.replace('_', '-')}" for k in not_applicable if k in explicit]
        if ignored:
            raise click.UsageError(
                f"Not used by --method {req['method']}: {', '.join(ignored)} "
                "(set as a flag or input-file column). Remove them or choose a method that "
                "uses them."
            )

        common = {
            "baseline_rate": req["baseline_rate"],
            "relative_mde": req["relative_mde"],
            "daily_total_traffic": req["daily_traffic"],
            "n_variants": req["n_variants"],
        }
        if req["method"] == "sequential":
            plan = sequential_duration_conversion(
                **common,
                alpha=req["alpha"],
                power=req["power"],
                sequential_tau=req.get("sequential_tau"),
            )
        elif req["method"] == "fixed":
            plan = fixed_horizon_duration_conversion(
                **common, alpha=req["alpha"], power=req["power"]
            )
        elif req["method"] == "bayesian":
            plan = bayesian_duration_conversion(
                **common,
                prob_threshold=req["prob_threshold"],
                max_expected_loss=req["max_expected_loss"],
                assurance_target=req["assurance_target"],
                max_days=req["max_days"],
            )
        else:
            raise click.UsageError(
                f"Unknown method '{req['method']}'. Use sequential, fixed, or bayesian."
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    plan_json = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "method": plan.method,
            "estimated_days": plan.estimated_days,
            "n_per_variant": plan.n_per_variant,
            "assumptions": plan.assumptions,
            "diagnostics": plan.diagnostics,
        },
        indent=2,
    )
    if output_path:
        _write_text(output_path, plan_json + "\n")
    else:
        click.echo(plan_json)


@cli.command()
@click.option("--strict", is_flag=True, help="Exit non-zero if any check fails.")
@click.option(
    "--json-output", "json_output", is_flag=True, help="Print machine-readable JSON output."
)
def doctor(strict: bool, json_output: bool) -> None:
    """Run environment and dependency checks for agents/CI."""
    result = run_doctor()
    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        status = "OK" if result["ok"] else "FAIL"
        click.echo(f"Doctor status: {status}")
        for check in result["checks"]:
            icon = "PASS" if check["passed"] else "FAIL"
            click.echo(
                f"- {icon} {check['name']}: {check['detail']} (required {check['required']})"
            )
    if strict and not result["ok"]:
        raise SystemExit(1)


@cli.command("example-input")
def example_input() -> None:
    """Print an example input JSON to stdout."""
    click.echo(json.dumps(example_payload(), indent=2))


if __name__ == "__main__":
    cli()
