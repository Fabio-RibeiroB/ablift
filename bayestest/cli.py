from __future__ import annotations

import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import click

from .connectors import build_duration_request_from_rows, build_payload_from_rows, read_table
from .engine import analyze as run_analysis, parse_payload
from .planning import bayesian_duration_conversion, frequentist_duration_conversion
from .reporting import build_markdown_report
from .text_parser import parse_duration_prompt, parse_variant_lines

ANALYZE_EPILOG = """\b
Examples:
  bayestest analyze --input experiment.json
  bayestest analyze --input experiment.csv
  bayestest analyze --input experiment.xlsx --sheet Results
  bayestest analyze --input experiment.csv --mapping mapping.json
  bayestest analyze --input experiment.csv --enable-recommendation --prob-threshold 0.9 --max-expected-loss 0.005
  bayestest analyze --input experiment.json --report report.md
"""

ANALYZE_TEXT_EPILOG = """\b
Examples:
  bayestest analyze-text --text "control: 1000 visitors, 40 conversions; v1: 1000 visitors, 45 conversions"
  bayestest analyze-text --text-file summary.txt --output result.json
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
    return data.get("tool", {}).get("bayestest", {})


def _load_analysis_payload(
    *,
    input_path: str,
    mapping_path: str | None,
    sheet: str | None,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    config_payload = _load_project_config()
    suffix = Path(input_path).suffix.lower()
    if suffix == ".json":
        input_payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        payload = _deep_merge(config_payload, _deep_merge(input_payload, defaults))
        payload.setdefault(
            "input_interpretation",
            {
                "source_type": "json",
                "mapping_used": False,
            },
        )
        return payload

    if suffix not in {".csv", ".xlsx", ".xlsm"}:
        raise click.ClickException("Unsupported analysis input. Use .json, .csv, .xlsx, or .xlsm.")

    mapping = None
    if mapping_path:
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    rows = read_table(input_path, sheet=sheet)
    table_defaults = _deep_merge(config_payload, defaults)
    return build_payload_from_rows(rows, mapping=mapping, defaults=table_defaults)


def example_payload() -> dict:
    return {
        "experiment_name": "checkout_button_color",
        "method": "bayesian",
        "primary_metric": "conversion_rate",
        "alpha": 0.05,
        "look_index": 3,
        "max_looks": 10,
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
        "decision_thresholds": {
            "bayes_prob_beats_control": 0.95,
            "max_expected_loss": 0.001,
        },
        "samples": 50000,
        "random_seed": 7,
    }


def example_mapping() -> dict:
    return {
        "experiment_name": "checkout_button_color",
        "method": "bayesian",
        "primary_metric": "conversion_rate",
        "alpha": 0.05,
        "look_index": 3,
        "max_looks": 10,
        "columns": {
            "variant": "variant_name",
            "visitors": "sessions",
            "conversions": "click_sessions",
            "is_control": "is_control",
            "revenue_sum": "revenue_sum",
            "revenue_sum_squares": "revenue_sum_squares",
        },
        "control": {
            "column": "variant_name",
            "value": "control",
        },
        "guardrails": [
            {
                "name": "p95_latency_ms",
                "control": 420.0,
                "treatment": 430.0,
                "direction": "decrease",
                "max_relative_change": 0.05,
            }
        ],
        "decision_thresholds": {
            "bayes_prob_beats_control": 0.95,
            "max_expected_loss": 0.001,
        },
        "samples": 50000,
        "random_seed": 7,
    }


def example_duration_prompt() -> str:
    return (
        "Traffic: 50000 visitors/day\n"
        "Baseline: 4%\n"
        "MDE: 5%\n"
        "Alpha: 0.05\n"
        "Power: 0.8\n"
        "Variants: 3\n"
        "Looks: 10\n"
    )


def example_duration_mapping() -> dict:
    return {
        "method": "frequentist",
        "columns": {
            "method": "method",
            "baseline_rate": "baseline_rate",
            "relative_mde": "relative_mde",
            "daily_traffic": "daily_traffic",
            "n_variants": "n_variants",
            "alpha": "alpha",
            "power": "power",
            "max_looks": "max_looks",
            "prob_threshold": "prob_threshold",
            "max_expected_loss": "max_expected_loss",
            "assurance_target": "assurance_target",
            "max_days": "max_days",
        },
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
@click.version_option(package_name="bayestest", prog_name="bayestest")
def cli() -> None:
    """Agent-friendly CLI for Bayesian and frequentist sequential A/B/n decisions."""


@cli.command(epilog=ANALYZE_EPILOG)
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Path to input .json, .csv, .xlsx, or .xlsm.")
@click.option("--mapping", "mapping_path", required=False, type=click.Path(exists=True), help="Optional JSON mapping for CSV/XLSX column names.")
@click.option("--sheet", required=False, default=None, help="Excel sheet name for .xlsx/.xlsm input.")
@click.option("--experiment-name", required=False, default=None, help="Optional experiment name override for CSV/XLSX input.")
@click.option("--method", type=click.Choice(["bayesian", "frequentist_sequential"]), default=None, help="Optional method override for CSV/XLSX input.")
@click.option("--primary-metric", type=click.Choice(["conversion_rate", "arpu"]), default=None, help="Optional metric override for CSV/XLSX input.")
@click.option("--enable-recommendation/--no-recommendation", default=None, help="Enable or disable automated recommendation output.")
@click.option("--prob-threshold", type=float, default=None, help="Bayesian probability-to-win threshold for recommendations.")
@click.option("--max-expected-loss", type=float, default=None, help="Bayesian expected-loss threshold for recommendations.")
@click.option("--alpha", type=float, default=None, help="Optional alpha override.")
@click.option("--look-index", type=int, default=None, help="Optional sequential look index override.")
@click.option("--max-looks", type=int, default=None, help="Optional sequential max looks override.")
@click.option("--information-fraction", type=float, default=None, help="Optional sequential information fraction override.")
@click.option("--samples", type=int, default=None, help="Optional Bayesian posterior sample count override.")
@click.option("--random-seed", type=int, default=None, help="Optional Bayesian random seed override.")
@click.option("--output", "output_path", required=False, type=click.Path(), help="Path to output JSON. If omitted, writes to stdout.")
@click.option("--report", "report_path", required=False, type=click.Path(), help="Optional path to markdown report output.")
def analyze(
    input_path: str,
    mapping_path: str | None,
    sheet: str | None,
    experiment_name: str | None,
    method: str | None,
    primary_metric: str | None,
    enable_recommendation: bool | None,
    prob_threshold: float | None,
    max_expected_loss: float | None,
    alpha: float | None,
    look_index: int | None,
    max_looks: int | None,
    information_fraction: float | None,
    samples: int | None,
    random_seed: int | None,
    output_path: str | None,
    report_path: str | None,
) -> None:
    """Analyze JSON, CSV, or XLSX experiment input."""
    decision_policy = None
    if enable_recommendation is not None or prob_threshold is not None or max_expected_loss is not None:
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
        "look_index": look_index,
        "max_looks": max_looks,
        "information_fraction": information_fraction,
        "samples": samples,
        "random_seed": random_seed,
    }
    try:
        input_payload = _load_analysis_payload(
            input_path=input_path,
            mapping_path=mapping_path,
            sheet=sheet,
            defaults=defaults,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if Path(input_path).suffix.lower() in {".csv", ".xlsx", ".xlsm"} and input_payload.get("experiment_name") in {None, "", "table_input_experiment"}:
        input_payload["experiment_name"] = Path(input_path).stem
    try:
        _emit_analysis(run_analysis(parse_payload(input_payload)), output_path, report_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("analyze-file", hidden=True)
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Path to CSV/XLSX.")
@click.option("--mapping", "mapping_path", required=False, type=click.Path(exists=True), help="Optional JSON mapping for CSV/XLSX column names.")
@click.option("--sheet", required=False, default=None, help="Excel sheet name.")
@click.option("--experiment-name", required=False, default=None, help="Optional experiment name override.")
@click.option("--method", type=click.Choice(["bayesian", "frequentist_sequential"]), default=None, help="Optional method override.")
@click.option("--primary-metric", type=click.Choice(["conversion_rate", "arpu"]), default=None, help="Optional metric override.")
@click.option("--output", "output_path", required=False, type=click.Path(), help="Path to output JSON. If omitted, writes to stdout.")
@click.option("--report", "report_path", required=False, type=click.Path(), help="Optional path to markdown report output.")
def analyze_file(
    input_path: str,
    mapping_path: str | None,
    sheet: str | None,
    experiment_name: str | None,
    method: str | None,
    primary_metric: str | None,
    output_path: str | None,
    report_path: str | None,
) -> None:
    """Deprecated alias for analyze on CSV/XLSX input."""
    click.echo("Warning: 'analyze-file' is deprecated; use 'analyze --input ...' instead.", err=True)
    try:
        input_payload = _load_analysis_payload(
            input_path=input_path,
            mapping_path=mapping_path,
            sheet=sheet,
            defaults={
                "experiment_name": experiment_name,
                "method": method,
                "primary_metric": primary_metric,
            },
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if input_payload.get("experiment_name") in {None, "", "table_input_experiment"}:
        input_payload["experiment_name"] = Path(input_path).stem
    try:
        _emit_analysis(run_analysis(parse_payload(input_payload)), output_path, report_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("analyze-text", epilog=ANALYZE_TEXT_EPILOG)
@click.option("--text", required=False, default=None, help="Raw pasted text.")
@click.option("--text-file", "text_file", required=False, type=click.Path(exists=True), help="Path to pasted text file.")
@click.option("--experiment-name", default="text_input_experiment", show_default=True)
@click.option("--method", default="bayesian", show_default=True)
@click.option("--primary-metric", default="conversion_rate", show_default=True)
@click.option("--output", "output_path", required=False, type=click.Path())
@click.option("--report", "report_path", required=False, type=click.Path())
def analyze_text(
    text: str | None,
    text_file: str | None,
    experiment_name: str,
    method: str,
    primary_metric: str,
    output_path: str | None,
    report_path: str | None,
) -> None:
    """Parse pasted variant lines and analyze."""
    if not text and text_file:
        text = Path(text_file).read_text(encoding="utf-8")
    if not text:
        raise click.UsageError("Provide --text or --text-file.")
    variants = parse_variant_lines(text)
    payload = {
        "experiment_name": experiment_name,
        "method": method,
        "primary_metric": primary_metric,
        "variants": variants,
    }
    _emit_analysis(run_analysis(parse_payload(payload)), output_path, report_path)


@cli.command()
@click.option("--method", type=click.Choice(["frequentist", "bayesian"]), default=None)
@click.option("--baseline-rate", type=float, default=None, help="Baseline conversion rate, decimal (e.g. 0.04).")
@click.option("--relative-mde", type=float, default=None, help="Relative MDE, decimal (e.g. 0.05 for +5%).")
@click.option("--daily-traffic", type=int, default=None, help="Total daily traffic across variants.")
@click.option("--n-variants", type=int, default=2, show_default=True)
@click.option("--alpha", type=float, default=0.05, show_default=True)
@click.option("--power", type=float, default=0.8, show_default=True)
@click.option("--max-looks", type=int, default=10, show_default=True)
@click.option("--prob-threshold", type=float, default=0.95, show_default=True)
@click.option("--max-expected-loss", type=float, default=0.001, show_default=True)
@click.option("--assurance-target", type=float, default=0.8, show_default=True)
@click.option("--max-days", type=int, default=60, show_default=True)
@click.option("--output", "output_path", required=False, type=click.Path())
@click.option("--prompt-text", required=False, default=None, help="Natural-language assumptions text.")
@click.option("--input", "input_path", required=False, type=click.Path(exists=True), help="CSV/XLSX input for duration assumptions.")
@click.option("--mapping", "mapping_path", required=False, type=click.Path(exists=True), help="JSON mapping for duration table columns.")
@click.option("--sheet", required=False, default=None, help="Excel sheet name for duration input.")
def duration(
    method: str | None,
    baseline_rate: float | None,
    relative_mde: float | None,
    daily_traffic: int | None,
    n_variants: int,
    alpha: float,
    power: float,
    max_looks: int,
    prob_threshold: float,
    max_expected_loss: float,
    assurance_target: float,
    max_days: int,
    output_path: str | None,
    prompt_text: str | None,
    input_path: str | None,
    mapping_path: str | None,
    sheet: str | None,
) -> None:
    """Estimate test duration from assumptions."""
    if input_path and mapping_path:
        rows = read_table(input_path, sheet=sheet)
        duration_mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        file_req = build_duration_request_from_rows(rows, duration_mapping)
        method = method or file_req["method"]
        baseline_rate = baseline_rate if baseline_rate is not None else file_req["baseline_rate"]
        relative_mde = relative_mde if relative_mde is not None else file_req["relative_mde"]
        daily_traffic = daily_traffic if daily_traffic is not None else file_req["daily_traffic"]
        n_variants = file_req["n_variants"]
        alpha = file_req["alpha"]
        power = file_req["power"]
        max_looks = file_req["max_looks"]
        prob_threshold = file_req["prob_threshold"]
        max_expected_loss = file_req["max_expected_loss"]
        assurance_target = file_req["assurance_target"]
        max_days = file_req["max_days"]

    if prompt_text:
        parsed = parse_duration_prompt(prompt_text)
        method = method or "frequentist"
        baseline_rate = baseline_rate if baseline_rate is not None else parsed["baseline_rate"]
        relative_mde = relative_mde if relative_mde is not None else parsed["relative_mde"]
        daily_traffic = daily_traffic if daily_traffic is not None else parsed["daily_total_traffic"]
        alpha = alpha if alpha is not None else parsed["alpha"]
        power = power if power is not None else parsed["power"]
        n_variants = n_variants if n_variants is not None else parsed["n_variants"]
        max_looks = max_looks if max_looks is not None else parsed["max_looks"]

    if not method:
        method = click.prompt("Method", type=click.Choice(["frequentist", "bayesian"]))
    if baseline_rate is None:
        baseline_rate = click.prompt("Baseline conversion rate (decimal, e.g. 0.04)", type=float)
    if relative_mde is None:
        relative_mde = click.prompt("Relative MDE (decimal, e.g. 0.05)", type=float)
    if daily_traffic is None:
        daily_traffic = click.prompt("Total daily traffic", type=int)

    if method == "frequentist":
        plan = frequentist_duration_conversion(
            baseline_rate=baseline_rate,
            relative_mde=relative_mde,
            daily_total_traffic=daily_traffic,
            n_variants=n_variants,
            alpha=alpha,
            power=power,
            max_looks=max_looks,
        )
    else:
        plan = bayesian_duration_conversion(
            baseline_rate=baseline_rate,
            relative_mde=relative_mde,
            daily_total_traffic=daily_traffic,
            n_variants=n_variants,
            prob_threshold=prob_threshold,
            max_expected_loss=max_expected_loss,
            assurance_target=assurance_target,
            max_days=max_days,
        )

    plan_json = json.dumps(
        {
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
@click.option("--json-output", "json_output", is_flag=True, help="Print machine-readable JSON output.")
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
            click.echo(f"- {icon} {check['name']}: {check['detail']} (required {check['required']})")
    if strict and not result["ok"]:
        raise SystemExit(1)


@cli.command("example-input")
def example_input() -> None:
    """Print an example input JSON to stdout."""
    click.echo(json.dumps(example_payload(), indent=2))


@cli.command("example-mapping")
def example_mapping_cmd() -> None:
    """Print an example mapping JSON for CSV/XLSX."""
    click.echo(json.dumps(example_mapping(), indent=2))


@cli.command("example-duration-prompt")
def example_duration_prompt_cmd() -> None:
    """Print a duration prompt text example."""
    click.echo(example_duration_prompt(), nl=False)


@cli.command("example-duration-mapping")
def example_duration_mapping_cmd() -> None:
    """Print a duration mapping JSON for CSV/XLSX."""
    click.echo(json.dumps(example_duration_mapping(), indent=2))





if __name__ == "__main__":
    cli()
