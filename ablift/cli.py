from __future__ import annotations

import importlib.metadata
import json
import math
import re
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click

from .connectors import (
    UsageOptionError,
    build_duration_request_from_dict,
    build_duration_request_from_rows,
    build_payload_from_rows,
    read_csv_text,
    read_table,
)
from .engine import analyze as run_analysis
from .engine import parse_payload
from .models import SCHEMA_VERSION
from .planning import (
    _resolve_split,
    bayesian_duration_arpu,
    bayesian_duration_conversion,
    fixed_horizon_duration_arpu,
    fixed_horizon_duration_conversion,
    sequential_duration_arpu,
    sequential_duration_conversion,
)
from .power import power_analysis
from .reporting import build_markdown_report
from .schemas import SCHEMA_NAMES, get_all_schemas, get_schema

ANALYZE_EPILOG = """\b
Examples:
  ablift analyze --input experiment.json
  ablift analyze --input experiment.csv
  ablift analyze --input experiment.csv --method fixed
  ablift analyze --input experiment.csv --expected-split 0.9,0.1
  ablift analyze --input experiment.xlsx --sheet Results
  ablift analyze --input experiment.csv --recommend --prob-threshold 0.9 --max-expected-loss 0.005
  ablift analyze --input experiment.json --report report.md
  ablift example-input | ablift analyze --input -
  cat experiment.csv | ablift analyze --input - --input-format csv
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


_ATOMIC_MERGE_KEYS = {"expected_split"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key not in _ATOMIC_MERGE_KEYS
            and isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def _parse_expected_split_option(raw: str) -> list[float] | dict[str, float]:
    """Parse --expected-split into a list of weights or a variant-name mapping."""
    text = raw.strip()
    if not text:
        raise ValueError("--expected-split cannot be empty.")

    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"Malformed --expected-split {raw!r}: empty entry between commas.")

    has_eq = ["=" in part for part in parts]
    if any(has_eq) and not all(has_eq):
        raise ValueError(
            f"Malformed --expected-split {raw!r}: cannot mix positional weights and name=weight pairs."
        )

    if all(has_eq):
        mapping: dict[str, float] = {}
        for part in parts:
            pieces = part.split("=")
            if len(pieces) != 2:
                raise ValueError(
                    f"Malformed --expected-split entry {part!r}: expected 'name=weight'."
                )
            name, weight_str = pieces[0].strip(), pieces[1].strip()
            if not name or not weight_str:
                raise ValueError(
                    f"Malformed --expected-split entry {part!r}: expected 'name=weight'."
                )
            if name in mapping:
                raise ValueError(f"Duplicate variant name {name!r} in --expected-split.")
            try:
                weight = float(weight_str)
            except ValueError:
                raise ValueError(
                    f"Invalid weight {weight_str!r} for variant {name!r} in --expected-split: "
                    "must be numeric."
                ) from None
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"Invalid weight {weight_str!r} for variant {name!r} in --expected-split: "
                    "must be a finite positive number."
                )
            mapping[name] = weight
        return mapping

    weights: list[float] = []
    for part in parts:
        try:
            weight = float(part)
        except ValueError:
            raise ValueError(
                f"Invalid weight {part!r} in --expected-split: must be numeric."
            ) from None
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"Invalid weight {part!r} in --expected-split: must be a finite positive number."
            )
        weights.append(weight)
    return weights


def _load_project_config(use_config: bool = True) -> tuple[dict[str, Any], str | None]:
    """Return ``[tool.ablift]`` from ``./pyproject.toml`` and the file's absolute path.

    The path is ``None`` when config is disabled, the file is missing, or it has no
    ``[tool.ablift]`` table, so callers can report exactly where settings came from.
    """
    if not use_config:
        return {}, None
    pyproject_path = Path.cwd() / "pyproject.toml"
    if not pyproject_path.is_file():
        return {}, None
    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise click.ClickException(
            f"Could not parse {pyproject_path}: {exc}. Fix it or pass --no-config."
        ) from exc
    config = data.get("tool", {}).get("ablift")
    if not isinstance(config, dict):
        return {}, None
    return config, str(pyproject_path.resolve())


def _read_stdin() -> str:
    raw = click.get_text_stream("stdin").read().lstrip("\ufeff")
    if not raw.strip():
        raise ValueError("Stdin input is empty.")
    return raw


def _validate_sheet_option(
    input_path: str | None, input_format: str | None, sheet: str | None
) -> None:
    if sheet is None:
        return
    if input_path is None:
        raise click.UsageError("--sheet requires --input with an .xlsx/.xlsm file.")
    if input_path == "-":
        raise click.UsageError("--sheet cannot be used when reading from stdin.")
    if input_format is not None or Path(input_path).suffix.lower() not in {".xlsx", ".xlsm"}:
        raise click.UsageError("--sheet is only supported for .xlsx/.xlsm input.")


def _merge_json_payload(
    input_payload: Any,
    config_payload: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(input_payload, dict):
        raise ValueError("JSON input must be an object with a 'variants' list.")

    input_variants = input_payload.get("variants")
    input_has_shares = isinstance(input_variants, list) and any(
        isinstance(row, dict) and row.get("expected_share") is not None for row in input_variants
    )
    effective_config = config_payload
    if (
        input_has_shares
        and input_payload.get("expected_split") is None
        and "expected_split" in config_payload
    ):
        effective_config = {k: v for k, v in config_payload.items() if k != "expected_split"}

    payload = _deep_merge(effective_config, _deep_merge(input_payload, defaults))
    if defaults.get("expected_split") is not None and isinstance(payload.get("variants"), list):
        payload["variants"] = [
            {k: v for k, v in row.items() if k != "expected_share"}
            if isinstance(row, dict)
            else row
            for row in payload["variants"]
        ]
    payload.setdefault("input_interpretation", {"source_type": "json"})
    return payload


def _reject_row_only_options_for_json(merged_defaults: dict[str, Any]) -> None:
    if merged_defaults.get("winsorize") is not None:
        raise UsageOptionError(
            "--winsorize requires row-level ARPU CSV/XLSX input; aggregated JSON inputs have no rows to cap."
        )
    if merged_defaults.get("bootstrap_samples") is not None:
        raise UsageOptionError(
            "--bootstrap-samples requires row-level ARPU CSV/XLSX input; aggregated JSON inputs have no rows to resample."
        )
    if merged_defaults.get("covariate_column") is not None:
        raise UsageOptionError(
            "--covariate-column requires row-level CSV/XLSX input; for JSON inputs, supply "
            "covariate_sum, covariate_sum_squares, and outcome_covariate_sum per variant."
        )


def _drop_unused_method_settings(
    settings: dict[str, Any], method: str, *, bootstrap: bool
) -> dict[str, Any]:
    """Drop method-scoped settings the effective method would not use.

    Applied to ``[tool.ablift]`` project defaults (e.g. ``samples`` for Bayesian runs) and
    to an input file whose method ``--method`` overrides: neither is an explicit request
    for this run, so neither may trip the engine's method-scoped rejections.
    """
    unused: set[str] = set()
    if method != "bayesian":
        unused.add("samples")
        if not bootstrap:
            unused.add("random_seed")
    if method != "sequential":
        unused.add("sequential_tau")
    return {k: v for k, v in settings.items() if k not in unused}


def _load_analysis_payload(
    *,
    input_path: str,
    input_format: str | None,
    sheet: str | None,
    defaults: dict[str, Any],
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    bootstrap = defaults.get("bootstrap_samples") is not None

    def load_json(raw: str) -> dict[str, Any]:
        _reject_row_only_options_for_json(_deep_merge(config_payload, defaults))
        input_payload = json.loads(raw)
        input_method = input_payload.get("method") if isinstance(input_payload, dict) else None
        method = str(
            defaults.get("method") or input_method or config_payload.get("method") or "bayesian"
        )
        if defaults.get("method") is not None and input_method not in {None, method}:
            # Like `duration`: --method overrides the file's method, so the file's settings
            # scoped to its own method were written for a different run.
            input_payload = _drop_unused_method_settings(input_payload, method, bootstrap=bootstrap)
        scoped = _drop_unused_method_settings(config_payload, method, bootstrap=bootstrap)
        return _merge_json_payload(input_payload, scoped, defaults)

    method = defaults.get("method") or config_payload.get("method") or "bayesian"
    merged_defaults = _deep_merge(
        _drop_unused_method_settings(config_payload, str(method), bootstrap=bootstrap),
        defaults,
    )
    if input_path == "-":
        raw = _read_stdin()
        fmt = input_format or "json"
        if fmt == "json":
            return load_json(raw)
        rows = read_csv_text(raw)
        payload = build_payload_from_rows(rows, defaults=merged_defaults)
        if payload.get("experiment_name") in {None, "", "table_input_experiment"}:
            payload["experiment_name"] = "stdin"
        return payload

    suffix = Path(input_path).suffix.lower()
    if input_format == "json" or (input_format is None and suffix == ".json"):
        return load_json(Path(input_path).read_text(encoding="utf-8-sig"))

    if input_format == "csv":
        rows = read_csv_text(Path(input_path).read_text(encoding="utf-8-sig"))
    elif suffix in {".csv", ".xlsx", ".xlsm"}:
        rows = read_table(input_path, sheet=sheet)
    else:
        raise click.ClickException("Unsupported analysis input. Use .json, .csv, .xlsx, or .xlsm.")

    payload = build_payload_from_rows(rows, defaults=merged_defaults)
    if payload.get("experiment_name") in {None, "", "table_input_experiment"}:
        payload["experiment_name"] = Path(input_path).stem
    return payload


def _load_duration_request(
    *,
    input_path: str,
    input_format: str | None,
    sheet: str | None,
) -> dict[str, Any]:
    if input_path == "-":
        raw = _read_stdin()
        fmt = input_format or ("json" if raw.lstrip().startswith("{") else "csv")
        if fmt == "json":
            return build_duration_request_from_dict(json.loads(raw))
        return build_duration_request_from_rows(read_csv_text(raw))

    suffix = Path(input_path).suffix.lower()
    if input_format == "json" or (input_format is None and suffix == ".json"):
        raw = Path(input_path).read_text(encoding="utf-8-sig")
        if not raw.strip():
            raise ValueError("Duration input file is empty.")
        return build_duration_request_from_dict(json.loads(raw))
    if input_format == "csv":
        return build_duration_request_from_rows(
            read_csv_text(Path(input_path).read_text(encoding="utf-8-sig"))
        )
    return build_duration_request_from_rows(read_table(input_path, sheet=sheet))


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
                "metric": "conversion_rate",
                "direction": "decrease",
                "max_relative_change": 0.03,
                "variants": [
                    {"name": "control", "visitors": 100000, "conversions": 36000},
                    {"name": "treatment_a", "visitors": 100000, "conversions": 36500},
                ],
            },
            {
                "name": "p95_latency_ms",
                "control": 420.0,
                "treatment": 430.0,
                "direction": "decrease",
                "max_relative_change": 0.05,
            },
        ],
        "secondary_metrics": [
            {
                "name": "revenue_per_visitor",
                "metric": "arpu",
                "variants": [
                    {
                        "name": "control",
                        "visitors": 100000,
                        "revenue_sum": 480000.0,
                        "revenue_sum_squares": 28800000.0,
                    },
                    {
                        "name": "treatment_a",
                        "visitors": 100000,
                        "revenue_sum": 505000.0,
                        "revenue_sum_squares": 30500000.0,
                    },
                ],
            }
        ],
        "samples": 50000,
        "random_seed": 7,
    }


_MIN_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("click", ">=8.0"),
    ("numpy", ">=1.22"),
    ("openpyxl", ">=3.1.0"),
)


def _version_meets_minimum(installed: str, minimum: str) -> bool:
    """Return True if PEP 440 version ``installed`` satisfies ``>= minimum``."""
    min_clean = minimum.removeprefix(">=").strip()
    inst_match = re.match(r"^\s*(\d+(?:\.\d+)*)(.*)$", installed)
    min_match = re.match(r"^\s*(\d+(?:\.\d+)*)", min_clean)
    if not inst_match or not min_match:
        return False
    inst_parts = [int(p) for p in inst_match.group(1).split(".")]
    min_parts = [int(p) for p in min_match.group(1).split(".")]
    width = max(len(inst_parts), len(min_parts))
    inst_tuple = tuple(inst_parts + [0] * (width - len(inst_parts)))
    min_tuple = tuple(min_parts + [0] * (width - len(min_parts)))
    if inst_tuple != min_tuple:
        return inst_tuple > min_tuple
    suffix = inst_match.group(2).strip().lower()
    # A pre-release (e.g. 1.22.0a1 / 1.22.0rc1 / 1.22.0.dev0) sorts below the final release.
    return not bool(re.match(r"^(?:\.?(?:a|b|rc|dev)\d*)", suffix))


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

    for dep, required in _MIN_DEPENDENCIES:
        try:
            version = importlib.metadata.version(dep)
            checks.append(
                {
                    "name": f"dependency_{dep}",
                    "passed": _version_meets_minimum(version, required),
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


def _validate_analyze_method_flags(
    *,
    effective_method: str,
    sequential_tau: float | None,
    prob_threshold: float | None,
    max_expected_loss: float | None,
    samples: int | None,
    random_seed: int | None,
    bootstrap_samples: int | None = None,
) -> None:
    if sequential_tau is not None and effective_method != "sequential":
        raise click.UsageError(
            f"--sequential-tau only applies to --method sequential, not --method {effective_method}."
        )
    if effective_method != "bayesian":
        bayesian_flags: list[str] = []
        if prob_threshold is not None:
            bayesian_flags.append("--prob-threshold")
        if max_expected_loss is not None:
            bayesian_flags.append("--max-expected-loss")
        if samples is not None:
            bayesian_flags.append("--samples")
        if random_seed is not None and bootstrap_samples is None:
            bayesian_flags.append("--random-seed")
        if bayesian_flags:
            joined = ", ".join(bayesian_flags)
            raise click.UsageError(
                f"{joined} only apply to --method bayesian, not --method {effective_method}."
            )


@click.group()
@click.version_option(package_name="ablift", prog_name="ablift")
def cli() -> None:
    """Agent-friendly CLI for Bayesian, sequential (mSPRT), and fixed-horizon A/B/n experiment analysis and planning."""


@cli.command(epilog=ANALYZE_EPILOG)
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, allow_dash=True),
    help="Path to input .json, .csv, .xlsx, or .xlsm, or '-' to read from stdin.",
)
@click.option(
    "--input-format",
    type=click.Choice(["json", "csv"]),
    default=None,
    help="Input format for stdin (defaults to json) or to override file extension.",
)
@click.option(
    "--sheet", required=False, default=None, help="Excel sheet name for .xlsx/.xlsm input."
)
@click.option(
    "--experiment-name",
    required=False,
    default=None,
    help="Optional experiment name override (applies to JSON, CSV/XLSX, and stdin inputs).",
)
@click.option(
    "--method",
    type=click.Choice(["bayesian", "sequential", "fixed"]),
    default=None,
    help=(
        "Statistical method: 'bayesian', 'sequential' (always-valid mSPRT), or "
        "'fixed' (fixed-horizon frequentist, valid only for a single planned look). "
        "[default: bayesian]"
    ),
)
@click.option(
    "--primary-metric",
    type=click.Choice(["conversion_rate", "arpu", "ratio"]),
    default=None,
    help=(
        "Primary metric override (applies to JSON, CSV/XLSX, and stdin inputs). "
        "[default: conversion_rate]"
    ),
)
@click.option(
    "--winsorize",
    type=float,
    default=None,
    help=(
        "Upper quantile cap in (0.5, 1.0) for pooled row-level ARPU winsorization (e.g. 0.99). "
        "Rejected on aggregated or non-ARPU inputs."
    ),
)
@click.option(
    "--bootstrap-samples",
    type=int,
    default=None,
    help=(
        "Optional percentile bootstrap resample count (>= 100) for row-level ARPU CI cross-check."
    ),
)
@click.option(
    "--covariate-column",
    type=str,
    default=None,
    help=(
        "Pre-experiment covariate column name or 1-based column index for CUPED variance "
        "reduction on row-level conversion_rate or arpu CSV/XLSX input."
    ),
)
@click.option(
    "--recommend/--no-recommend",
    "recommend",
    default=None,
    help=(
        "Opt in to (or disable) automated ship/wait/rollback recommendation output "
        "(off by default across all methods; overrides input and config)."
    ),
)
@click.option(
    "--prob-threshold",
    type=float,
    default=None,
    help="Bayesian probability-to-win threshold for recommendations (implies --recommend). [default: 0.95]",
)
@click.option(
    "--max-expected-loss",
    type=float,
    default=None,
    help="Bayesian expected-loss threshold for recommendations (implies --recommend). [default: 0.001]",
)
@click.option(
    "--alpha",
    type=float,
    default=None,
    help="Family-wise alpha override (overrides input and config). [default: 0.05]",
)
@click.option(
    "--sequential-tau",
    type=float,
    default=None,
    help="mSPRT mixing parameter (prior std on effect size; sequential only). Defaults to 10% of control baseline.",
)
@click.option(
    "--samples",
    type=int,
    default=None,
    help="Bayesian posterior sample count override. [default: 50000]",
)
@click.option(
    "--random-seed",
    type=int,
    default=None,
    help="Random seed override for Bayesian sampling or ARPU bootstrap. [default: 7]",
)
@click.option(
    "--expected-split",
    type=str,
    default=None,
    help=(
        "Expected traffic split weights for SRM check, as comma-separated numbers "
        "(e.g. '0.9,0.1' or '90,10') or name=weight pairs (e.g. 'control=0.8,v1=0.2')."
    ),
)
@click.option(
    "--no-config",
    "no_config",
    is_flag=True,
    help=(
        "Ignore [tool.ablift] in ./pyproject.toml. Without it, analysis_settings.config_source "
        "reports the config file that was applied."
    ),
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
    input_format: str | None,
    sheet: str | None,
    experiment_name: str | None,
    method: str | None,
    primary_metric: str | None,
    winsorize: float | None,
    bootstrap_samples: int | None,
    covariate_column: str | None,
    recommend: bool | None,
    prob_threshold: float | None,
    max_expected_loss: float | None,
    alpha: float | None,
    sequential_tau: float | None,
    samples: int | None,
    random_seed: int | None,
    expected_split: str | None,
    no_config: bool,
    output_path: str | None,
    report_path: str | None,
) -> None:
    """Analyze JSON, CSV, XLSX, or stdin experiment input."""
    _validate_sheet_option(input_path, input_format, sheet)

    if winsorize is not None and (not math.isfinite(winsorize) or not (0.5 < winsorize < 1.0)):
        raise click.UsageError(
            "--winsorize must be a finite quantile strictly between 0.5 and 1.0 (e.g. 0.99)."
        )
    if bootstrap_samples is not None and bootstrap_samples < 100:
        raise click.UsageError("--bootstrap-samples must be an integer >= 100.")

    decision_policy = None
    if recommend is not None or prob_threshold is not None or max_expected_loss is not None:
        decision_policy = {
            "enabled": True if recommend is None else recommend,
            "bayes_prob_beats_control": prob_threshold,
            "max_expected_loss": max_expected_loss,
        }

    parsed_expected_split = None
    if expected_split is not None:
        try:
            parsed_expected_split = _parse_expected_split_option(expected_split)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    defaults = {
        "experiment_name": experiment_name,
        "method": method,
        "primary_metric": primary_metric,
        "winsorize": winsorize,
        "bootstrap_samples": bootstrap_samples,
        "covariate_column": covariate_column,
        "decision_policy": decision_policy,
        "alpha": alpha,
        "sequential_tau": sequential_tau,
        "samples": samples,
        "random_seed": random_seed,
        "expected_split": parsed_expected_split,
    }
    config_payload, config_source = _load_project_config(use_config=not no_config)
    try:
        input_payload = _load_analysis_payload(
            input_path=input_path,
            input_format=input_format,
            sheet=sheet,
            defaults=defaults,
            config_payload=config_payload,
        )
    except UsageOptionError as exc:
        raise click.UsageError(str(exc)) from exc
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        source = "stdin" if input_path == "-" else input_path
        raise click.ClickException(f"Could not read {source}: {exc}") from exc
    except MemoryError as exc:
        raise click.ClickException(_OUT_OF_MEMORY) from exc

    effective_method = str(input_payload.get("method") or "bayesian")
    _validate_analyze_method_flags(
        effective_method=effective_method,
        sequential_tau=sequential_tau,
        prob_threshold=prob_threshold,
        max_expected_loss=max_expected_loss,
        samples=samples,
        random_seed=random_seed,
        bootstrap_samples=bootstrap_samples,
    )

    try:
        result = run_analysis(parse_payload(input_payload))
        result.analysis_settings.config_source = config_source
        _emit_analysis(result, output_path, report_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except MemoryError as exc:
        raise click.ClickException(_OUT_OF_MEMORY) from exc


_OUT_OF_MEMORY = (
    "Ran out of memory. Lower --samples or --bootstrap-samples, or pass aggregated "
    "per-variant sums instead of row-level data."
)


def _parse_positional_expected_split(raw: str) -> list[float]:
    """Parse --expected-split for duration/power, requiring positional weights (control first)."""
    try:
        parsed = _parse_expected_split_option(raw)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if isinstance(parsed, dict):
        raise click.UsageError(
            "--expected-split takes positional comma-separated positive weights in variant order "
            "(control first), e.g. '0.9,0.1', not name=weight pairs."
        )
    return parsed


_METHOD_HINT = (
    "Use 'sequential' if you'll check results as they come in, 'fixed' for a single planned look"
)


def _missing_method_error(
    choices: tuple[str, ...], *, from_input: bool = False
) -> click.UsageError:
    """Planning commands have no default method, so a plan never silently switches framework."""
    where = " (as a flag or a 'method' input field)" if from_input else ""
    hint = _METHOD_HINT + (
        ", 'bayesian' to plan a Bayesian analysis." if "bayesian" in choices else "."
    )
    return click.UsageError(
        f"Missing --method{where}. Choose one of: {', '.join(repr(c) for c in choices)}. {hint}"
    )


DURATION_EPILOG = """\b
Examples:
  ablift duration --method sequential --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000
  ablift duration --method fixed --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3
  ablift duration --method fixed --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --expected-split 0.9,0.1
  ablift duration --method bayesian --baseline-rate 0.04 --relative-mde 0.08 --daily-traffic 60000
  ablift duration --method sequential --metric arpu --baseline-mean 12.5 --baseline-sd 25.0 --relative-mde 0.05 --daily-traffic 50000
  ablift duration --method fixed --metric arpu --baseline-mean 12.5 --baseline-cv 2.0 --relative-mde 0.05 --daily-traffic 50000
  ablift duration --input assumptions.csv   # needs a method column (or pass --method)
  echo '{"method": "sequential", "baseline_rate": 0.04, "relative_mde": 0.05, "daily_traffic": 50000}' | ablift duration --input -
  cat assumptions.csv | ablift duration --input - --input-format csv
"""


_BAYESIAN_ONLY = ("prob_threshold", "max_expected_loss", "assurance_target", "max_days")
_METHOD_SCOPED = ("alpha", "power", "sequential_tau", *_BAYESIAN_ONLY)
_CONVERSION_ONLY = ("baseline_rate",)
_ARPU_ONLY = ("baseline_mean", "baseline_sd", "baseline_cv")
_METRIC_SCOPED = (*_CONVERSION_ONLY, *_ARPU_ONLY)


@cli.command(epilog=DURATION_EPILOG)
@click.option(
    "--metric",
    type=click.Choice(["conversion_rate", "arpu"]),
    default=None,
    help="Primary metric to plan for. [default: conversion_rate]",
)
@click.option(
    "--method",
    type=click.Choice(["sequential", "fixed", "bayesian"]),
    default=None,
    help="Planning model; required here or as a 'method' input field. 'sequential' "
    "matches 'analyze --method sequential' (mSPRT), 'fixed' a single planned look, "
    "'bayesian' 'analyze --method bayesian'.",
)
@click.option(
    "--baseline-rate",
    type=float,
    default=None,
    help="Baseline conversion rate, decimal (e.g. 0.04; for --metric conversion_rate).",
)
@click.option(
    "--baseline-mean",
    type=float,
    default=None,
    help="Baseline ARPU mean (for --metric arpu).",
)
@click.option(
    "--baseline-sd",
    type=float,
    default=None,
    help="Baseline ARPU standard deviation (for --metric arpu; mutually exclusive with --baseline-cv).",
)
@click.option(
    "--baseline-cv",
    type=float,
    default=None,
    help="Baseline ARPU coefficient of variation, sd / |mean| (for --metric arpu; mutually exclusive with --baseline-sd).",
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
@click.option(
    "--expected-split",
    type=str,
    default=None,
    help=(
        "Expected traffic split weights in variant order (control first), as comma-separated "
        "positive numbers (e.g. '0.9,0.1' or '80,10,10'). Normalized to sum to 1."
    ),
)
@click.option("--alpha", type=float, default=None, help="Family-wise alpha. [default: 0.05]")
@click.option("--power", type=float, default=None, help="Target power. [default: 0.8]")
@click.option(
    "--sequential-tau",
    type=float,
    default=None,
    help="mSPRT mixing parameter. Defaults to 10% of the baseline rate or mean.",
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
@click.option(
    "--output",
    "output_path",
    required=False,
    type=click.Path(),
    help="Path to output JSON. If omitted, writes to stdout.",
)
@click.option(
    "--input",
    "input_path",
    required=False,
    type=click.Path(exists=True, allow_dash=True),
    help="CSV/XLSX/JSON file or '-' for stdin (metric, baseline_rate / baseline_mean + "
    "baseline_sd|cv, relative_mde, daily_traffic, ...). CLI flags override input values.",
)
@click.option(
    "--input-format",
    type=click.Choice(["json", "csv"]),
    default=None,
    help="Input format for stdin (auto-detected if omitted) or to override file extension.",
)
@click.option("--sheet", required=False, default=None, help="Excel sheet name for duration input.")
def duration(
    metric: str | None,
    method: str | None,
    baseline_rate: float | None,
    baseline_mean: float | None,
    baseline_sd: float | None,
    baseline_cv: float | None,
    relative_mde: float | None,
    daily_traffic: int | None,
    n_variants: int | None,
    expected_split: str | None,
    alpha: float | None,
    power: float | None,
    sequential_tau: float | None,
    prob_threshold: float | None,
    max_expected_loss: float | None,
    assurance_target: float | None,
    max_days: int | None,
    output_path: str | None,
    input_path: str | None,
    input_format: str | None,
    sheet: str | None,
) -> None:
    """Estimate how long an experiment needs to run."""
    if input_format is not None and input_path is None:
        raise click.UsageError("--input-format requires --input.")
    _validate_sheet_option(input_path, input_format, sheet)
    parsed_expected_split = (
        _parse_positional_expected_split(expected_split) if expected_split is not None else None
    )
    flags = {
        "metric": metric,
        "method": method,
        "baseline_rate": baseline_rate,
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "baseline_cv": baseline_cv,
        "relative_mde": relative_mde,
        "daily_traffic": daily_traffic,
        "n_variants": n_variants,
        "expected_split": parsed_expected_split,
        "alpha": alpha,
        "power": power,
        "sequential_tau": sequential_tau,
        "prob_threshold": prob_threshold,
        "max_expected_loss": max_expected_loss,
        "assurance_target": assurance_target,
        "max_days": max_days,
    }
    req: dict[str, Any] = {
        "metric": "conversion_rate",
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
        file_has_both_sd_cv = False
        if input_path:
            file_req = _load_duration_request(
                input_path=input_path,
                input_format=input_format,
                sheet=sheet,
            )
            # When --method or --metric overrides the input's value, settings scoped to the
            # overridden method/metric were written for a different mode, so ignore them.
            method_overridden = method is not None and file_req.get("method", method) != method
            metric_overridden = metric is not None and file_req.get("metric", metric) != metric
            file_keys = set(file_req)
            if method_overridden:
                file_keys -= set(_METHOD_SCOPED)
            if metric_overridden:
                file_keys -= set(_METRIC_SCOPED)
            else:
                file_has_both_sd_cv = "baseline_sd" in file_req and "baseline_cv" in file_req
            req.update(
                {
                    k: v
                    for k, v in file_req.items()
                    if not (method_overridden and k in _METHOD_SCOPED)
                    and not (metric_overridden and k in _METRIC_SCOPED)
                }
            )
        req.update({k: v for k, v in flags.items() if v is not None})
        if baseline_sd is not None and baseline_cv is None:
            req.pop("baseline_cv", None)
        elif baseline_cv is not None and baseline_sd is None:
            req.pop("baseline_sd", None)

        if req["metric"] not in {"conversion_rate", "arpu"}:
            raise click.UsageError(
                f"Unknown metric '{req['metric']}'. Use conversion_rate or arpu."
            )
        if req.get("method") not in {None, "sequential", "fixed", "bayesian"}:
            raise click.UsageError(
                f"Unknown method '{req['method']}'. Use sequential, fixed, or bayesian."
            )

        # Reject settings the chosen metric or planner would silently ignore.
        explicit = {k for k, v in flags.items() if v is not None} | file_keys
        not_applicable_metric = {
            "conversion_rate": _ARPU_ONLY,
            "arpu": _CONVERSION_ONLY,
        }[req["metric"]]
        ignored_metric = [
            f"--{k.replace('_', '-')}" for k in not_applicable_metric if k in explicit
        ]
        if ignored_metric:
            raise click.UsageError(
                f"Not used by --metric {req['metric']}: {', '.join(ignored_metric)} "
                "(set as a flag or input-file column). Remove them or choose a metric that "
                "uses them."
            )

        if req["metric"] == "conversion_rate":
            missing = [
                f"--{k.replace('_', '-')}"
                for k in ("baseline_rate", "relative_mde", "daily_traffic")
                if req.get(k) is None
            ]
            if missing:
                raise click.UsageError(f"Missing required assumptions: {', '.join(missing)}.")
        else:
            if (baseline_sd is not None and baseline_cv is not None) or file_has_both_sd_cv:
                raise click.UsageError("Specify either --baseline-sd or --baseline-cv, not both.")
            missing = []
            if req.get("baseline_mean") is None:
                missing.append("--baseline-mean")
            if req.get("baseline_sd") is None and req.get("baseline_cv") is None:
                missing.append("--baseline-sd (or --baseline-cv)")
            if req.get("relative_mde") is None:
                missing.append("--relative-mde")
            if req.get("daily_traffic") is None:
                missing.append("--daily-traffic")
            if missing:
                raise click.UsageError(f"Missing required assumptions: {', '.join(missing)}.")

        if req.get("method") is None:
            raise _missing_method_error(("sequential", "fixed", "bayesian"), from_input=True)
        not_applicable_method = {
            "bayesian": ("alpha", "power", "sequential_tau"),
            "fixed": ("sequential_tau", *_BAYESIAN_ONLY),
            "sequential": _BAYESIAN_ONLY,
        }[req["method"]]
        ignored_method = [
            f"--{k.replace('_', '-')}" for k in not_applicable_method if k in explicit
        ]
        if ignored_method:
            raise click.UsageError(
                f"Not used by --method {req['method']}: {', '.join(ignored_method)} "
                "(set as a flag or input-file column). Remove them or choose a method that "
                "uses them."
            )

        if req.get("expected_split") is not None:
            if req["n_variants"] < 2:
                raise click.UsageError(
                    "n_variants must be at least 2 (control plus one treatment)."
                )
            try:
                _resolve_split(req["expected_split"], req["n_variants"])
            except ValueError as exc:
                raise click.UsageError(str(exc)) from exc

        if req["metric"] == "conversion_rate":
            common: dict[str, Any] = {
                "baseline_rate": req["baseline_rate"],
                "relative_mde": req["relative_mde"],
                "daily_total_traffic": req["daily_traffic"],
                "n_variants": req["n_variants"],
                "expected_split": req.get("expected_split"),
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
            else:
                plan = bayesian_duration_conversion(
                    **common,
                    prob_threshold=req["prob_threshold"],
                    max_expected_loss=req["max_expected_loss"],
                    assurance_target=req["assurance_target"],
                    max_days=req["max_days"],
                )
        else:
            arpu_common = {
                "baseline_mean": req["baseline_mean"],
                "baseline_sd": req.get("baseline_sd"),
                "baseline_cv": req.get("baseline_cv"),
                "relative_mde": req["relative_mde"],
                "daily_total_traffic": req["daily_traffic"],
                "n_variants": req["n_variants"],
                "expected_split": req.get("expected_split"),
            }
            if req["method"] == "sequential":
                plan = sequential_duration_arpu(
                    **arpu_common,
                    alpha=req["alpha"],
                    power=req["power"],
                    sequential_tau=req.get("sequential_tau"),
                )
            elif req["method"] == "fixed":
                plan = fixed_horizon_duration_arpu(
                    **arpu_common, alpha=req["alpha"], power=req["power"]
                )
            else:
                plan = bayesian_duration_arpu(
                    **arpu_common,
                    prob_threshold=req["prob_threshold"],
                    max_expected_loss=req["max_expected_loss"],
                    assurance_target=req["assurance_target"],
                    max_days=req["max_days"],
                )
    except UsageOptionError as exc:
        raise click.UsageError(str(exc)) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    except MemoryError as exc:
        raise click.ClickException(_OUT_OF_MEMORY) from exc

    plan_json = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "method": plan.method,
            "estimated_days": plan.estimated_days,
            "n_per_variant": plan.n_per_variant,
            "n_by_variant": plan.n_by_variant,
            "n_total": plan.n_total,
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


POWER_EPILOG = """\b
Examples:
  ablift power --method fixed --baseline-rate 0.08 --n-per-variant 10000 --relative-mde 0.1
  ablift power --method fixed --baseline-rate 0.08 --n-per-variant 10000
  ablift power --method fixed --baseline-rate 0.08 --n-total 20000 --expected-split 0.9,0.1 --relative-mde 0.1
  ablift power --method fixed --metric arpu --baseline-mean 12.5 --baseline-sd 25 --n-per-variant 20000
  ablift power --method sequential --baseline-rate 0.04 --n-per-variant 50000 --relative-mde 0.1
"""


@cli.command(epilog=POWER_EPILOG)
@click.option(
    "--metric",
    type=click.Choice(["conversion_rate", "arpu"]),
    default="conversion_rate",
    show_default=True,
)
@click.option(
    "--method",
    type=click.Choice(["fixed", "sequential"]),
    default=None,
    help="Required. 'fixed' is the classic single-look test; 'sequential' matches mSPRT analysis.",
)
@click.option(
    "--n-per-variant",
    type=int,
    default=None,
    help="Users per variant (equal split; mutually exclusive with --n-total).",
)
@click.option(
    "--n-total",
    type=int,
    default=None,
    help="Total users across all variants (required with --expected-split; mutually exclusive with --n-per-variant).",
)
@click.option(
    "--expected-split",
    type=str,
    default=None,
    help=(
        "Expected traffic split weights in variant order (control first), as comma-separated "
        "positive numbers (e.g. '0.9,0.1' or '80,10,10'). Use with --n-total."
    ),
)
@click.option(
    "--relative-mde",
    type=float,
    default=None,
    help="Relative lift to detect (e.g. 0.1 for +10%). Give it to get the power.",
)
@click.option(
    "--power",
    "target_power",
    type=float,
    default=None,
    help="Target power. Give it (or neither) to get the minimum detectable lift. [default: 0.8]",
)
@click.option("--baseline-rate", type=float, default=None, help="Baseline conversion rate.")
@click.option("--baseline-mean", type=float, default=None, help="Baseline ARPU mean.")
@click.option("--baseline-sd", type=float, default=None, help="Baseline ARPU standard deviation.")
@click.option("--baseline-cv", type=float, default=None, help="Baseline ARPU sd / mean.")
@click.option("--n-variants", type=int, default=2, show_default=True, help="Including control.")
@click.option("--alpha", type=float, default=0.05, show_default=True, help="Family-wise alpha.")
@click.option(
    "--sequential-tau",
    type=float,
    default=None,
    help="mSPRT mixing parameter (sequential only). Defaults to 10% of the baseline.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Path to output JSON. If omitted, writes to stdout.",
)
def power(
    metric: str,
    method: str | None,
    n_per_variant: int | None,
    n_total: int | None,
    expected_split: str | None,
    relative_mde: float | None,
    target_power: float | None,
    baseline_rate: float | None,
    baseline_mean: float | None,
    baseline_sd: float | None,
    baseline_cv: float | None,
    n_variants: int,
    alpha: float,
    sequential_tau: float | None,
    output_path: str | None,
) -> None:
    """Power for a given sample size, or the smallest lift it can detect."""
    if method is None:
        raise _missing_method_error(("sequential", "fixed"))
    if (n_per_variant is None) == (n_total is None):
        raise click.UsageError("Give exactly one of --n-per-variant or --n-total.")
    parsed_expected_split = (
        _parse_positional_expected_split(expected_split) if expected_split is not None else None
    )
    if parsed_expected_split is not None:
        if n_per_variant is not None:
            raise click.UsageError("Use --n-total (not --n-per-variant) with --expected-split.")
        if n_variants < 2:
            raise click.UsageError("n_variants must be at least 2 (control plus one treatment).")
        try:
            _resolve_split(parsed_expected_split, n_variants)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
    if relative_mde is not None and target_power is not None:
        raise click.UsageError(
            "Give --relative-mde to get the power, or --power to get the detectable lift, not both."
        )
    if sequential_tau is not None and method != "sequential":
        raise click.UsageError("--sequential-tau only applies to --method sequential.")
    wrong = (
        [
            ("--baseline-mean", baseline_mean),
            ("--baseline-sd", baseline_sd),
            ("--baseline-cv", baseline_cv),
        ]
        if metric == "conversion_rate"
        else [("--baseline-rate", baseline_rate)]
    )
    unused = [flag for flag, value in wrong if value is not None]
    if unused:
        raise click.UsageError(f"Not used by --metric {metric}: {', '.join(unused)}.")
    if metric == "conversion_rate" and baseline_rate is None:
        raise click.UsageError("Missing --baseline-rate.")
    if metric == "arpu" and (
        baseline_mean is None or (baseline_sd is None) == (baseline_cv is None)
    ):
        raise click.UsageError(
            "--metric arpu needs --baseline-mean and one of --baseline-sd or --baseline-cv."
        )

    try:
        result = power_analysis(
            metric=metric,
            method=method,
            n_per_variant=n_per_variant,
            n_total=n_total,
            expected_split=parsed_expected_split,
            relative_mde=relative_mde,
            power=None if relative_mde is not None else (target_power or 0.8),
            baseline_rate=baseline_rate,
            baseline_mean=baseline_mean,
            baseline_sd=baseline_sd,
            baseline_cv=baseline_cv,
            n_variants=n_variants,
            alpha=alpha,
            sequential_tau=sequential_tau,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    rendered = json.dumps({"schema_version": SCHEMA_VERSION, **asdict(result)}, indent=2)
    if output_path:
        _write_text(output_path, rendered + "\n")
    else:
        click.echo(rendered)


@cli.command("example-input")
def example_input() -> None:
    """Print an example input JSON to stdout."""
    click.echo(json.dumps(example_payload(), indent=2))


SCHEMA_EPILOG = """\b
Examples:
  ablift schema analyze-input
  ablift schema analyze-output
  ablift schema duration-input
  ablift schema duration-output
  ablift schema power-output
  ablift schema --all
  ablift schema analyze-input --output analyze-input.schema.json
"""


@cli.command(epilog=SCHEMA_EPILOG)
@click.argument("target", required=False, type=click.Choice(list(SCHEMA_NAMES)))
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Emit all input and output JSON Schemas in a single object.",
)
@click.option(
    "--output",
    "output_path",
    required=False,
    type=click.Path(),
    help="Path to write JSON Schema output. If omitted, writes to stdout.",
)
def schema(target: str | None, show_all: bool, output_path: str | None) -> None:
    """Print JSON Schema (Draft 2020-12) for ablift input and output contracts."""
    if show_all and target is not None:
        raise click.UsageError("Specify either a schema target or --all, not both.")
    if not show_all and target is None:
        names = ", ".join(SCHEMA_NAMES)
        raise click.UsageError(f"Specify a schema target ({names}) or pass --all.")

    payload = get_all_schemas() if show_all else get_schema(target or "")
    rendered = json.dumps(payload, indent=2)
    if output_path:
        _write_text(output_path, rendered + "\n")
    else:
        click.echo(rendered)


if __name__ == "__main__":
    cli()
