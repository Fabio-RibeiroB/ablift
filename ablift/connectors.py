from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import load_workbook

from .stats import bootstrap_mean_diff_ci, row_level_tail_diagnostics, winsorize_pooled


class UsageOptionError(ValueError):
    """Raised when a CLI/analysis option is incompatible with the input shape or metric."""


def read_csv_text(text: str) -> list[dict[str, Any]]:
    # Strip a leading UTF-8 BOM if present (e.g. Excel "CSV UTF-8" piped via stdin).
    cleaned = text.lstrip("\ufeff")
    try:
        return [
            row
            for row in csv.DictReader(io.StringIO(cleaned, newline=""))
            if not _is_blank_row(row)
        ]
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV input: {exc}") from exc


def read_table(path: str, sheet: str | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        # utf-8-sig strips the byte-order mark Excel adds when saving "CSV UTF-8".
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            try:
                return [row for row in csv.DictReader(f) if not _is_blank_row(row)]
            except csv.Error as exc:
                raise ValueError(f"Malformed CSV input: {exc}") from exc

    if suffix in {".xlsx", ".xlsm"}:
        wb = load_workbook(filename=path, read_only=True, data_only=True)
        if sheet is not None and sheet not in wb.sheetnames:
            raise ValueError(
                f"Sheet '{sheet}' not found. Available sheets: {', '.join(wb.sheetnames)}."
            )
        ws = wb[sheet] if sheet else wb.active
        if ws is None:
            raise ValueError("Workbook has no active sheet; pass --sheet.")
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [str(h).strip() if h is not None else "" for h in rows[0]]
        out: list[dict[str, Any]] = []
        for row in rows[1:]:
            obj: dict[str, Any] = {}
            for i, key in enumerate(headers):
                if not key:
                    continue
                obj[key] = row[i] if i < len(row) else None
            if not _is_blank_row(obj):
                out.append(obj)
        return out

    raise ValueError("Unsupported file type. Use .csv or .xlsx/.xlsm")


def _is_blank_row(row: dict[str, Any]) -> bool:
    return all(v is None or str(v).strip() == "" for v in row.values())


def _to_int(value: Any, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"Missing required integer field '{field_name}'.")
    if isinstance(value, bool):
        raise ValueError(f"Column '{field_name}' must be numeric, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Column '{field_name}' must be numeric, got {value!r}.") from None
    if not number.is_integer():
        raise ValueError(f"Column '{field_name}' must be a whole number, got {value!r}.")
    return int(number)


def _to_float_or_none(value: Any, field_name: str = "value") -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Column '{field_name}' must be numeric, got {value!r}.") from None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "control"}:
        return True
    try:
        # Numeric tokens such as "1.0" (common in exported CSVs) or 1.0 from XLSX.
        return float(text) == 1.0
    except ValueError:
        return False


_BOOL_STRINGS = {"0", "1", "true", "false", "yes", "no", "y", "n"}


def _is_binary_col(values: list[Any]) -> bool:
    """Return True if all non-None values are 0, 1, boolean, or boolean-string."""
    for v in values:
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            continue
        if str(v).strip().lower() in _BOOL_STRINGS:
            continue
        try:
            f = float(v)
            if f not in (0.0, 1.0):
                return False
        except (ValueError, TypeError):
            return False
    return True


_BOOL_WORDS = {"true", "false", "yes", "no", "y", "n"}


def _is_boolean_word_col(values: list[Any]) -> bool:
    """True if every non-empty value is a bool or a boolean word (not a number)."""
    seen = False
    for v in values:
        if v is None or v == "":
            continue
        if isinstance(v, bool) or str(v).strip().lower() in _BOOL_WORDS:
            seen = True
            continue
        return False
    return seen


def _find_control_column(rows: list[dict[str, Any]], skip_cols: int) -> str | None:
    """Return the name of a boolean-valued column beyond the first skip_cols positional columns.

    A boolean-valued column (true/false, yes/no or 1/0) whose header mentions "control" is
    always the control column, so an all-false column is reported as "no control" instead of
    being read as a metric. Without that header hint, the column must hold boolean words
    (true/false, yes/no, y/n) with at least one true and one false value.
    Columns are searched from the right so that numeric metric columns are not mistaken for it.
    """
    if not rows:
        return None
    all_cols = list(rows[0].keys())
    candidates = list(reversed(all_cols[skip_cols:]))
    for col in candidates:
        if "control" in str(col).lower() and _is_binary_col([row.get(col) for row in rows]):
            return col
    for col in candidates:
        values = [row.get(col) for row in rows]
        # Without a header hint, only boolean words count: numeric 0/1 is too easily a real
        # metric (e.g. a tiny revenue_sum_squares) to be silently reinterpreted.
        if not _is_boolean_word_col(values):
            continue
        has_true = any(_to_bool(v) for v in values)
        has_false = any(not _to_bool(v) for v in values if v not in (None, ""))
        if has_true and has_false:
            return col
    return None


_CONTROL_FLAG_HEADER = re.compile(r"^(is[ _-]?)?control([ _-]?(group|flag|arm|variant))?$", re.I)


def _reject_misplaced_control_column(rows: list[dict[str, Any]], value_positions: int) -> None:
    """Raise if a control flag column sits where a value column is expected.

    Inputs are read by position, so ``name, is_control, visitors, conversions`` would read
    the flag as visitor counts. Catch the header-hinted case and say where the column goes.
    """
    cols = list(rows[0].keys())
    for col in cols[1 : 1 + value_positions]:
        # Only flag-like headers (is_control, control, control_group, ...): a metric such as
        # control_conversions may legitimately hold 0/1 values.
        if _CONTROL_FLAG_HEADER.match(str(col).strip()) and _is_binary_col(
            [row.get(col) for row in rows]
        ):
            raise ValueError(
                f"Column '{col}' looks like the control flag, but columns are read by position "
                "and the control flag must be the last column. Move it to the end, e.g. "
                "name, visitors, conversions, is_control (aggregated) or "
                "name, outcome, is_control (row-level).\n" + AGGREGATED_LAYOUT_HELP
            )


def _to_finite_float(value: Any, field_name: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Missing required numeric field '{field_name}'.")
    if isinstance(value, bool):
        raise ValueError(f"Column '{field_name}' must be numeric, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Column '{field_name}' must be numeric, got {value!r}.") from None
    if not math.isfinite(number):
        raise ValueError(f"Column '{field_name}' must be finite, got {value!r}.")
    return number


def detect_input_shape(rows: list[dict[str, Any]], primary_metric: str | None = None) -> str:
    """Return 'row_level' or 'aggregated' for a positional table.

    Repeated variant names mean row-level data, whose layout must be
    ``name, outcome [, is_control]`` (or ``name, numerator, denominator [, is_control]``
    when ``primary_metric == 'ratio'``). A table with repeated names and more columns than that
    is an aggregated table with duplicate rows, which is rejected rather than misread.
    """
    if not rows:
        raise ValueError("Input table is empty.")
    cols = list(rows[0].keys())
    names = [str(row.get(cols[0], "")).strip() for row in rows]
    names = [n for n in names if n]
    if not names:
        raise ValueError("Input has no variant data: the first column (variant name) is empty.")
    if primary_metric == "ratio":
        _reject_misplaced_control_column(rows, value_positions=2)
        if len(set(names)) == len(names):
            return "aggregated"
        control_col = _find_control_column(rows, skip_cols=3)
        data_cols = [c for c in cols if c != control_col]
        if len(data_cols) == 3:
            return "row_level"
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"Variant name(s) {', '.join(duplicates)} appear in more than one row, but the table "
            f"has {len(data_cols) - 1} value columns. Row-level ratio input must be exactly: "
            "name, numerator, denominator [, is_control]. For aggregated ratio input, combine rows "
            "so each variant appears once.\n" + AGGREGATED_LAYOUT_HELP
        )

    _reject_misplaced_control_column(rows, value_positions=1)
    if len(set(names)) == len(names):
        return "aggregated"

    control_col = _find_control_column(rows, skip_cols=2)
    data_cols = [c for c in cols if c != control_col]
    if len(data_cols) <= 2:
        return "row_level"
    duplicates = sorted({n for n in names if names.count(n) > 1})
    raise ValueError(
        f"Variant name(s) {', '.join(duplicates)} appear in more than one row, but the table "
        f"has {len(data_cols) - 1} value columns. Row-level input must be exactly: "
        "name, outcome [, is_control]. For aggregated input, combine rows so each variant "
        "appears once.\n" + AGGREGATED_LAYOUT_HELP
    )


def _build_payload_from_ratio_row_level(
    rows: list[dict[str, Any]], defaults: dict[str, Any]
) -> dict[str, Any]:
    if defaults.get("winsorize") is not None:
        raise UsageOptionError("--winsorize is only supported for row-level ARPU data.")
    if defaults.get("bootstrap_samples") is not None:
        raise UsageOptionError("--bootstrap-samples is only supported for row-level ARPU data.")
    cols = list(rows[0].keys())
    if len(cols) < 3:
        raise ValueError(
            "Row-level ratio input requires at least 3 columns: variant name, numerator, denominator."
        )
    variant_col = cols[0]
    num_col = cols[1]
    den_col = cols[2]
    control_col = _find_control_column(rows, skip_cols=3)
    data_cols = [c for c in cols if c != control_col]
    if len(data_cols) != 3:
        raise ValueError(
            "Row-level ratio input must have layout: name, numerator, denominator [, is_control]."
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = str(row.get(variant_col, "")).strip()
        if not name:
            continue
        groups.setdefault(name, []).append(row)

    if not groups:
        raise ValueError("Row-level input has no variant data.")

    first_variant = next(iter(groups))
    variants: list[dict[str, Any]] = []
    for name, group_rows in groups.items():
        if control_col:
            is_ctrl = any(_to_bool(r.get(control_col)) for r in group_rows)
        else:
            is_ctrl = name == first_variant

        num_sum = 0.0
        num_sum_sq = 0.0
        den_sum = 0.0
        den_sum_sq = 0.0
        num_den_sum = 0.0
        for row in group_rows:
            num = _to_finite_float(row.get(num_col), num_col)
            den = _to_finite_float(row.get(den_col), den_col)
            if den < 0.0:
                raise ValueError(
                    f"Variant {name!r}: denominator column '{den_col}' cannot be negative "
                    f"(got {den!r})."
                )
            num_sum += num
            num_sum_sq += num * num
            den_sum += den
            den_sum_sq += den * den
            num_den_sum += num * den

        variants.append(
            {
                "name": name,
                "is_control": is_ctrl,
                "visitors": len(group_rows),
                "numerator_sum": num_sum,
                "numerator_sum_squares": num_sum_sq,
                "denominator_sum": den_sum,
                "denominator_sum_squares": den_sum_sq,
                "numerator_denominator_sum": num_den_sum,
            }
        )

    column_roles: dict[str, str] = {
        "variant": variant_col,
        "numerator": num_col,
        "denominator": den_col,
    }
    if control_col:
        column_roles["is_control"] = control_col
    return _build_final_payload(
        variants, "ratio", defaults, source_type="row_level", column_roles=column_roles
    )


def _build_payload_from_row_level(
    rows: list[dict[str, Any]], defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    defaults = defaults or {}
    if defaults.get("primary_metric") == "ratio":
        return _build_payload_from_ratio_row_level(rows, defaults)
    cols = list(rows[0].keys())
    if len(cols) < 2:
        raise ValueError("Row-level input requires at least 2 columns: variant name and outcome.")
    variant_col = cols[0]
    outcome_col = cols[1]
    control_col = _find_control_column(rows, skip_cols=2)

    # Group rows by variant, preserving first-appearance order
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = str(row.get(variant_col, "")).strip()
        if not name:
            continue
        if name not in groups:
            groups[name] = []
        groups[name].append(row)

    if not groups:
        raise ValueError("Row-level input has no variant data.")

    all_outcomes = [row.get(outcome_col) for row in rows if row.get(outcome_col) not in (None, "")]
    is_conversion = _is_binary_col(all_outcomes)
    primary_metric = defaults.get("primary_metric") or (
        "conversion_rate" if is_conversion else "arpu"
    )

    if primary_metric == "arpu" and is_conversion:
        raise ValueError("Row-level outcome column is binary; use primary_metric conversion_rate.")
    if primary_metric == "conversion_rate" and not is_conversion:
        raise ValueError(
            "Row-level outcome column is continuous; conversion_rate needs 0/1 outcomes."
        )

    winsorize = defaults.get("winsorize")
    bootstrap_samples = defaults.get("bootstrap_samples")
    if primary_metric != "arpu":
        if winsorize is not None:
            raise UsageOptionError("--winsorize is only supported for row-level ARPU data.")
        if bootstrap_samples is not None:
            raise UsageOptionError("--bootstrap-samples is only supported for row-level ARPU data.")

    first_variant = next(iter(groups))
    is_control_by_name: dict[str, bool] = {}
    for name, group_rows in groups.items():
        if control_col:
            is_control_by_name[name] = any(_to_bool(r.get(control_col)) for r in group_rows)
        else:
            is_control_by_name[name] = name == first_variant

    extra_interpretation: dict[str, Any] = {}
    variants: list[dict[str, Any]] = []
    if is_conversion:
        for name, group_rows in groups.items():
            variants.append(
                {
                    "name": name,
                    "is_control": is_control_by_name[name],
                    "visitors": len(group_rows),
                    "conversions": sum(_to_bool(r.get(outcome_col)) for r in group_rows),
                }
            )
    else:
        raw_arrays: dict[str, np.ndarray] = {}
        for name, group_rows in groups.items():
            values = [
                _to_finite_float(v, outcome_col)
                for r in group_rows
                if (v := r.get(outcome_col)) not in (None, "")
            ]
            raw_arrays[name] = np.asarray(values, dtype=float)

        if winsorize is not None:
            arrays, winsor_diag = winsorize_pooled(raw_arrays, winsorize)
            extra_interpretation["winsorization"] = winsor_diag
        else:
            arrays = raw_arrays

        extra_interpretation["arpu_tail_diagnostics"] = row_level_tail_diagnostics(arrays)

        for name, arr in arrays.items():
            variants.append(
                {
                    "name": name,
                    "is_control": is_control_by_name[name],
                    "visitors": int(arr.size),
                    "conversions": int(np.sum(arr > 0.0)),
                    "revenue_sum": float(np.sum(arr)),
                    "revenue_sum_squares": float(np.sum(arr * arr)),
                }
            )

        if bootstrap_samples is not None:
            control_names = [n for n, is_c in is_control_by_name.items() if is_c]
            treatment_names = [n for n, is_c in is_control_by_name.items() if not is_c]
            if len(control_names) == 1 and treatment_names:
                method = defaults.get("method") or "bayesian"
                raw_alpha = defaults.get("alpha")
                alpha = 0.05 if raw_alpha is None else float(raw_alpha)
                if method in {"fixed", "sequential"}:
                    alpha_boot = alpha / max(len(treatment_names), 1)
                else:
                    alpha_boot = 0.05
                raw_seed = defaults.get("random_seed")
                base_seed = 7 if raw_seed is None else int(raw_seed)
                control_arr = arrays[control_names[0]]
                boot_ci: dict[str, dict[str, Any]] = {}
                for idx, t_name in enumerate(treatment_names):
                    lo, hi = bootstrap_mean_diff_ci(
                        control_arr,
                        arrays[t_name],
                        n_bootstrap=bootstrap_samples,
                        alpha=alpha_boot,
                        seed=base_seed + idx,
                    )
                    boot_ci[t_name] = {
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_bootstrap": int(bootstrap_samples),
                        "confidence_level": float(1.0 - alpha_boot),
                    }
                extra_interpretation["bootstrap_ci"] = boot_ci

    column_roles = {"variant": variant_col, "outcome": outcome_col}
    if control_col:
        column_roles["is_control"] = control_col
    return _build_final_payload(
        variants,
        primary_metric,
        defaults,
        source_type="row_level",
        column_roles=column_roles,
        extra_interpretation=extra_interpretation,
    )


AGGREGATED_LAYOUT_HELP = (
    "Aggregated input layouts (one row per variant, columns by position):\n"
    "  name, visitors, conversions [, is_control]\n"
    "  name, visitors, revenue_sum, revenue_sum_squares [, is_control]\n"
    "  name, visitors, conversions, revenue_sum, revenue_sum_squares [, is_control]\n"
    "  name, visitors, numerator_sum, numerator_sum_squares, denominator_sum, "
    "denominator_sum_squares, numerator_denominator_sum [, is_control]"
)


def _build_payload_from_aggregated(
    rows: list[dict[str, Any]], defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    defaults = defaults or {}
    if defaults.get("winsorize") is not None:
        raise UsageOptionError(
            "--winsorize requires row-level ARPU data (aggregated inputs have no individual rows to cap)."
        )
    if defaults.get("bootstrap_samples") is not None:
        raise UsageOptionError(
            "--bootstrap-samples requires row-level ARPU data (aggregated inputs have no individual rows to resample)."
        )
    cols = list(rows[0].keys())
    if len(cols) < 3:
        raise ValueError(
            "Aggregated input requires at least 3 columns: variant name, visitors, metric.\n"
            + AGGREGATED_LAYOUT_HELP
        )
    _reject_misplaced_control_column(rows, value_positions=2)
    variant_col = cols[0]
    visitors_col = cols[1]
    control_col = _find_control_column(rows, skip_cols=3)
    metric_cols = [c for c in cols[2:] if c != control_col]

    conversions_col: str | None = None
    revenue_cols: tuple[str, str] | None = None
    ratio_cols: tuple[str, str, str, str, str] | None = None
    if len(metric_cols) == 1:
        conversions_col = metric_cols[0]
        for row in rows:
            value = _to_float_or_none(row.get(conversions_col), conversions_col)
            visitors = _to_float_or_none(row.get(visitors_col), visitors_col)
            if value is None or visitors is None:
                continue
            if value != int(value) or value > visitors:
                raise ValueError(
                    f"Column '{conversions_col}' looks like revenue, not conversions. ARPU from "
                    "aggregated data also needs a revenue_sum_squares column to estimate "
                    "variance; alternatively provide row-level data.\n" + AGGREGATED_LAYOUT_HELP
                )
    elif len(metric_cols) == 2:
        revenue_cols = (metric_cols[0], metric_cols[1])
    elif len(metric_cols) == 3:
        conversions_col = metric_cols[0]
        revenue_cols = (metric_cols[1], metric_cols[2])
    elif len(metric_cols) == 5:
        ratio_cols = (
            metric_cols[0],
            metric_cols[1],
            metric_cols[2],
            metric_cols[3],
            metric_cols[4],
        )
    else:
        col_list = ", ".join(str(c) for c in metric_cols)
        raise ValueError(
            f"Could not interpret {len(metric_cols)} metric columns ({col_list}).\n"
            + AGGREGATED_LAYOUT_HELP
        )

    inferred_metric = (
        "ratio"
        if ratio_cols is not None
        else ("arpu" if revenue_cols is not None else "conversion_rate")
    )
    primary_metric = defaults.get("primary_metric") or inferred_metric
    if primary_metric == "arpu" and revenue_cols is None:
        raise ValueError(
            "primary_metric 'arpu' needs revenue_sum and revenue_sum_squares columns.\n"
            + AGGREGATED_LAYOUT_HELP
        )
    if primary_metric == "conversion_rate" and conversions_col is None:
        raise ValueError(
            "primary_metric 'conversion_rate' needs a conversions column.\n"
            + AGGREGATED_LAYOUT_HELP
        )
    if primary_metric == "ratio" and ratio_cols is None:
        raise ValueError(
            "primary_metric 'ratio' needs numerator_sum, numerator_sum_squares, denominator_sum, "
            "denominator_sum_squares, and numerator_denominator_sum columns.\n"
            + AGGREGATED_LAYOUT_HELP
        )

    variants: list[dict[str, Any]] = []
    first_row = True
    for row in rows:
        name = str(row.get(variant_col, "")).strip()
        if not name:
            continue
        is_ctrl = _to_bool(row.get(control_col)) if control_col else first_row
        first_row = False

        variant_obj: dict[str, Any] = {
            "name": name,
            "is_control": is_ctrl,
            "visitors": _to_int(row.get(visitors_col), visitors_col),
        }
        if conversions_col is not None:
            variant_obj["conversions"] = _to_int(row.get(conversions_col), conversions_col)
        if revenue_cols is not None:
            variant_obj["revenue_sum"] = _to_float_or_none(
                row.get(revenue_cols[0]), revenue_cols[0]
            )
            variant_obj["revenue_sum_squares"] = _to_float_or_none(
                row.get(revenue_cols[1]), revenue_cols[1]
            )
        if ratio_cols is not None:
            variant_obj["numerator_sum"] = _to_finite_float(row.get(ratio_cols[0]), ratio_cols[0])
            variant_obj["numerator_sum_squares"] = _to_finite_float(
                row.get(ratio_cols[1]), ratio_cols[1]
            )
            variant_obj["denominator_sum"] = _to_finite_float(row.get(ratio_cols[2]), ratio_cols[2])
            variant_obj["denominator_sum_squares"] = _to_finite_float(
                row.get(ratio_cols[3]), ratio_cols[3]
            )
            variant_obj["numerator_denominator_sum"] = _to_finite_float(
                row.get(ratio_cols[4]), ratio_cols[4]
            )
        variants.append(variant_obj)

    column_roles: dict[str, str] = {"variant": variant_col, "visitors": visitors_col}
    if conversions_col:
        column_roles["conversions"] = conversions_col
    if revenue_cols:
        column_roles["revenue_sum"] = revenue_cols[0]
        column_roles["revenue_sum_squares"] = revenue_cols[1]
    if ratio_cols:
        column_roles["numerator_sum"] = ratio_cols[0]
        column_roles["numerator_sum_squares"] = ratio_cols[1]
        column_roles["denominator_sum"] = ratio_cols[2]
        column_roles["denominator_sum_squares"] = ratio_cols[3]
        column_roles["numerator_denominator_sum"] = ratio_cols[4]
    if control_col:
        column_roles["is_control"] = control_col

    return _build_final_payload(
        variants, primary_metric, defaults, source_type="aggregated", column_roles=column_roles
    )


def _build_final_payload(
    variants: list[dict[str, Any]],
    primary_metric: str,
    defaults: dict[str, Any],
    source_type: str,
    column_roles: dict[str, str],
    extra_interpretation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    interpretation: dict[str, Any] = {
        "source_type": source_type,
        "variant_count": len(variants),
        "column_roles": column_roles,
    }
    if extra_interpretation:
        interpretation.update(extra_interpretation)
    return {
        "experiment_name": defaults.get("experiment_name", "table_input_experiment"),
        "method": defaults.get("method", "bayesian"),
        "primary_metric": primary_metric,
        "alpha": defaults.get("alpha", 0.05),
        "sequential_tau": defaults.get("sequential_tau"),
        "decision_thresholds": defaults.get("decision_thresholds"),
        "decision_policy": defaults.get("decision_policy"),
        "guardrails": defaults.get("guardrails", []),
        "samples": defaults.get("samples", 50000),
        "random_seed": defaults.get("random_seed", 7),
        "expected_split": defaults.get("expected_split"),
        "variants": variants,
        "input_interpretation": interpretation,
    }


def build_payload_from_rows(
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_metric = (defaults or {}).get("primary_metric")
    shape = detect_input_shape(rows, primary_metric=primary_metric)
    if shape == "row_level":
        return _build_payload_from_row_level(rows, defaults)
    return _build_payload_from_aggregated(rows, defaults)


_DURATION_FIELDS: dict[str, type] = {
    "metric": str,
    "method": str,
    "baseline_rate": float,
    "baseline_mean": float,
    "baseline_sd": float,
    "baseline_cv": float,
    "relative_mde": float,
    "daily_traffic": int,
    "n_variants": int,
    "alpha": float,
    "power": float,
    "sequential_tau": float,
    "prob_threshold": float,
    "max_expected_loss": float,
    "assurance_target": float,
    "max_days": int,
}


def _parse_duration_mapping(
    raw: dict[Any, Any],
    *,
    ignore_empty_keys: bool,
    key_label: str,
) -> dict[str, Any]:
    row = {
        str(k).strip().lower(): v
        for k, v in raw.items()
        if k is not None and (not ignore_empty_keys or str(k).strip() != "")
    }
    unknown = sorted(k for k in row if k not in _DURATION_FIELDS)
    if unknown:
        formatted = ", ".join(repr(k) if not k else k for k in unknown)
        raise ValueError(
            f"Unknown duration {key_label}(s): {formatted}. "
            f"Expected any of: {', '.join(_DURATION_FIELDS)}."
        )
    out: dict[str, Any] = {}
    for key, kind in _DURATION_FIELDS.items():
        value = row.get(key)
        if value is None or value == "":
            continue
        if kind is str:
            if not isinstance(value, str):
                raise ValueError(f"Field '{key}' must be a string, got {value!r}.")
            stripped = value.strip().lower()
            if not stripped:
                continue
            out[key] = stripped
        elif kind is int:
            out[key] = _to_int(value, key)
        else:
            if isinstance(value, bool):
                raise ValueError(f"Field '{key}' must be numeric, got {value!r}.")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Field '{key}' must be numeric, got {value!r}.") from None
            if not math.isfinite(number):
                raise ValueError(f"Field '{key}' must be a finite number, got {value!r}.")
            out[key] = number
    return out


def build_duration_request_from_dict(data: Any) -> dict[str, Any]:
    """Read duration assumptions from a JSON object with named fields."""
    if not isinstance(data, dict):
        raise ValueError("Duration JSON input must be a JSON object.")
    return _parse_duration_mapping(data, ignore_empty_keys=False, key_label="field")


def build_duration_request_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Read duration assumptions from the first row of a table with named columns.

    Only columns that are present and non-empty are returned; callers supply defaults.
    """
    if not rows:
        raise ValueError("Duration input table is empty.")
    return _parse_duration_mapping(rows[0], ignore_empty_keys=True, key_label="column")
