from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


def read_table(path: str, sheet: str | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        # utf-8-sig strips the byte-order mark Excel adds when saving "CSV UTF-8".
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return [row for row in csv.DictReader(f) if not _is_blank_row(row)]

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
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"Column '{field_name}' must be numeric, got {value!r}.") from None


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
    return text in {"1", "true", "yes", "y", "control"}


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


def _find_control_column(rows: list[dict[str, Any]], skip_cols: int) -> str | None:
    """Return the name of a boolean-valued column beyond the first skip_cols positional columns.

    A boolean-valued column whose header mentions "control" is always the control column (so
    a column of all-false values is reported as "no control" instead of being read as a
    metric). Otherwise a boolean column must contain at least one true and one false value.
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
        if not _is_binary_col(values):
            continue
        has_true = any(_to_bool(v) for v in values)
        has_false = any(not _to_bool(v) for v in values if v not in (None, ""))
        if has_true and has_false:
            return col
    return None


def detect_input_shape(rows: list[dict[str, Any]]) -> str:
    """Return 'row_level' or 'aggregated' for a positional table.

    Repeated variant names mean row-level data, whose layout must be
    ``name, outcome [, is_control]``. A table with repeated names and more columns than that
    is an aggregated table with duplicate rows, which is rejected rather than misread.
    """
    if not rows:
        raise ValueError("Input table is empty.")
    cols = list(rows[0].keys())
    names = [str(row.get(cols[0], "")).strip() for row in rows]
    names = [n for n in names if n]
    if not names:
        raise ValueError("Input has no variant data: the first column (variant name) is empty.")
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


def _build_payload_from_row_level(
    rows: list[dict[str, Any]], defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    defaults = defaults or {}
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

    first_variant = next(iter(groups))
    variants: list[dict[str, Any]] = []
    for name, group_rows in groups.items():
        if control_col:
            is_ctrl = any(_to_bool(r.get(control_col)) for r in group_rows)
        else:
            is_ctrl = name == first_variant

        variant_obj: dict[str, Any] = {"name": name, "is_control": is_ctrl}
        if is_conversion:
            variant_obj["visitors"] = len(group_rows)
            variant_obj["conversions"] = sum(_to_bool(r.get(outcome_col)) for r in group_rows)
        else:
            values = [float(v) for r in group_rows if (v := r.get(outcome_col)) not in (None, "")]
            variant_obj["visitors"] = len(values)
            variant_obj["conversions"] = sum(1 for v in values if v > 0)
            variant_obj["revenue_sum"] = sum(values)
            variant_obj["revenue_sum_squares"] = sum(v * v for v in values)
        variants.append(variant_obj)

    if primary_metric == "arpu" and is_conversion:
        raise ValueError("Row-level outcome column is binary; use primary_metric conversion_rate.")
    if primary_metric == "conversion_rate" and not is_conversion:
        raise ValueError(
            "Row-level outcome column is continuous; conversion_rate needs 0/1 outcomes."
        )

    column_roles = {"variant": variant_col, "outcome": outcome_col}
    if control_col:
        column_roles["is_control"] = control_col
    return _build_final_payload(
        variants, primary_metric, defaults, source_type="row_level", column_roles=column_roles
    )


AGGREGATED_LAYOUT_HELP = (
    "Aggregated input layouts (one row per variant, columns by position):\n"
    "  name, visitors, conversions [, is_control]\n"
    "  name, visitors, revenue_sum, revenue_sum_squares [, is_control]\n"
    "  name, visitors, conversions, revenue_sum, revenue_sum_squares [, is_control]"
)


def _build_payload_from_aggregated(
    rows: list[dict[str, Any]], defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    defaults = defaults or {}
    cols = list(rows[0].keys())
    if len(cols) < 3:
        raise ValueError(
            "Aggregated input requires at least 3 columns: variant name, visitors, metric.\n"
            + AGGREGATED_LAYOUT_HELP
        )
    variant_col = cols[0]
    visitors_col = cols[1]
    control_col = _find_control_column(rows, skip_cols=3)
    metric_cols = [c for c in cols[2:] if c != control_col]

    conversions_col: str | None = None
    revenue_cols: tuple[str, str] | None = None
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
    else:
        raise ValueError(
            f"Could not interpret {len(metric_cols)} metric columns ({', '.join(metric_cols)}).\n"
            + AGGREGATED_LAYOUT_HELP
        )

    primary_metric = defaults.get("primary_metric") or (
        "arpu" if revenue_cols is not None else "conversion_rate"
    )
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
        variants.append(variant_obj)

    column_roles: dict[str, str] = {"variant": variant_col, "visitors": visitors_col}
    if conversions_col:
        column_roles["conversions"] = conversions_col
    if revenue_cols:
        column_roles["revenue_sum"] = revenue_cols[0]
        column_roles["revenue_sum_squares"] = revenue_cols[1]
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
) -> dict[str, Any]:
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
        "variants": variants,
        "input_interpretation": {
            "source_type": source_type,
            "variant_count": len(variants),
            "column_roles": column_roles,
        },
    }


def build_payload_from_rows(
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shape = detect_input_shape(rows)
    if shape == "row_level":
        return _build_payload_from_row_level(rows, defaults)
    return _build_payload_from_aggregated(rows, defaults)


_DURATION_FIELDS: dict[str, type] = {
    "method": str,
    "baseline_rate": float,
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


def build_duration_request_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Read duration assumptions from the first row of a table with named columns.

    Only columns that are present and non-empty are returned; callers supply defaults.
    """
    if not rows:
        raise ValueError("Duration input table is empty.")
    row = {str(k).strip().lower(): v for k, v in rows[0].items() if k is not None}
    unknown = sorted(k for k in row if k and k not in _DURATION_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown duration column(s): {', '.join(unknown)}. "
            f"Expected any of: {', '.join(_DURATION_FIELDS)}."
        )
    out: dict[str, Any] = {}
    for key, kind in _DURATION_FIELDS.items():
        value = row.get(key)
        if value is None or value == "":
            continue
        if kind is str:
            out[key] = str(value).strip().lower()
        elif kind is int:
            out[key] = int(float(value))
        else:
            out[key] = float(value)
    return out
