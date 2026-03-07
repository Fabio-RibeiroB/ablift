from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


def read_table(path: str, sheet: str | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    if suffix in {".xlsx", ".xlsm"}:
        wb = load_workbook(filename=path, read_only=True, data_only=True)
        ws = wb[sheet] if sheet else wb.active
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
            out.append(obj)
        return out

    raise ValueError("Unsupported file type. Use .csv or .xlsx/.xlsm")


def _to_int(value: Any, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"Missing required integer field '{field_name}'.")
    return int(float(value))


def _to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "control"}


def build_payload_from_rows(
    rows: list[dict[str, Any]],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    columns = mapping.get("columns", {})
    name_col = columns.get("variant", "variant")
    visitors_col = columns.get("visitors", "visitors")
    conversions_col = columns.get("conversions", "conversions")
    is_control_col = columns.get("is_control")
    revenue_sum_col = columns.get("revenue_sum")
    revenue_sum_sq_col = columns.get("revenue_sum_squares")

    control_rule = mapping.get("control", {})
    control_col = control_rule.get("column")
    control_value = control_rule.get("value")

    variants: list[dict[str, Any]] = []
    for row in rows:
        variant_name = str(row.get(name_col, "")).strip()
        if not variant_name:
            continue

        is_control = False
        if is_control_col:
            is_control = _to_bool(row.get(is_control_col))
        elif control_col is not None and control_value is not None:
            is_control = str(row.get(control_col, "")).strip() == str(control_value)

        variant_obj = {
            "name": variant_name,
            "visitors": _to_int(row.get(visitors_col), visitors_col),
            "conversions": _to_int(row.get(conversions_col), conversions_col),
            "is_control": is_control,
        }

        if revenue_sum_col:
            variant_obj["revenue_sum"] = _to_float_or_none(row.get(revenue_sum_col))
        if revenue_sum_sq_col:
            variant_obj["revenue_sum_squares"] = _to_float_or_none(row.get(revenue_sum_sq_col))

        variants.append(variant_obj)

    payload = {
        "experiment_name": mapping["experiment_name"],
        "method": mapping["method"],
        "primary_metric": mapping.get("primary_metric", "conversion_rate"),
        "alpha": mapping.get("alpha", 0.05),
        "look_index": mapping.get("look_index", 1),
        "max_looks": mapping.get("max_looks", 1),
        "information_fraction": mapping.get("information_fraction"),
        "decision_thresholds": mapping.get("decision_thresholds", {}),
        "guardrails": mapping.get("guardrails", []),
        "samples": mapping.get("samples", 50000),
        "random_seed": mapping.get("random_seed", 7),
        "variants": variants,
    }

    return payload


def build_duration_request_from_rows(
    rows: list[dict[str, Any]],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Duration input table is empty.")

    row = rows[0]
    columns = mapping.get("columns", {})

    def get_value(key: str, default: Any = None) -> Any:
        col = columns.get(key)
        if col is None:
            return mapping.get(key, default)
        value = row.get(col)
        return mapping.get(key, default) if value in (None, "") else value

    method = str(get_value("method", "frequentist")).strip().lower()
    baseline_rate = float(get_value("baseline_rate"))
    relative_mde = float(get_value("relative_mde"))
    daily_traffic = int(float(get_value("daily_traffic")))

    out = {
        "method": method,
        "baseline_rate": baseline_rate,
        "relative_mde": relative_mde,
        "daily_traffic": daily_traffic,
        "n_variants": int(float(get_value("n_variants", 2))),
        "alpha": float(get_value("alpha", 0.05)),
        "power": float(get_value("power", 0.8)),
        "max_looks": int(float(get_value("max_looks", 10))),
        "prob_threshold": float(get_value("prob_threshold", 0.95)),
        "max_expected_loss": float(get_value("max_expected_loss", 0.001)),
        "assurance_target": float(get_value("assurance_target", 0.8)),
        "max_days": int(float(get_value("max_days", 60))),
    }
    return out
