from __future__ import annotations

import re
from typing import Any


def parse_duration_prompt(text: str) -> dict[str, Any]:
    def extract_float(keys: list[str], default: float | None = None) -> float | None:
        for key in keys:
            m = re.search(rf"{key}\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*%?", text, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
        return default

    def extract_int(keys: list[str], default: int | None = None) -> int | None:
        value = extract_float(keys)
        if value is None:
            return default
        return int(round(value))

    traffic = extract_int([r"traffic", r"daily[_ ]?traffic", r"visitors[_ ]?per[_ ]?day"])
    baseline_pct = extract_float([r"baseline", r"baseline[_ ]?rate"])
    mde_pct = extract_float([r"mde", r"relative[_ ]?mde"])
    alpha = extract_float([r"alpha"], default=0.05)
    power = extract_float([r"power"], default=0.8)
    n_variants = extract_int([r"variants", r"arms"], default=2)
    max_looks = extract_int([r"looks", r"max[_ ]?looks"], default=10)

    return {
        "daily_total_traffic": traffic,
        "baseline_rate": None if baseline_pct is None else baseline_pct / 100.0,
        "relative_mde": None if mde_pct is None else mde_pct / 100.0,
        "alpha": alpha,
        "power": power,
        "n_variants": n_variants,
        "max_looks": max_looks,
    }
