"""Verify that the reference SQL queries in docs/sql.md run and feed into ablift."""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from ablift.cli import cli

SQL_DOC = Path(__file__).resolve().parents[1] / "docs" / "sql.md"


def _extract_sql_blocks() -> list[str]:
    text = SQL_DOC.read_text(encoding="utf-8")
    blocks = [match.group(1).strip() for match in re.finditer(r"```sql\n(.*?)```", text, re.S)]
    assert len(blocks) >= 2, f"Expected main and CUPED SQL blocks in {SQL_DOC}"
    return blocks


def _seed_database() -> sqlite3.Connection:
    """Create an in-memory SQLite database covering common experimentation traps.

    Fixture summary (clean units after deduping exposures and dropping mixed-variant users):
      control (5 visitors):
        - u1: 0 post-exposure orders (1 pre-exposure order of 5.0 -> ignored by main query,
              used as CUPED covariate)
        - u2: 0 orders at all
        - u3: duplicate exposure rows in control; 1 post-exposure order of 20.0 (between the two
              exposures, captured via MIN(exposed_at)) + 1 pre-exposure order of 10.0
        - u4: 2 post-exposure orders (15.0, 25.0 -> unit total 40.0, square 1600.0, not 850.0)
              + 1 pre-exposure order of 30.0
        - u5: 1 post-exposure order of 10.0; 0 pre-exposure orders
      treatment (5 visitors):
        - u6: 0 orders at all
        - u7: 1 post-exposure order of 30.0 + 1 pre-exposure order of 15.0
        - u8: 3 post-exposure orders (10.0, 20.0, 5.0 -> unit total 35.0, square 1225.0,
              not 525.0) + 2 pre-exposure orders (12.0, 8.0 -> covariate 20.0)
        - u9: duplicate exposure rows in treatment; 1 post-exposure order of 50.0
              + 1 pre-exposure order of 25.0
        - u10: 0 post-exposure orders + 1 pre-exposure order of 4.0
      excluded:
        - u_mixed: exposed to both control and treatment, with large orders (500.0 pre, 999.0 post)
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE exposures (
            user_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            exposed_at TEXT NOT NULL
        );
        CREATE TABLE orders (
            user_id TEXT NOT NULL,
            revenue REAL NOT NULL,
            ordered_at TEXT NOT NULL
        );
        """
    )

    exposures = [
        ("u1", "control", "2026-01-10T10:00:00Z"),
        ("u2", "control", "2026-01-10T11:00:00Z"),
        # Duplicate exposure rows for the same user in control:
        ("u3", "control", "2026-01-10T08:00:00Z"),
        ("u3", "control", "2026-01-10T12:00:00Z"),
        ("u4", "control", "2026-01-10T10:00:00Z"),
        ("u5", "control", "2026-01-10T10:00:00Z"),
        ("u6", "treatment", "2026-01-10T10:00:00Z"),
        ("u7", "treatment", "2026-01-10T10:00:00Z"),
        ("u8", "treatment", "2026-01-10T10:00:00Z"),
        # Duplicate exposure rows for the same user in treatment:
        ("u9", "treatment", "2026-01-10T09:00:00Z"),
        ("u9", "treatment", "2026-01-10T15:00:00Z"),
        ("u10", "treatment", "2026-01-10T10:00:00Z"),
        # User exposed to both variants (must be excluded):
        ("u_mixed", "control", "2026-01-10T10:00:00Z"),
        ("u_mixed", "treatment", "2026-01-10T11:00:00Z"),
    ]
    orders = [
        # Pre-exposure orders (excluded from main query, used as covariate X in CUPED):
        ("u1", 5.0, "2026-01-05T09:00:00Z"),
        ("u3", 10.0, "2026-01-04T08:00:00Z"),
        ("u4", 30.0, "2026-01-03T12:00:00Z"),
        ("u7", 15.0, "2026-01-02T14:00:00Z"),
        ("u8", 12.0, "2026-01-02T10:00:00Z"),
        ("u8", 8.0, "2026-01-06T10:00:00Z"),
        ("u9", 25.0, "2026-01-01T10:00:00Z"),
        ("u10", 4.0, "2026-01-05T10:00:00Z"),
        ("u_mixed", 500.0, "2026-01-05T10:00:00Z"),
        # Post-exposure orders:
        ("u3", 20.0, "2026-01-10T09:00:00Z"),
        ("u4", 15.0, "2026-01-11T10:00:00Z"),
        ("u4", 25.0, "2026-01-12T10:00:00Z"),
        ("u5", 10.0, "2026-01-11T11:00:00Z"),
        ("u7", 30.0, "2026-01-11T09:00:00Z"),
        ("u8", 10.0, "2026-01-11T08:00:00Z"),
        ("u8", 20.0, "2026-01-11T12:00:00Z"),
        ("u8", 5.0, "2026-01-12T18:00:00Z"),
        ("u9", 50.0, "2026-01-11T10:00:00Z"),
        ("u_mixed", 999.0, "2026-01-12T10:00:00Z"),
    ]
    conn.executemany("INSERT INTO exposures VALUES (?, ?, ?)", exposures)
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?)", orders)
    return conn


def _rows_to_csv(columns: list[str], rows: list[tuple[object, ...]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buf.getvalue()


def test_reference_sql_aggregates_match_hand_computed_values_and_pipe_into_cli() -> None:
    main_sql = _extract_sql_blocks()[0]
    conn = _seed_database()
    try:
        cursor = conn.execute(main_sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    finally:
        conn.close()

    assert columns == ["name", "visitors", "conversions", "revenue_sum", "revenue_sum_squares"]
    assert len(rows) == 2

    # Hand-computed from per-unit totals:
    # control:   revenues [0, 0, 20, 40, 10] -> sum=70, sum_sq=0+0+400+1600+100=2100
    # treatment: revenues [0, 30, 35, 50, 0] -> sum=115, sum_sq=0+900+1225+2500+0=4625
    by_variant = {row[0]: row for row in rows}
    assert rows[0][0] == "control"

    ctrl = by_variant["control"]
    assert ctrl[1] == 5
    assert ctrl[2] == 3
    assert ctrl[3] == pytest.approx(70.0)
    assert ctrl[4] == pytest.approx(2100.0)

    trt = by_variant["treatment"]
    assert trt[1] == 5
    assert trt[2] == 3
    assert trt[3] == pytest.approx(115.0)
    assert trt[4] == pytest.approx(4625.0)

    csv_text = _rows_to_csv(columns, rows)
    runner = CliRunner()

    # Default ARPU analysis from the 5-column aggregated CSV over stdin:
    arpu_result = runner.invoke(
        cli,
        ["analyze", "--input", "-", "--input-format", "csv", "--samples", "1000"],
        input=csv_text,
    )
    assert arpu_result.exit_code == 0, arpu_result.output
    arpu_payload = json.loads(arpu_result.output)
    assert arpu_payload["control_variant"] == "control"
    assert arpu_payload["analysis_settings"]["primary_metric"] == "arpu"
    assert arpu_payload["srm"]["observed"] == [5, 5]
    assert arpu_payload["comparisons"][0]["control_rate"] == pytest.approx(70.0 / 5)
    assert arpu_payload["comparisons"][0]["treatment_rate"] == pytest.approx(115.0 / 5)

    # Conversion-rate analysis from the exact same 5-column CSV:
    conv_result = runner.invoke(
        cli,
        [
            "analyze",
            "--input",
            "-",
            "--input-format",
            "csv",
            "--primary-metric",
            "conversion_rate",
            "--samples",
            "1000",
        ],
        input=csv_text,
    )
    assert conv_result.exit_code == 0, conv_result.output
    conv_payload = json.loads(conv_result.output)
    assert conv_payload["analysis_settings"]["primary_metric"] == "conversion_rate"
    assert conv_payload["srm"]["observed"] == [5, 5]
    assert conv_payload["comparisons"][0]["control_rate"] == pytest.approx(3 / 5)
    assert conv_payload["comparisons"][0]["treatment_rate"] == pytest.approx(3 / 5)


def test_reference_cuped_sql_aggregates_match_hand_computed_values_and_pipe_into_cli() -> None:
    cuped_sql = _extract_sql_blocks()[1]
    conn = _seed_database()
    try:
        cursor = conn.execute(cuped_sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    finally:
        conn.close()

    assert columns == [
        "name",
        "visitors",
        "conversions",
        "revenue_sum",
        "revenue_sum_squares",
        "covariate_sum",
        "covariate_sum_squares",
        "revenue_covariate_sum",
        "conversion_covariate_sum",
    ]
    assert len(rows) == 2

    # Hand-computed from per-unit pre/post revenue pairs:
    # control:   (y, x) = [(0,5), (0,0), (20,10), (40,30), (10,0)]
    #   cov_sum = 45, cov_sum_sq = 1025, revenue_cov_sum = 0+0+200+1200+0 = 1400
    #   converted = [0,0,1,1,1] -> conversion_cov_sum = 10+30+0 = 40
    # treatment: (y, x) = [(0,0), (30,15), (35,20), (50,25), (0,4)]
    #   cov_sum = 64, cov_sum_sq = 1266, revenue_cov_sum = 0+450+700+1250+0 = 2400
    #   converted = [0,1,1,1,0] -> conversion_cov_sum = 15+20+25 = 60
    records = [dict(zip(columns, row, strict=True)) for row in rows]
    by_name = {rec["name"]: rec for rec in records}

    ctrl = by_name["control"]
    assert ctrl["visitors"] == 5
    assert ctrl["conversions"] == 3
    assert ctrl["revenue_sum"] == pytest.approx(70.0)
    assert ctrl["revenue_sum_squares"] == pytest.approx(2100.0)
    assert ctrl["covariate_sum"] == pytest.approx(45.0)
    assert ctrl["covariate_sum_squares"] == pytest.approx(1025.0)
    assert ctrl["revenue_covariate_sum"] == pytest.approx(1400.0)
    assert ctrl["conversion_covariate_sum"] == pytest.approx(40.0)

    trt = by_name["treatment"]
    assert trt["visitors"] == 5
    assert trt["conversions"] == 3
    assert trt["revenue_sum"] == pytest.approx(115.0)
    assert trt["revenue_sum_squares"] == pytest.approx(4625.0)
    assert trt["covariate_sum"] == pytest.approx(64.0)
    assert trt["covariate_sum_squares"] == pytest.approx(1266.0)
    assert trt["revenue_covariate_sum"] == pytest.approx(2400.0)
    assert trt["conversion_covariate_sum"] == pytest.approx(60.0)

    # Each metric takes its own cross-product as outcome_covariate_sum, as the doc says.
    for metric, cross in (
        ("arpu", "revenue_covariate_sum"),
        ("conversion_rate", "conversion_covariate_sum"),
    ):
        variants = [
            {
                **{k: v for k, v in rec.items() if not k.endswith("_covariate_sum")},
                "outcome_covariate_sum": rec[cross],
                "is_control": rec["name"] == "control",
            }
            for rec in records
        ]
        payload = {
            "experiment_name": "sql_cuped_fixture",
            "method": "sequential",
            "primary_metric": metric,
            "variants": variants,
        }
        result = CliRunner().invoke(cli, ["analyze", "--input", "-"], input=json.dumps(payload))
        assert result.exit_code == 0, (metric, result.output)
        out = json.loads(result.output)
        assert out["srm"]["observed"] == [5, 5]
        assert "cuped" in out["analysis_settings"]["input_interpretation"]


def test_documented_is_control_column_is_accepted() -> None:
    """The doc's `CASE WHEN ... THEN 1 ELSE 0 END AS is_control` column must parse."""
    csv_text = (
        "name,visitors,conversions,revenue_sum,revenue_sum_squares,is_control\n"
        "baseline,5,3,70.0,2100.0,1\n"
        "treatment,5,3,115.0,4625.0,0\n"
    )
    result = CliRunner().invoke(
        cli, ["analyze", "--input", "-", "--input-format", "csv"], input=csv_text
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["comparisons"][0]["control"] == "baseline"


def test_documented_cuped_json_example_runs_through_cli() -> None:
    text = SQL_DOC.read_text(encoding="utf-8")
    json_blocks = [
        match.group(1).strip() for match in re.finditer(r"```json\n(.*?)```", text, re.S)
    ]
    assert len(json_blocks) == 1, f"Expected one CUPED JSON block in {SQL_DOC}"
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "--input", "-"], input=json_blocks[0])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert "cuped" in out["analysis_settings"]["input_interpretation"]
