#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/examples/out"

mkdir -p "$OUT_DIR"

uv run ablift analyze \
  --input "$ROOT_DIR/examples/conversion_multivariant.csv" \
  --mapping "$ROOT_DIR/examples/mapping_conversion_bayes.json" \
  --output "$OUT_DIR/conversion_bayes_output.json" \
  --report "$OUT_DIR/conversion_bayes_report.md"

uv run ablift analyze \
  --input "$ROOT_DIR/examples/arpu_bayesian.csv" \
  --mapping "$ROOT_DIR/examples/mapping_arpu_bayes.json" \
  --output "$OUT_DIR/arpu_bayes_output.json" \
  --report "$OUT_DIR/arpu_bayes_report.md"

uv run ablift analyze \
  --input "$ROOT_DIR/examples/arpu_sequential_look3.csv" \
  --mapping "$ROOT_DIR/examples/mapping_arpu_seq_look3.json" \
  --output "$OUT_DIR/arpu_seq_look3_output.json" \
  --report "$OUT_DIR/arpu_seq_look3_report.md"

uv run ablift analyze \
  --input "$ROOT_DIR/examples/arpu_bayesian.xlsx" \
  --mapping "$ROOT_DIR/examples/mapping_arpu_bayes.json" \
  --sheet Sheet1 \
  --output "$OUT_DIR/arpu_bayes_xlsx_output.json" \
  --report "$OUT_DIR/arpu_bayes_xlsx_report.md"

uv run ablift duration \
  --input "$ROOT_DIR/examples/duration_inputs.csv" \
  --mapping "$ROOT_DIR/examples/duration_mapping.json" \
  --output "$OUT_DIR/duration_csv_output.json"

uv run ablift duration \
  --input "$ROOT_DIR/examples/duration_inputs.xlsx" \
  --mapping "$ROOT_DIR/examples/duration_mapping.json" \
  --sheet Sheet1 \
  --output "$OUT_DIR/duration_xlsx_output.json"

echo "Demo outputs written to $OUT_DIR"
