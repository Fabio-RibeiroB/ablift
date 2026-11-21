#!/usr/bin/env bash
# Run every bundled example end to end. Outputs go to examples/out/ (git-ignored).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EX="$ROOT_DIR/examples"
OUT_DIR="$EX/out"
mkdir -p "$OUT_DIR"

run() { echo "+ ablift $*" >&2; uv run --project "$ROOT_DIR" ablift "$@"; }

run analyze --input "$EX/conversion_multivariant.csv" \
  --enable-recommendation --prob-threshold 0.95 --max-expected-loss 0.001 \
  --output "$OUT_DIR/conversion_bayes_output.json" --report "$OUT_DIR/conversion_bayes_report.md"

run analyze --input "$EX/conversion_multivariant.csv" --method sequential \
  --output "$OUT_DIR/conversion_seq_output.json" --report "$OUT_DIR/conversion_seq_report.md"

run analyze --input "$EX/arpu_bayesian.csv" \
  --output "$OUT_DIR/arpu_bayes_output.json" --report "$OUT_DIR/arpu_bayes_report.md"

run analyze --input "$EX/arpu_bayesian.xlsx" --sheet Sheet1 \
  --output "$OUT_DIR/arpu_bayes_xlsx_output.json" --report "$OUT_DIR/arpu_bayes_xlsx_report.md"

run analyze --input "$EX/arpu_sequential.csv" --method sequential \
  --output "$OUT_DIR/arpu_seq_output.json" --report "$OUT_DIR/arpu_seq_report.md"

run duration --input "$EX/duration_inputs.csv" --output "$OUT_DIR/duration_csv_output.json"

run duration --input "$EX/duration_inputs.xlsx" --sheet Sheet1 \
  --output "$OUT_DIR/duration_xlsx_output.json"

echo "Demo outputs written to $OUT_DIR" >&2
