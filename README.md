# ablift

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

`ablift` is an agent-friendly CLI for A/B/n decisions.

> Early release (alpha): this project is work in progress and not production-hardened yet.

It supports:
- Bayesian conversion-rate decisions (Beta-Binomial)
- Bayesian ARPU probability-to-win from aggregate revenue stats
- Frequentist sequential decisions (O'Brien-Fleming alpha spending)
- Multi-variant (`A/B/n`) comparisons against one control
- Guardrail checks (latency, bounce rate, error rate, etc.)
- SRM detection (sample ratio mismatch) for data quality
- Structured JSON output for agents and automations
- Markdown report generation for human review
- CSV/XLSX ingestion with auto-detected column layout

## Install

```bash
uv tool install .
```

This installs `ablift` as a normal CLI tool and exposes the `ablift` command on your `PATH`.

To work from the repo during development:

```bash
uv sync --group test
```

Then run commands through the repo-managed environment:

```bash
uv run ablift --help
```

If you want an editable install in an existing environment:

```bash
uv pip install -e .
```

## Quickstart

Show available commands:

```bash
ablift --help
ablift analyze --help
```

Generate a JSON input template:

```bash
ablift example-input > input.json
```

Run a Bayesian analysis from JSON:

```bash
ablift analyze \
  --input input.json \
  --output output.json \
  --report report.md
```

Run the same workflow from a CSV file with inferred columns:

```bash
ablift analyze \
  --input examples/conversion_multivariant.csv \
  --output output.json \
  --report report.md
```

Run with an explicit Bayesian decision policy:

```bash
ablift analyze --input bayesian_input.csv
```

If you are developing from the repo, use the same commands prefixed with `uv run`:

```bash
uv run ablift analyze \
  --input input.json \
  --output output.json \
  --report report.md
```

## Command summary

- `ablift analyze`: analyze `.json`, `.csv`, `.xlsx`, or `.xlsm` input
- `ablift duration`: estimate runtime needed for a test
- `ablift doctor`: verify the environment and required dependencies
- `ablift example-input`: print a starter JSON payload

## Choosing the metric model

Use `primary_metric: "conversion_rate"` when each row represents:
- a denominator count
- a success count
- successes that cannot exceed the denominator

Examples that fit this model:
- sessions and click-through sessions
- users and purchasers
- emails delivered and openers
- visits and visits with at least one signup

Use `primary_metric: "arpu"` when you have aggregate revenue statistics per variant:
- `visitors`
- `conversions`
- `revenue_sum`
- `revenue_sum_squares`

If your metric can happen multiple times per unit and the count can exceed the denominator, it is not a fit for the current `conversion_rate` model. For example:
- visits and total clicks across all visits
- users and page views
- sessions and multiple add-to-cart events

In those cases, either:
- redefine the outcome as binary per unit, such as "visit with at least one click"
- or use a different metric/model outside the current scope of this tool

## Input rules

The input contract is strict about semantics, not source column names.

`ablift` separates:
- inference settings: priors, posterior sample count, random seed
- decision policy: thresholds used to turn estimates into actions

For Bayesian runs, recommendations are optional. If you do not provide a decision policy, `ablift` reports the posterior estimates but leaves `recommendation` as `null`.

For CSV/XLSX input, the JSON output also includes `analysis_settings.input_interpretation` so agents can see the detected input shape (`row_level` or `aggregated`) and whether ARPU results are approximate.

For analysis inputs:
- at least 2 variants are required
- exactly 1 variant must have `is_control: true`
- `method` must be `bayesian` or `frequentist_sequential`
- `primary_metric` must be `conversion_rate` or `arpu`
- `visitors` must be a positive integer
- `conversions` must be a non-negative integer
- `conversions` cannot exceed `visitors`
- for `arpu`, each variant must also include `revenue_sum` and `revenue_sum_squares`

Bayesian defaults:
- posterior sample count defaults to `50000`
- random seed defaults to `7`
- conversion-rate prior is `Beta(1, 1)`
- ARPU prior is a weak `Normal-Inverse-Gamma` prior with `mu0=0`, `kappa0=1e-6`, `alpha0=1`, `beta0=1`

Bayesian recommendations require an explicit decision policy, either in the input payload, via `[tool.ablift]` in `pyproject.toml`, or via CLI flags such as:
- `--enable-recommendation`
- `--prob-threshold`
- `--max-expected-loss`

Internal field meanings:
- `name`: variant label
- `visitors`: denominator units exposed to the experiment
- `conversions`: success units for the primary outcome
- `is_control`: whether the row is the control variant

## JSON input

Top-level fields:
- `experiment_name` (str)
- `method` (`"bayesian"` or `"frequentist_sequential"`)
- `primary_metric` (`"conversion_rate"` or `"arpu"`)
- `alpha` (float, default `0.05`)
- `look_index` and `max_looks` (ints, sequential mode)
- `information_fraction` (optional float in `(0, 1]`, overrides look/max_looks)
- `variants` (list): exactly one row must include `"is_control": true`
- `guardrails` (optional list)
- `decision_policy` (optional):
  - `enabled` (bool)
  - `bayes_prob_beats_control`
  - `max_expected_loss`
- `decision_thresholds` (legacy alias, still supported):
  - `bayes_prob_beats_control` (default `0.95`)
  - `max_expected_loss` (default `0.001`)
- `samples` (default `50000`, Bayesian mode)
- `random_seed` (default `7`)

Variant row:

```json
{
  "name": "control",
  "visitors": 100000,
  "conversions": 4000,
  "is_control": true
}
```

For `primary_metric: "arpu"`, each variant also needs:
- `revenue_sum`
- `revenue_sum_squares`

Minimal Bayesian conversion-rate example:

```json
{
  "experiment_name": "homepage_cta",
  "method": "bayesian",
  "primary_metric": "conversion_rate",
  "variants": [
    {"name": "control", "visitors": 50000, "conversions": 2000, "is_control": true},
    {"name": "v1", "visitors": 50000, "conversions": 2080, "is_control": false}
  ]
}
```

Bayesian ARPU probability-to-win example:

```json
{
  "experiment_name": "pricing_page",
  "method": "bayesian",
  "primary_metric": "arpu",
  "variants": [
    {"name": "control", "visitors": 10000, "conversions": 550, "revenue_sum": 22000, "revenue_sum_squares": 150000, "is_control": true},
    {"name": "v1", "visitors": 10000, "conversions": 570, "revenue_sum": 23500, "revenue_sum_squares": 170000, "is_control": false}
  ]
}
```

## CSV/XLSX input

`ablift analyze` reads CSV and XLSX files using **positional columns** — column names are ignored, column order determines the role.

### Aggregated input (one row per variant)

| Col 1 | Col 2 | Col 3 | Col 4 (optional) |
|-------|-------|-------|-----------------|
| variant name | visitors | conversions **or** revenue_sum | is_control (boolean) |

If col 3 values are integers ≤ col 2 → conversion rate analysis.
If col 3 values are floats or exceed col 2 → ARPU analysis (approximate, flagged in output).

### Row-level input (one row per observation)

| Col 1 | Col 2 | Col 3 (optional) |
|-------|-------|-----------------|
| variant name | outcome | is_control (boolean) |

If col 2 values are 0/1 → conversion rate analysis (tool counts visitors and conversions per variant).
If col 2 values are continuous → ARPU analysis (tool computes revenue_sum and revenue_sum_squares per variant).

### Control detection

The first variant (first row or first group) is treated as control by default. To override, include a boolean column (values like `true`/`false`, `1`/`0`, `yes`/`no`) — the tool detects it automatically.

Use `--sheet` to select a worksheet in `.xlsx`/`.xlsm` files.

### Reusable project config

Use `[tool.ablift]` in `pyproject.toml` when you want one decision policy and one set of Bayesian inference settings reused across analyses in the project.

Example:

```toml
[tool.ablift]
method = "bayesian"
primary_metric = "conversion_rate"
samples = 50000
random_seed = 7

[tool.ablift.decision_policy]
enabled = true
bayes_prob_beats_control = 0.95
max_expected_loss = 0.001
```

Precedence is:
- CLI flags
- input file
- `pyproject.toml` `[tool.ablift]`
- built-in defaults

Bundled examples:
- [examples/conversion_multivariant.csv](examples/conversion_multivariant.csv)
- [examples/arpu_bayesian.csv](examples/arpu_bayesian.csv)

## More examples

```json
{
  "experiment_name": "homepage_cta",
  "method": "bayesian",
  "primary_metric": "conversion_rate",
  "variants": [
    {"name": "control", "visitors": 50000, "conversions": 2000, "is_control": true},
    {"name": "v1", "visitors": 50000, "conversions": 2080, "is_control": false},
    {"name": "v2", "visitors": 50000, "conversions": 2140, "is_control": false}
  ]
}
```

Sequential ARPU at an early look:

```json
{
  "experiment_name": "checkout_flow",
  "method": "frequentist_sequential",
  "primary_metric": "arpu",
  "alpha": 0.05,
  "look_index": 3,
  "max_looks": 10,
  "variants": [
    {"name": "control", "visitors": 12000, "conversions": 610, "revenue_sum": 21000, "revenue_sum_squares": 150000, "is_control": true},
    {"name": "v1", "visitors": 12000, "conversions": 640, "revenue_sum": 22400, "revenue_sum_squares": 167000, "is_control": false}
  ]
}
```

## Development

Run the CLI from the repo:

```bash
uv run ablift --help
```

Run the test suite:

```bash
uv run pytest
```

Run all bundled demos:

```bash
make demo
```

Check environment readiness:

```bash
uv run ablift doctor
uv run ablift doctor --json
uv run ablift doctor --strict
```

Estimate duration from assumptions:

```bash
uv run ablift duration \
  --method frequentist \
  --baseline-rate 0.04 \
  --relative-mde 0.05 \
  --daily-traffic 50000 \
  --n-variants 3 \
  --max-looks 10
```

Estimate Bayesian duration:

```bash
uv run ablift duration \
  --method bayesian \
  --baseline-rate 0.04 \
  --relative-mde 0.05 \
  --daily-traffic 50000 \
  --n-variants 3 \
  --max-days 60
```

Estimate duration from CSV/XLSX (column names must match field names: `baseline_rate`, `relative_mde`, `daily_traffic`, etc.):

```bash
uv run ablift duration \
  --input examples/duration_inputs.xlsx \
  --sheet Sheet1
```

## Agent Playbook

1. Run `ablift analyze --input ...` with a JSON, CSV, or XLSX file.
2. For CSV/XLSX, format data with positional columns (see CSV/XLSX input section).
3. Read `recommendation.action`, `decision_confidence`, and `risk_flags`.
4. If `action=continue_collecting_data`, schedule the next look.
5. If `action=investigate_data_quality`, resolve SRM or tracking issues before deciding to ship.
6. For planning questions, run `ablift duration`.

## Input validation errors

Common errors and fixes:
- `Exactly one variant must have is_control=true`:
  mark one and only one control row.
- `conversions cannot exceed visitors`:
  the metric is a repeated-event count rather than a binary outcome — redefine as binary per unit.
- `primary_metric must be 'conversion_rate' or 'arpu'`:
  fix the input payload value.

## Agent output contract

`recommendation` contains:
- `action` (`ship_*`, `continue_collecting_data`, `do_not_ship`, `investigate_data_quality`, `stop_and_rollback`)
- `rationale`
- `decision_confidence` (0 to 1)
- `next_best_action`
- `risk_flags` (e.g. `srm_detected`, `guardrail_failure`)

If no explicit Bayesian decision policy is provided, `recommendation` is `null` and `analysis_settings.decision_policy` shows that no automated decision policy was applied.

## Notes

- This tool currently uses aggregate statistics for ARPU.
- For production, add metric-specific robust models, stronger QA checks, and regression tests.
