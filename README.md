# ablift

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
- CSV/XLSX ingestion with mapping files

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

Run from CSV/XLSX with an explicit mapping file when the source headers are unusual:

```bash
ablift analyze \
  --input examples/conversion_multivariant.csv \
  --mapping examples/mapping_conversion_bayes.json \
  --output output.json \
  --report report.md
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
- `ablift analyze-text`: parse pasted stats text into an analysis payload
- `ablift duration`: estimate runtime needed for a test
- `ablift doctor`: verify the environment and required dependencies
- `ablift example-input`: print a starter JSON payload
- `ablift example-mapping`: print a starter mapping file

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

For CSV/XLSX input, the JSON output also includes `analysis_settings.input_interpretation` so agents can see:
- whether a mapping file was used
- which columns were inferred
- which columns were ultimately resolved
- how the control row was identified

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

The source headers do not need to be named `visitors` and `conversions`. You can map any business-specific column names into these internal fields.

Good source-column examples:
- `sessions` -> `visitors`
- `click_sessions` -> `conversions`
- `users_exposed` -> `visitors`
- `orders` -> `conversions`
- `visits` -> `visitors`
- `signup_visits` -> `conversions`

Ambiguous or risky source-column examples:
- `clicks` -> `conversions` only if each denominator unit can contribute at most one click
- `orders` -> `conversions` only if the metric is "users who ordered" or "visits with an order", not total order count when repeats are possible

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

For tabular input, `ablift analyze` tries to work out of the box.

It can infer common headers such as:
- variant: `variant`, `variant_name`, `group`, `arm`, `treatment`
- denominator: `visitors`, `users`, `sessions`, `visits`, `exposures`
- success count: `conversions`, `orders`, `purchases`, `signups`, `click_sessions`
- control flag: `is_control`, `control`

If there is no explicit control column, `ablift` also treats a variant named `control` as the control row.

### What mapping means

A mapping file is a translation layer from raw source headers into the canonical fields used by the analysis engine.

You only need a mapping when:
- your headers do not match the common aliases
- you want to force a specific interpretation
- you want to store analysis settings together with the column mapping

Generate a mapping template:

```bash
ablift example-mapping > mapping.json
```

Use `analyze` with `--mapping` when your source data uses business-specific column names that cannot be inferred reliably.

Mapping keys:
- `columns.variant`, `columns.visitors`, `columns.conversions`
- optional `columns.is_control`
- optional `columns.revenue_sum`, `columns.revenue_sum_squares`
- optional control detection fallback: `control.column` + `control.value`

Example source CSV using business names:

```csv
variant_name,sessions,click_sessions,is_control
control,50000,2000,true
v1,50000,2080,false
v2,50000,2140,false
```

Matching mapping file:

```json
{
  "experiment_name": "homepage_cta",
  "method": "bayesian",
  "primary_metric": "conversion_rate",
  "columns": {
    "variant": "variant_name",
    "visitors": "sessions",
    "conversions": "click_sessions",
    "is_control": "is_control"
  }
}
```

Run it:

```bash
ablift analyze \
  --input input.csv \
  --mapping mapping.json
```

The same structure works for `.xlsx` and `.xlsm`. Use `--sheet` to select a worksheet when needed.

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
- [examples/mapping_conversion_bayes.json](examples/mapping_conversion_bayes.json)
- [examples/arpu_bayesian.csv](examples/arpu_bayesian.csv)
- [examples/mapping_arpu_bayes.json](examples/mapping_arpu_bayes.json)
- [examples/duration_mapping.json](examples/duration_mapping.json)

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

Analyze pasted stats text:

```bash
uv run ablift analyze-text \
  --text "Variant A: 100 conversions out of 2000 visitors\nVariant B: 125 conversions out of 2000 visitors" \
  --experiment-name pasted_example
```

Estimate duration from CSV/XLSX:

```bash
uv run ablift duration \
  --input examples/duration_inputs.xlsx \
  --mapping examples/duration_mapping.json \
  --sheet Sheet1
```

## Agent Playbook

1. Try `ablift analyze --input ...` directly first.
2. If the table is unusual, build `mapping.json` to translate source headers into `ablift` fields.
3. Run `ablift analyze ...`.
4. Read `recommendation.action`, `decision_confidence`, and `risk_flags`.
5. If `action=continue_collecting_data`, schedule the next look.
6. If `action=investigate_data_quality`, resolve SRM or tracking issues before deciding to ship.
7. For planning questions, run `ablift duration`.
8. For pasted stats messages, run `ablift analyze-text`.

## Input validation errors

Common errors and fixes:
- `Exactly one variant must have is_control=true`:
  mark one and only one control row.
- `conversions cannot exceed visitors`:
  fix aggregation query or mapped columns. This often means the metric is a repeated-event count rather than a binary outcome.
- `ARPU requires revenue_sum and revenue_sum_squares`:
  include both revenue aggregate columns.
- `primary_metric must be 'conversion_rate' or 'arpu'`:
  fix mapping or input payload values.

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
