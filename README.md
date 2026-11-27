# ablift

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

`ablift` is an agent-friendly CLI for A/B/n decisions.

> Early release (alpha): this project is work in progress and not production-hardened yet.

It supports:
- Bayesian conversion-rate decisions (Beta-Binomial)
- Bayesian ARPU decisions from aggregate revenue statistics (Normal-Inverse-Gamma)
- Sequential decisions with mSPRT and always-valid confidence intervals: look as often as you like
- Fixed-horizon frequentist decisions (`--method fixed`): two-proportion z-test and Welch's t-test (single planned look)
- Multi-variant (`A/B/n`) comparisons against one control
- Per-variant statistical guardrails (plus backwards-compatible threshold guardrails) and secondary metrics in JSON inputs
- SRM detection (sample ratio mismatch) for data quality
- Test duration planning (sequential, fixed-horizon, Bayesian assurance)
- Structured JSON output (versioned with `schema_version`) for agents and automations
- Markdown report generation for human review
- CSV/XLSX ingestion with positional column layout

## Install

```bash
uv tool install ablift      # or: pipx install ablift / pip install ablift
```

This installs `ablift` as a normal CLI tool and exposes the `ablift` command on your `PATH`.
To install from a checkout instead, run `uv tool install .` in the repo.

To work from the repo during development:

```bash
uv sync
uv run ablift --help
```

If you want an editable install in an existing environment:

```bash
uv pip install -e .
```

## Quickstart

The commands below assume `ablift` is on your `PATH`; from the repo, prefix them with `uv run`.

```bash
ablift --help
ablift analyze --help
```

Generate a JSON input template and analyze it (or pipe directly over stdin with `--input -`):

```bash
ablift example-input > input.json
ablift analyze --input input.json --output output.json --report report.md

# Or pipe directly without temporary files:
ablift example-input | ablift analyze --input -
```

Analyze a CSV (positional columns, see [CSV/XLSX input](#csvxlsx-input)). The Bayesian method is the default; without a decision policy it reports estimates and leaves `recommendation` as `null`:

```bash
ablift analyze --input examples/conversion_multivariant.csv
```

Add a Bayesian decision policy to get a recommendation:

```bash
ablift analyze --input examples/conversion_multivariant.csv \
  --enable-recommendation --prob-threshold 0.95 --max-expected-loss 0.001
```

Run a sequential (mSPRT) analysis. It always produces a recommendation:

```bash
ablift analyze --input examples/arpu_sequential.csv --method sequential
```

Run a fixed-horizon frequentist analysis (only valid when evaluated once at the planned end of the test):

```bash
ablift analyze --input examples/conversion_multivariant.csv --method fixed
```

Plan how long a test needs to run:

```bash
ablift duration --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3
```

## Command summary

- `ablift analyze`: analyze `.json`, `.csv`, `.xlsx`, `.xlsm`, or stdin (`--input -`)
- `ablift duration`: estimate runtime needed for a conversion-rate or ARPU test (flags, file, or `--input -`)
- `ablift schema`: emit JSON Schema (Draft 2020-12) for `analyze-input`, `analyze-output`, `duration-input`, `duration-output`, or `--all`
- `ablift doctor`: verify the environment and required dependencies (`--json-output`, `--strict`)
- `ablift example-input`: print a starter JSON payload

Useful `analyze` flags: `--input-format {json,csv}`, `--method {bayesian,sequential,fixed}`, `--primary-metric {conversion_rate,arpu}`, `--alpha`, `--sequential-tau`, `--samples`, `--random-seed`, `--expected-split`, `--enable-recommendation/--no-recommendation`, `--prob-threshold`, `--max-expected-loss`, `--experiment-name`, `--sheet`, `--output`, `--report`. Flags override values in the input.

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

Use `primary_metric: "arpu"` (average revenue per visitor) when you have row-level per-visitor revenue or aggregate revenue statistics per variant:
- `visitors`
- `revenue_sum`
- `revenue_sum_squares` (needed to estimate variance)

For heavy-tailed row-level ARPU data (e.g. whale-dominated revenue):
- Pass `--winsorize Q` (with `0.5 < Q < 1.0`, e.g. `--winsorize 0.99`) to cap outcomes at the **pooled** Q-quantile across all variants before analysis (preventing any arm from shifting its own cap). Cap value, capped counts, and variance reduction are recorded in `analysis_settings.input_interpretation.winsorization`. Passing `--winsorize` with aggregated inputs or non-ARPU metrics exits with a usage error (exit code 2).
- `ablift` automatically checks ARPU tail heaviness (exact skewness, max/mean ratio, and Kohavi's $355 g_1^2$ CLT rule for row-level data; CV $> 3.0$ for aggregated data) and attaches `heavy_tailed_revenue` to `recommendation.risk_flags` and `analysis_settings.input_interpretation.arpu_tail_diagnostics` when tails are heavy.
- Optionally pass `--bootstrap-samples N` (`>= 100`) on row-level ARPU inputs to compute a percentile bootstrap confidence interval cross-check in `analysis_settings.input_interpretation.bootstrap_ci`.

Use `primary_metric: "ratio"` when your randomization unit (e.g. `visitors` / users) differs from the metric denominator — such as click-through rate (clicks / impressions), average order value (revenue / orders), or pages per session. `ablift` estimates the variance via the bivariate Delta method (Deng, Knoblich & Lu 2018) across randomization units so intra-user correlation is accounted for accurately across `bayesian`, `sequential`, and `fixed` modes, as well as statistical guardrails and secondary metrics.

For `primary_metric: "ratio"`, provide either:
- **Row-level CSV/XLSX** (one row per randomization unit) with `--primary-metric ratio` and columns `name, numerator, denominator [, is_control]` (per-unit denominator may be `0`, e.g. users with zero orders)
- **Aggregated JSON/CSV/XLSX** with per-variant sufficient statistics:
  - `visitors` (at least `2` randomization units)
  - `numerator_sum`, `numerator_sum_squares`
  - `denominator_sum` (must be positive), `denominator_sum_squares`
  - `numerator_denominator_sum` ($\sum_i Y_i X_i$)

## Input rules

The input contract is strict about semantics, not source column names.

`ablift` separates:
- inference settings: priors, posterior sample count, random seed
- decision policy: thresholds used to turn Bayesian estimates into actions

For Bayesian runs, recommendations are optional. If you do not provide a decision policy, `ablift` reports the posterior estimates but leaves `recommendation` as `null`. Sequential and fixed-horizon runs always produce a recommendation; the Bayesian decision policy does not apply to them.

For analysis inputs:
- at least 2 variants are required
- exactly 1 variant must have `is_control: true`
- `method` must be `bayesian`, `sequential`, or `fixed`
- `primary_metric` must be `conversion_rate`, `arpu`, or `ratio`
- `visitors` must be a positive integer (at least `2` for `ratio`)
- `conversions` must be a non-negative integer and cannot exceed `visitors`
- `alpha` must be in `(0, 1)`; `sequential_tau` must be positive (and is only valid for `sequential`)
- for `arpu`, each variant must also include `revenue_sum` and `revenue_sum_squares`
- for `ratio`, each variant must include `numerator_sum`, `numerator_sum_squares`, `denominator_sum`, `denominator_sum_squares`, and `numerator_denominator_sum`

Bayesian defaults:
- posterior sample count defaults to `50000`
- random seed defaults to `7`
- conversion-rate prior is `Beta(1, 1)`
- ARPU prior is a weak `Normal-Inverse-Gamma` prior with `mu0=0`, `kappa0=1e-6`, `alpha0=1`, `beta0=1`
- ratio posterior uses an asymptotic normal approximation parameterized by the Delta-method variance (`delta_normal_asymptotic`)

Bayesian recommendations require an explicit decision policy, either in the input payload, via `[tool.ablift]` in `pyproject.toml`, or via CLI flags:
- `--enable-recommendation`
- `--prob-threshold` (default `0.95`)
- `--max-expected-loss` (default `0.001`)

Sequential defaults:
- `alpha` defaults to `0.05`
- `sequential_tau` (CLI `--sequential-tau`) is the mSPRT mixing parameter: the prior standard deviation of the effect on the absolute scale. It defaults to 10% of the control baseline (conversion rate, ARPU, or ratio). Pass it explicitly if you expect much larger or smaller effects.
- You may look at results as often as you like: the false-positive rate stays at `alpha`.

Fixed-horizon defaults (`--method fixed`):
- `alpha` defaults to `0.05` (Bonferroni-adjusted across `n_variants - 1` comparisons, matching `ablift duration --method fixed`).
- **Single-look assumption:** fixed-horizon p-values and confidence intervals are only valid when evaluated **once** at the planned end of the test. Do not peek repeatedly with `--method fixed`; use `--method sequential` for ongoing monitoring.
- Sequential-only (`--sequential-tau`) and Bayesian-only (`--samples`, `--random-seed`, `--enable-recommendation`/`--no-recommendation`, `--prob-threshold`, `--max-expected-loss`) CLI flags are rejected with exit code 2 when `--method fixed` is chosen.

Internal field meanings:
- `name`: variant label
- `visitors`: denominator units exposed to the experiment
- `conversions`: success units for the primary outcome
- `is_control`: whether the row is the control variant

## JSON input

Top-level fields:
- `experiment_name` (str)
- `method` (`"bayesian"`, `"sequential"`, or `"fixed"`)
- `primary_metric` (`"conversion_rate"`, `"arpu"`, or `"ratio"`)
- `alpha` (float, default `0.05`)
- `sequential_tau` (optional float, sequential mode; default 10% of the control baseline)
- `variants` (list): exactly one row must include `"is_control": true`
- `expected_split` (optional): expected traffic allocation weights for the SRM check, either as a list of positive numbers matching `variants` order (e.g. `[0.9, 0.1]` or `[90, 10]`) or an object mapping each variant name to a positive number (e.g. `{"control": 0.8, "v1": 0.2}`). Weights are normalized to sum to `1.0`. Defaults to an equal `1/k` split when omitted. Cannot be combined with per-variant `expected_share`.
- `guardrails` (optional list): supports both **statistical per-variant guardrails** (`metric` + `variants`) and backwards-compatible **threshold guardrails** (`control` + `treatment` floats)
- `secondary_metrics` (optional list): additional `conversion_rate`, `arpu`, or `ratio` metrics evaluated with the same inference method for full-experiment readouts (informational only; never block recommendations)
- `decision_policy` (optional, Bayesian mode):
  - `enabled` (bool)
  - `bayes_prob_beats_control` (default `0.95`)
  - `max_expected_loss` (default `0.001`)
- `samples` (default `50000`, Bayesian mode)
- `random_seed` (default `7`)

`decision_thresholds` is still accepted as a legacy alias for an enabled `decision_policy`; use `decision_policy` in new inputs.

Variant row:

```json
{
  "name": "control",
  "visitors": 100000,
  "conversions": 4000,
  "is_control": true
}
```

Each variant row also accepts an optional `expected_share` (positive number) for unequal traffic splits. If used, it must be set on every variant (and not combined with top-level `expected_split`); weights are normalized to sum to `1.0`. On the CLI, `--expected-split` overrides any split from the input file or `pyproject.toml`, accepting either comma-separated weights in variant order (`--expected-split 0.9,0.1` or `--expected-split 90,10`) or `name=weight` pairs (`--expected-split control=0.8,v1=0.2`).

For `primary_metric: "arpu"`, each variant also needs `revenue_sum` and `revenue_sum_squares`. For `primary_metric: "ratio"`, each variant needs `numerator_sum`, `numerator_sum_squares`, `denominator_sum`, `denominator_sum_squares`, and `numerator_denominator_sum`.

Guardrails (`ablift example-input` prints a full multi-metric example):

- **Statistical per-variant guardrail** (recommended, especially for A/B/n tests): evaluates each treatment arm against control using the active inference method (`bayesian`, `sequential`, or `fixed`) so noisy small-sample fluctuations do not falsely block shipping, and a guardrail failure on one variant does not block shipping another winning variant. `visitors` and `is_control` are inherited from matching primary `variants` when omitted:

```json
{
  "name": "bounce_rate",
  "metric": "conversion_rate",
  "direction": "decrease",
  "max_relative_change": 0.03,
  "variants": [
    {"name": "control", "visitors": 100000, "conversions": 36000},
    {"name": "treatment_a", "visitors": 100000, "conversions": 36500}
  ]
}
```

- **Simple threshold guardrail** (backwards-compatible deterministic check comparing two point floats):

```json
{"name": "p95_latency_ms", "control": 420.0, "treatment": 430.0, "direction": "decrease", "max_relative_change": 0.05}
```

Secondary metric entry (`metric` is `"conversion_rate"`, `"arpu"`, or `"ratio"`; `visitors` and `is_control` default to the matching primary variant):

```json
{
  "name": "revenue_per_visitor",
  "metric": "arpu",
  "variants": [
    {"name": "control", "revenue_sum": 480000.0, "revenue_sum_squares": 28800000.0},
    {"name": "treatment_a", "revenue_sum": 505000.0, "revenue_sum_squares": 30500000.0}
  ]
}
```

Bayesian conversion-rate example with a decision policy:

```json
{
  "experiment_name": "homepage_cta",
  "method": "bayesian",
  "primary_metric": "conversion_rate",
  "decision_policy": {
    "enabled": true,
    "bayes_prob_beats_control": 0.95,
    "max_expected_loss": 0.001
  },
  "variants": [
    {"name": "control", "visitors": 50000, "conversions": 2000, "is_control": true},
    {"name": "v1", "visitors": 50000, "conversions": 2080, "is_control": false},
    {"name": "v2", "visitors": 50000, "conversions": 2140, "is_control": false}
  ]
}
```

Bayesian ARPU example:

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

Sequential ARPU example (re-run it on cumulative data whenever new data arrives):

```json
{
  "experiment_name": "checkout_flow",
  "method": "sequential",
  "primary_metric": "arpu",
  "alpha": 0.05,
  "sequential_tau": 0.2,
  "variants": [
    {"name": "control", "visitors": 12000, "conversions": 610, "revenue_sum": 21000, "revenue_sum_squares": 150000, "is_control": true},
    {"name": "v1", "visitors": 12000, "conversions": 640, "revenue_sum": 22400, "revenue_sum_squares": 167000, "is_control": false}
  ]
}
```

## CSV/XLSX input

`ablift analyze` reads CSV and XLSX files using **positional columns**: column names are ignored, column order determines the role. The first row must be a header row. Use `--sheet` to select a worksheet in `.xlsx`/`.xlsm` files.

### Aggregated input (one row per variant)

| Columns (in order) | Analysis |
|--------------------|----------|
| `name, visitors, conversions [, is_control]` | `conversion_rate` |
| `name, visitors, revenue_sum, revenue_sum_squares [, is_control]` | `arpu` |
| `name, visitors, conversions, revenue_sum, revenue_sum_squares [, is_control]` | `arpu` by default; `--primary-metric conversion_rate` uses the conversions column |

Notes:
- `is_control` is an optional boolean column anywhere after column 3. If its header contains "control", values may be `true`/`false`, `yes`/`no` or `1`/`0`. Under any other header it must use words (`true`/`false`, `yes`/`no`), with at least one of each, because a numeric 0/1 column could be a real metric. Without it, the first row is the control.
- A single revenue column without sum of squares is rejected with a clear error, because variance cannot be estimated. Add `revenue_sum_squares`, or provide row-level data.
- Any other number of metric columns is an error.

### Row-level input (one row per observation)

| Col 1 | Col 2 | Col 3 (optional) |
|-------|-------|-----------------|
| variant name | outcome | is_control (boolean) |

- If outcomes are all 0/1 the analysis is `conversion_rate` (the tool counts visitors and conversions per variant).
- If outcomes are continuous the analysis is `arpu`, computed exactly from the rows (`revenue_sum` and `revenue_sum_squares` per variant).
- Rows are detected as row-level when the first column has repeated values.
- The first variant to appear is the control unless a boolean control column is present.

The JSON output records what was detected in `analysis_settings.input_interpretation`:

```json
{
  "source_type": "aggregated",
  "variant_count": 3,
  "column_roles": {
    "variant": "variant_name",
    "visitors": "users",
    "conversions": "orders",
    "is_control": "is_control"
  }
}
```

### Bundled examples

- [examples/conversion_multivariant.csv](examples/conversion_multivariant.csv): 3-variant conversion rate
- [examples/arpu_bayesian.csv](examples/arpu_bayesian.csv) and [examples/arpu_bayesian.xlsx](examples/arpu_bayesian.xlsx): ARPU from aggregates
- [examples/arpu_sequential.csv](examples/arpu_sequential.csv): ARPU, use with `--method sequential`
- [examples/duration_inputs.csv](examples/duration_inputs.csv) and [examples/duration_inputs.xlsx](examples/duration_inputs.xlsx): conversion-rate duration assumptions
- [examples/duration_arpu.csv](examples/duration_arpu.csv): ARPU duration assumptions

`make demo` runs all of them end to end and writes outputs to `examples/out/` (git-ignored).

## Stdin input

Both `ablift analyze` and `ablift duration` accept `--input -` to read from stdin without creating temporary files:

```bash
# Analyze piped JSON (default format for analyze --input -)
ablift example-input | ablift analyze --input -

# Analyze piped CSV (pass --input-format csv)
cat examples/conversion_multivariant.csv | ablift analyze --input - --input-format csv

# Plan duration from piped JSON or CSV (auto-detected if --input-format is omitted)
echo '{"baseline_rate": 0.04, "relative_mde": 0.05, "daily_traffic": 50000}' | ablift duration --input -
cat examples/duration_inputs.csv | ablift duration --input - --input-format csv
```

Notes:
- For `analyze --input -`, stdin defaults to JSON. Pass `--input-format csv` for piped CSV (aggregated or row-level); `experiment_name` defaults to `"stdin"` unless `--experiment-name` is passed.
- For `duration --input -`, omitting `--input-format` auto-detects JSON when the trimmed input starts with `{`, and treats other input as CSV.
- `--input-format {json,csv}` can also be used with file paths when the extension does not indicate the format.
- `--sheet` is only valid for `.xlsx`/`.xlsm` files and is rejected with exit code 2 on stdin.

## Reusable project config

Use `[tool.ablift]` in `pyproject.toml` (read from the current directory) when you want one decision policy and one set of inference settings reused across analyses in the project.

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

## Duration planning

```bash
# Sequential conversion rate (default): mSPRT sample size
ablift duration --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3

# Classic fixed-horizon (only valid if you look once, at the end)
ablift duration --method fixed --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3

# Bayesian assurance (simulation-based)
ablift duration --method bayesian --baseline-rate 0.04 --relative-mde 0.08 --daily-traffic 60000 --max-days 60

# ARPU duration planning (pass --baseline-sd or --baseline-cv)
ablift duration --metric arpu --baseline-mean 12.5 --baseline-sd 25.0 --relative-mde 0.05 --daily-traffic 50000
ablift duration --metric arpu --method fixed --baseline-mean 12.5 --baseline-cv 2.0 --relative-mde 0.05 --daily-traffic 50000
ablift duration --metric arpu --method bayesian --baseline-mean 12.5 --baseline-sd 25.0 --relative-mde 0.1 --daily-traffic 50000 --max-expected-loss 0.05

# From a CSV/XLSX/JSON file or stdin
ablift duration --input examples/duration_inputs.csv
ablift duration --input examples/duration_inputs.xlsx --sheet Sheet1
ablift duration --input examples/duration_arpu.csv
echo '{"baseline_rate": 0.04, "relative_mde": 0.05, "daily_traffic": 50000}' | ablift duration --input -
cat examples/duration_inputs.csv | ablift duration --input - --input-format csv
```

Methods:
- `sequential` (default): smallest sample size per variant at which the always-valid mSPRT test reaches the target power. This is a conservative upper bound, since the test may stop earlier if the true effect exceeds the MDE. Diagnostics include `fixed_horizon_n_per_variant` and `sequential_inflation` (typically about 2x).
- `fixed`: classic fixed-horizon sample size (two-proportion z test for `conversion_rate`, two-sample z test for `arpu`). Only valid if you analyze once. The output reports `"method": "fixed_horizon"`.
- `bayesian`: simulation-based assurance, the probability of reaching the Bayesian decision thresholds within `--max-days` (Beta-Binomial for `conversion_rate`, Normal-Inverse-Gamma for `arpu`).

Both `sequential` and `fixed` apply a Bonferroni correction across `n_variants - 1` comparisons.

Inputs:
- `--metric` chooses `conversion_rate` (default) or `arpu`.
- For `conversion_rate`: `--baseline-rate`, `--relative-mde`, and `--daily-traffic` are required.
- For `arpu`: `--baseline-mean`, either `--baseline-sd` or `--baseline-cv` (`sd = |mean| * cv`, not both), `--relative-mde`, and `--daily-traffic` are required.
- Required assumptions can come from flags or `--input`. The command never prompts; missing values are a usage error (exit code 2).
- Optional: `--method`, `--n-variants` (default 2, including control), `--alpha` (0.05), `--power` (0.8), `--sequential-tau` (defaults to 10% of the baseline rate or mean), `--prob-threshold`, `--max-expected-loss`, `--assurance-target`, `--max-days`, `--input-format {json,csv}`, `--output`.
- `--input` takes a CSV/XLSX/JSON file or `-` for stdin with **named** fields/columns: `metric`, `method`, `baseline_rate`, `baseline_mean`, `baseline_sd`, `baseline_cv`, `relative_mde`, `daily_traffic`, `n_variants`, `alpha`, `power`, `sequential_tau`, `prob_threshold`, `max_expected_loss`, `assurance_target`, `max_days`. Unknown columns or fields are an error. CLI flags override input values.
- Settings the chosen metric or method does not use are a usage error rather than being silently ignored: `--baseline-rate` is rejected for `arpu`; `--baseline-mean`, `--baseline-sd`, and `--baseline-cv` are rejected for `conversion_rate`; `--alpha`, `--power` and `--sequential-tau` do not apply to `bayesian`; `--prob-threshold`, `--max-expected-loss`, `--assurance-target` and `--max-days` apply only to `bayesian`; `--sequential-tau` applies only to `sequential`.

Example `--input` file:

```csv
method,baseline_rate,relative_mde,daily_traffic,n_variants,alpha,power
sequential,0.04,0.05,50000,3,0.05,0.8
```

ARPU `--input` file:

```csv
metric,method,baseline_mean,baseline_sd,relative_mde,daily_traffic,n_variants,alpha,power
arpu,sequential,12.5,25.0,0.05,50000,2,0.05,0.8
```

## Statistical methods

- **Conversion rate (Bayesian):** Beta-Binomial posterior with a `Beta(1, 1)` prior. Pairwise (`probability_beats_control`, `expected_loss`) and joint multi-arm (`probability_best`, `multi_arm_expected_loss`) metrics come from the same joint posterior draw matrix across all arms; `ci_low`/`ci_high` are 95% credible intervals.
- **ARPU (Bayesian):** Normal-Inverse-Gamma posterior for the mean revenue per visitor, built from `visitors`, `revenue_sum` and `revenue_sum_squares`, with the same pairwise and joint multi-arm summaries.
- **Sequential (mSPRT):** mixture sequential probability ratio test (Johari et al. 2017) on the difference in means, giving an e-value and a Bonferroni-adjusted always-valid p-value `min(1, n_comparisons/e_value)` (so `p_value <= alpha` exactly when `significant`), plus an always-valid confidence interval. A comparison is significant when `e_value >= e_threshold`. The mixing parameter is `sequential_tau`.
- **Fixed-horizon frequentist (`--method fixed`):** two-proportion z-test with unpooled confidence intervals for `conversion_rate`, and Welch's two-sample t-test (Welch-Satterthwaite degrees of freedom) with t-based confidence intervals for `arpu`. Only valid when evaluated **once** at the planned experiment end; do not peek repeatedly.
- **Multiple variants:** in both sequential and fixed-horizon modes, alpha is Bonferroni-corrected: `alpha_per_comparison = alpha / number_of_treatments` (with `e_threshold = 1 / alpha_per_comparison` for sequential, and `p_value = min(1, raw_p_value * number_of_treatments)` plus simultaneous Bonferroni-adjusted CIs for fixed-horizon). Bayesian mode applies no alpha correction and reports joint multi-arm probabilities (`probability_best`, summing to `1.0` across control and all treatments) and `multi_arm_expected_loss = E[max_j(theta_j) - theta_i]`. Both are reported in `analysis_settings.multiple_comparisons`.
- **SRM:** chi-square goodness-of-fit of the visitor counts against the configured expected traffic proportions (defaults to an equal `1/k` split; configurable via `expected_split`, per-variant `expected_share`, or `--expected-split`). Failure when `p < 0.001`.

## Schema introspection

Agents and CI pipelines can discover all supported input fields, enums, constraints, and output contracts directly as JSON Schema (Draft 2020-12) without scraping documentation:

```bash
ablift schema analyze-input
ablift schema analyze-output
ablift schema duration-input
ablift schema duration-output
ablift schema --all
ablift schema analyze-output --output analyze-output.schema.json
```

## Agent playbook

1. Inspect contracts programmatically with `ablift schema --all` (or `ablift schema analyze-input` / `analyze-output` / `duration-input` / `duration-output`).
2. Run `ablift analyze --input ...` with a JSON, CSV, or XLSX file, or `--input -` to pipe JSON/CSV over stdin without temporary files.
3. For CSV/XLSX (or piped CSV with `--input-format csv`), format data with positional columns (see CSV/XLSX input).
4. Read `recommendation.action`, `decision_confidence`, and `risk_flags`.
5. If `action=continue_collecting_data`, re-run on the cumulative data later. With `--method sequential` this is always safe, however often you look.
6. If `action=investigate_data_quality`, resolve SRM or tracking issues before deciding to ship.
7. For planning questions, run `ablift duration` (flags, file, or `--input -`).

Exit codes:
- `0`: success
- `1`: analysis or input error (a clean `Error: ...` message on stderr, no traceback)
- `2`: CLI usage error (bad flag value, missing file, missing required duration assumptions)

Common errors and fixes:
- `Exactly one variant must have is_control=true`: mark one and only one control row.
- `conversions cannot exceed visitors`: the metric is a repeated-event count rather than a binary outcome. Redefine it as binary per unit.
- `Column '...' looks like revenue, not conversions`: add a `revenue_sum_squares` column (or use row-level data) for ARPU.
- `Unknown duration column(s)`: use only the named columns listed in Duration planning.

## Agent output contract

Output JSON starts with `"schema_version": "1"` (for both `analyze` and `duration`).

`analysis_settings` includes `multiple_comparisons` (`correction`, `n_comparisons`, and for sequential/fixed `alpha_per_comparison`), `sequential_tau`, `priors`, `decision_policy` and `input_interpretation`.

`srm` includes `passed`, `p_value`, `observed`, `expected`, `expected_proportions` (normalized to sum to `1.0`), and `reason`.

Each entry in `comparisons` has:
- `control_rate`, `treatment_rate`, `absolute_lift`, `relative_lift`
- `ci_low`, `ci_high`: interval on the **absolute** lift scale. Bayesian: 95% credible interval. Sequential: always-valid confidence interval. Fixed: Bonferroni-adjusted `(1 - alpha)` frequentist confidence interval (unpooled Wald for `conversion_rate`, Welch t for `arpu`).
- Bayesian only: `probability_beats_control`, `expected_loss` (pairwise vs. control), `probability_best` (probability this arm is best overall across control and all treatments), `multi_arm_expected_loss` (`E[max_j(theta_j) - theta_i]`), `relative_ci_low`, `relative_ci_high`. For 2-variant tests, `probability_best == probability_beats_control` and `multi_arm_expected_loss == expected_loss`.
- Sequential only: `p_value` (always-valid and Bonferroni-adjusted, `min(1, n_comparisons/e_value)`; `p_value <= alpha` exactly when `significant`), `e_value`, `e_threshold`, `significant`
- Fixed only: `p_value` (Bonferroni-adjusted, `min(1, raw_p_value * n_comparisons)`; `p_value <= alpha` exactly when `significant`), `raw_p_value`, `test_statistic` (`z` for `conversion_rate`, Welch `t` for `arpu`), `degrees_of_freedom` (Welch-Satterthwaite df for `arpu`, `null` for `conversion_rate`), `significant`

`bayesian_arms` (Bayesian mode only; `null` for other methods) is a list covering **every** arm (control and all treatments, in input order) with `variant`, `is_control`, `probability_best` (sums to `1.0` across all arms), and `multi_arm_expected_loss`.

`guardrails` is a list of evaluated guardrail results. Each entry has `name`, `direction`, `control`, `treatment`, `relative_change`, `max_relative_change`, `passed`, `reason`, plus:
- `mode`: `"statistical"` for per-variant statistical guardrails, `"threshold"` for legacy scalar guardrails
- `variant`: treatment variant name for statistical guardrails, or `null` for legacy global threshold guardrails
- `comparison`: full `ComparisonResult` statistical estimate (lift, interval, p-value / posterior probability) for statistical guardrails, or `null` for threshold guardrails

`secondary_metrics` is a list of informational metric readouts (empty list `[]` when omitted), each with `name`, `metric`, `comparisons`, and `bayesian_arms` (Bayesian mode only; `null` otherwise). Secondary metrics never block `recommendation`.

`recommendation` contains:
- `action` (`ship_<variant>`, `continue_collecting_data`, `do_not_ship`, `investigate_data_quality`, `stop_and_rollback`)
- `rationale`
- `decision_confidence` (0 to 1)
- `next_best_action`
- `risk_flags`

How the Bayesian recommendation is chosen (when an explicit `decision_policy` is enabled):
- `ship_<variant>`: evaluates every eligible treatment against `absolute_lift > 0`, `probability_beats_control >= bayes_prob_beats_control`, and `expected_loss <= max_expected_loss`; among qualifying winners, picks the Bayes-optimal arm minimizing `multi_arm_expected_loss` (tie-breaking on higher `absolute_lift`), so a noisy arm with a slightly higher point estimate cannot overshadow a qualifying winner.
- `do_not_ship`: only if every eligible treatment has `absolute_lift < 0` and `probability_beats_control <= 1 - bayes_prob_beats_control`.
- `continue_collecting_data`: otherwise, with `decision_confidence` set to the leading eligible treatment's `probability_beats_control`.

How the sequential recommendation is chosen:
- `ship_<variant>`: the best significantly positive eligible variant, even if others are not significant. `decision_confidence = 1 - p_value` of the winner. If another eligible variant is significantly worse, `other_variant_significantly_worse` is added to `risk_flags`.
- `stop_and_rollback`: only if every eligible treatment is significantly worse. `decision_confidence = 1 - p_value` of the losers (the largest of them).
- `continue_collecting_data`: otherwise. `decision_confidence = 1 - p_value` of the leading eligible comparison (the smallest adjusted p-value), so higher means closer to significance; it stays below `1 - alpha` until a decision is reached. `some_variants_significantly_worse` is added if at least one eligible variant is significantly worse.

How the fixed-horizon recommendation is chosen:
- `ship_<variant>`: the best significantly positive eligible variant (`p_value <= alpha`). `decision_confidence = 1 - p_value` of the winner. Adds `other_variant_significantly_worse` if another eligible treatment is significantly worse.
- `do_not_ship`: if every eligible treatment is significantly worse (`decision_confidence = 1 - max(p_value)`), or if no eligible treatment reached significance at the planned look (`decision_confidence = 1 - min(p_value)`, reminding callers not to peek repeatedly with `--method fixed`).

Applies to all methods, checked first:
- SRM failure gives `investigate_data_quality`, flag `srm_detected`, `decision_confidence = 1 - SRM p-value`.
- Legacy global threshold guardrail failure, or per-variant statistical guardrail failures on **all** treatments, gives `do_not_ship`, flag `guardrail_failure`, `decision_confidence = 1.0`.
- Per-variant statistical guardrail failure on a **subset** of treatments excludes only those failing treatment(s) from shipping; another winning treatment can still receive `ship_<variant>` with `other_variant_guardrail_failure` in `risk_flags` (or `some_variants_guardrail_failure` if remaining treatments are inconclusive).

If no explicit Bayesian decision policy is provided, `recommendation` is `null` and `analysis_settings.decision_policy` shows that no automated decision policy was applied.

## Limitations

- Aggregated ARPU inputs cannot be winsorized after aggregation (only a CV-based `heavy_tailed_revenue` warning is emitted); pass row-level revenue with `--winsorize Q` (and optionally `--bootstrap-samples N`) when whale outliers are present.
- mSPRT guarantees assume you re-run the analysis on cumulative data as it arrives. The tool keeps no state between looks.
- No variance reduction (CUPED) yet.

## License

MIT. See [LICENSE](LICENSE).

## Development

```bash
uv sync                  # installs the test and lint groups
uv run pytest            # test suite
uv run ruff check .      # lint
uv run ty check          # type check
make demo                # run all bundled examples into examples/out/
uv run ablift doctor     # environment check (also --json-output, --strict)
```
