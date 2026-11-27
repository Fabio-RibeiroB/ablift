# Inputs

How to give `ablift analyze` your experiment data: which metric model to use, the JSON contract, positional CSV/XLSX layouts, stdin, and project-level defaults.

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
- Optionally pass `--bootstrap-samples N` (`>= 100`, and at most 1,000,000,000 resampled rows in total, about 10 s of CPU; larger values are a usage error that states the maximum for your data) on row-level ARPU inputs to compute a percentile bootstrap confidence interval cross-check in `analysis_settings.input_interpretation.bootstrap_ci`.

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
- inference settings: method, metric, priors, alpha, posterior sample count, random seed
- decision policy: opt-in rule-based recommendations (off by default across all methods)

By default, `ablift analyze` returns statistical estimates (`comparisons`, `bayesian_arms`), `srm`, `guardrails`, `secondary_metrics`, and top-level `risk_flags`, leaving `recommendation` as `null`. Pass `--recommend` (or set `decision_policy.enabled = true` in JSON or `pyproject.toml`) to opt in to an automated recommendation on any method (`bayesian`, `sequential`, or `fixed`). Use `--no-recommend` to disable a policy configured in the input file or `pyproject.toml`.

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
- posterior sample count defaults to `50000`, and `samples × variants` may not exceed 20,000,000 (about 160 MB per array); larger values are rejected with the maximum for your variant count
- random seed defaults to `7`
- conversion-rate prior is `Beta(1, 1)`
- ARPU prior is a weak `Normal-Inverse-Gamma` prior with `mu0=0`, `kappa0=1e-6`, `alpha0=1`, `beta0=1`
- ratio posterior uses an asymptotic normal approximation parameterized by the Delta-method variance (`delta_normal_asymptotic`)

Opt-in recommendations (all methods):
- `--recommend` / `--no-recommend` enables or disables automated recommendation output (works with `bayesian`, `sequential`, and `fixed`).
- In Bayesian mode, passing `--prob-threshold` (default `0.95`) or `--max-expected-loss` (default `0.001`) also implies `--recommend`.

Sequential defaults:
- `alpha` defaults to `0.05`
- `sequential_tau` (CLI `--sequential-tau`) is the mSPRT mixing parameter: the prior standard deviation of the effect on the absolute scale. It defaults to 10% of the control baseline (conversion rate, ARPU, or ratio). Pass it explicitly if you expect much larger or smaller effects.
- You may look at results as often as you like: the false-positive rate stays at `alpha`.

Fixed-horizon defaults (`--method fixed`):
- `alpha` defaults to `0.05` (Bonferroni-adjusted across `n_variants - 1` comparisons, matching `ablift duration --method fixed`).
- **Single-look assumption:** fixed-horizon p-values and confidence intervals are only valid when evaluated **once** at the planned end of the test. Do not peek repeatedly with `--method fixed`; use `--method sequential` for ongoing monitoring.

Method-scoped CLI flag validation:
- `--sequential-tau` is rejected with exit code 2 unless `--method sequential` is active.
- `--samples`, `--prob-threshold`, and `--max-expected-loss` (and `--random-seed` unless `--bootstrap-samples` is also passed) are rejected with exit code 2 unless `--method bayesian` is active.

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
- `sequential_tau` (optional float, sequential mode only; rejected for `bayesian` and `fixed`; default 10% of the control baseline)
- `variants` (list): exactly one row must include `"is_control": true`
- `expected_split` (optional): expected traffic allocation weights for the SRM check, either as a list of positive numbers matching `variants` order (e.g. `[0.9, 0.1]` or `[90, 10]`) or an object mapping each variant name to a positive number (e.g. `{"control": 0.8, "v1": 0.2}`). Weights are normalized to sum to `1.0`. Defaults to an equal `1/k` split when omitted. Cannot be combined with per-variant `expected_share`.
- `guardrails` (optional list): supports both **statistical per-variant guardrails** (`metric` + `variants`) and backwards-compatible **threshold guardrails** (`control` + `treatment` floats)
- `secondary_metrics` (optional list): additional `conversion_rate`, `arpu`, or `ratio` metrics evaluated with the same inference method for full-experiment readouts (informational only; never block recommendations)
- `decision_policy` (optional, opt-in across all methods):
  - `enabled` (bool, default `false` when omitted)
  - `bayes_prob_beats_control` (Bayesian mode only, default `0.95`)
  - `max_expected_loss` (Bayesian mode only, default `0.001`)
- `samples` (default `50000`, Bayesian mode only; rejected for `sequential` and `fixed`)
- `random_seed` (default `7`, Bayesian mode only; rejected for `sequential` and `fixed`). On the CLI, `--random-seed` also seeds the row-level ARPU `--bootstrap-samples` CI for any method.

These method-scoped fields mirror the CLI, which rejects `--samples`, `--random-seed`, and `--sequential-tau` for methods that ignore them. When `--method` overrides the file's `method`, the file's settings for its own method are dropped instead of rejected.

`decision_thresholds` is still accepted as a legacy alias for an enabled Bayesian `decision_policy`; use `decision_policy` in new inputs.

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

For **CUPED variance reduction** on `conversion_rate` or `arpu`, include `covariate_sum`, `covariate_sum_squares`, and `outcome_covariate_sum` on every variant (using a strictly pre-experiment covariate `X`), or pass `--covariate-column <name|1-based-index>` on row-level CSV/XLSX inputs. `ablift` estimates a within-arm pooled `theta` (invariant to treatment effects), runs `fixed`, `sequential`, or `bayesian` inference on the CUPED-adjusted arm moments, reports diagnostics (`theta`, `r_squared`, `variance_reduction_pct`, `covariate_balance_p_value`) in `analysis_settings.input_interpretation.cuped`, and flags `covariate_imbalance` if pre-experiment covariate balance fails (`p < 0.001`).

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

- [examples/checkout_test.csv](../examples/checkout_test.csv): 3-variant conversion rate with a clear winner (the README demo)
- [examples/conversion_multivariant.csv](../examples/conversion_multivariant.csv): 3-variant conversion rate
- [examples/arpu_bayesian.csv](../examples/arpu_bayesian.csv) and [examples/arpu_bayesian.xlsx](../examples/arpu_bayesian.xlsx): ARPU from aggregates
- [examples/arpu_sequential.csv](../examples/arpu_sequential.csv): ARPU, use with `--method sequential`
- [examples/duration_inputs.csv](../examples/duration_inputs.csv) and [examples/duration_inputs.xlsx](../examples/duration_inputs.xlsx): conversion-rate duration assumptions
- [examples/duration_arpu.csv](../examples/duration_arpu.csv): ARPU duration assumptions

`make demo` runs all of them end to end and writes outputs to `examples/out/` (git-ignored).

## Stdin input

Both `ablift analyze` and `ablift duration` accept `--input -` to read from stdin without creating temporary files:

```bash
# Analyze piped JSON (default format for analyze --input -)
ablift example-input | ablift analyze --input -

# Analyze piped CSV (pass --input-format csv)
cat examples/conversion_multivariant.csv | ablift analyze --input - --input-format csv

# Plan duration from piped JSON or CSV (auto-detected if --input-format is omitted)
echo '{"method": "sequential", "baseline_rate": 0.04, "relative_mde": 0.05, "daily_traffic": 50000}' | ablift duration --input -
cat examples/duration_inputs.csv | ablift duration --input - --input-format csv
```

Notes:
- For `analyze --input -`, stdin defaults to JSON. Pass `--input-format csv` for piped CSV (aggregated or row-level); `experiment_name` defaults to `"stdin"` unless `--experiment-name` is passed.
- For `duration --input -`, omitting `--input-format` auto-detects JSON when the trimmed input starts with `{`, and treats other input as CSV.
- `--input-format {json,csv}` can also be used with file paths when the extension does not indicate the format.
- `--sheet` is only valid for `.xlsx`/`.xlsm` files and is rejected with exit code 2 on stdin.

## Reusable project config

Use `[tool.ablift]` in `pyproject.toml` (read from the current directory) when you want one decision policy and one set of inference settings reused across analyses in the project. Only `ablift analyze` reads it; `duration` and `power` do not.

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

Method-scoped config values (`samples`, `random_seed`, `sequential_tau`) only apply to runs of the method that uses them, so a project config tuned for Bayesian runs does not block `--method sequential` or `--method fixed`.

Every `analyze` output records the file that was applied in `analysis_settings.config_source` (its absolute path, or `null` when no `[tool.ablift]` table was found). Pass `--no-config` to ignore `pyproject.toml` entirely, for example when running from a directory whose project config is not meant for this analysis:

```bash
ablift analyze --input examples/checkout_test.csv --no-config
```
