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
- Multi-variant (`A/B/n`) comparisons against one control
- Guardrail checks (latency, bounce rate, error rate, etc.)
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

Generate a JSON input template and analyze it:

```bash
ablift example-input > input.json
ablift analyze --input input.json --output output.json --report report.md
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

Plan how long a test needs to run:

```bash
ablift duration --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3
```

## Command summary

- `ablift analyze`: analyze `.json`, `.csv`, `.xlsx`, or `.xlsm` input
- `ablift duration`: estimate runtime needed for a conversion-rate test
- `ablift doctor`: verify the environment and required dependencies (`--json-output`, `--strict`)
- `ablift example-input`: print a starter JSON payload

Useful `analyze` flags: `--method {bayesian,sequential}`, `--primary-metric {conversion_rate,arpu}`, `--alpha`, `--sequential-tau`, `--samples`, `--random-seed`, `--enable-recommendation/--no-recommendation`, `--prob-threshold`, `--max-expected-loss`, `--experiment-name`, `--sheet`, `--output`, `--report`. Flags override values in the input file.

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

Use `primary_metric: "arpu"` (average revenue per visitor) when you have aggregate revenue statistics per variant:
- `visitors`
- `revenue_sum`
- `revenue_sum_squares` (needed to estimate variance)

If your metric can happen multiple times per unit and the count can exceed the denominator, it is not a fit for the `conversion_rate` model. For example:
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
- decision policy: thresholds used to turn Bayesian estimates into actions

For Bayesian runs, recommendations are optional. If you do not provide a decision policy, `ablift` reports the posterior estimates but leaves `recommendation` as `null`. Sequential runs always produce a recommendation; the decision policy does not apply to them.

For analysis inputs:
- at least 2 variants are required
- exactly 1 variant must have `is_control: true`
- `method` must be `bayesian` or `sequential`
- `primary_metric` must be `conversion_rate` or `arpu`
- `visitors` must be a positive integer
- `conversions` must be a non-negative integer and cannot exceed `visitors`
- `alpha` must be in `(0, 1)`; `sequential_tau` must be positive
- for `arpu`, each variant must also include `revenue_sum` and `revenue_sum_squares`

Bayesian defaults:
- posterior sample count defaults to `50000`
- random seed defaults to `7`
- conversion-rate prior is `Beta(1, 1)`
- ARPU prior is a weak `Normal-Inverse-Gamma` prior with `mu0=0`, `kappa0=1e-6`, `alpha0=1`, `beta0=1`

Bayesian recommendations require an explicit decision policy, either in the input payload, via `[tool.ablift]` in `pyproject.toml`, or via CLI flags:
- `--enable-recommendation`
- `--prob-threshold` (default `0.95`)
- `--max-expected-loss` (default `0.001`)

Sequential defaults:
- `alpha` defaults to `0.05`
- `sequential_tau` (CLI `--sequential-tau`) is the mSPRT mixing parameter: the prior standard deviation of the effect on the absolute scale. It defaults to 10% of the control baseline (conversion rate or ARPU). Pass it explicitly if you expect much larger or smaller effects.
- You may look at results as often as you like: the false-positive rate stays at `alpha`.

Internal field meanings:
- `name`: variant label
- `visitors`: denominator units exposed to the experiment
- `conversions`: success units for the primary outcome
- `is_control`: whether the row is the control variant

## JSON input

Top-level fields:
- `experiment_name` (str)
- `method` (`"bayesian"` or `"sequential"`)
- `primary_metric` (`"conversion_rate"` or `"arpu"`)
- `alpha` (float, default `0.05`)
- `sequential_tau` (optional float, sequential mode; default 10% of the control baseline)
- `variants` (list): exactly one row must include `"is_control": true`
- `guardrails` (optional list)
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

For `primary_metric: "arpu"`, each variant also needs `revenue_sum` and `revenue_sum_squares`.

Guardrail row (`ablift example-input` prints a full example):

```json
{"name": "bounce_rate", "control": 0.36, "treatment": 0.365, "direction": "decrease", "max_relative_change": 0.03}
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
- [examples/duration_inputs.csv](examples/duration_inputs.csv) and [examples/duration_inputs.xlsx](examples/duration_inputs.xlsx): duration assumptions

`make demo` runs all of them end to end and writes outputs to `examples/out/` (git-ignored).

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
# Sequential (default): mSPRT sample size
ablift duration --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3

# Classic fixed-horizon (only valid if you look once, at the end)
ablift duration --method fixed --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3

# Bayesian assurance (simulation-based)
ablift duration --method bayesian --baseline-rate 0.04 --relative-mde 0.08 --daily-traffic 60000 --max-days 60

# From a CSV/XLSX with named columns
ablift duration --input examples/duration_inputs.csv
ablift duration --input examples/duration_inputs.xlsx --sheet Sheet1
```

Methods:
- `sequential` (default): smallest sample size per variant at which the always-valid mSPRT test reaches the target power. This is a conservative upper bound, since the test may stop earlier if the true effect exceeds the MDE. Diagnostics include `fixed_horizon_n_per_variant` and `sequential_inflation` (typically about 2x).
- `fixed`: classic fixed-horizon two-proportion sample size. Only valid if you analyze once. The output reports `"method": "fixed_horizon"`.
- `bayesian`: simulation-based assurance, the probability of reaching the Bayesian decision thresholds within `--max-days`.

Both `sequential` and `fixed` apply a Bonferroni correction across `n_variants - 1` comparisons.

Inputs:
- `--baseline-rate`, `--relative-mde` and `--daily-traffic` are required, from flags or `--input`. The command never prompts; missing values are a usage error (exit code 2).
- Optional: `--method`, `--n-variants` (default 2, including control), `--alpha` (0.05), `--power` (0.8), `--sequential-tau`, `--prob-threshold`, `--max-expected-loss`, `--assurance-target`, `--max-days`, `--output`.
- `--input` takes a CSV/XLSX whose first row has **named** columns: `method`, `baseline_rate`, `relative_mde`, `daily_traffic`, `n_variants`, `alpha`, `power`, `sequential_tau`, `prob_threshold`, `max_expected_loss`, `assurance_target`, `max_days`. Unknown columns are an error. CLI flags override file values.
- Settings the chosen method does not use are a usage error rather than being silently ignored: `--alpha`, `--power` and `--sequential-tau` do not apply to `bayesian`; `--prob-threshold`, `--max-expected-loss`, `--assurance-target` and `--max-days` apply only to `bayesian`; `--sequential-tau` applies only to `sequential`.
- Duration planning is conversion-rate only.

Example `--input` file:

```csv
method,baseline_rate,relative_mde,daily_traffic,n_variants,alpha,power
sequential,0.04,0.05,50000,3,0.05,0.8
```

## Statistical methods

- **Conversion rate (Bayesian):** Beta-Binomial posterior with a `Beta(1, 1)` prior. Probability to beat control and expected loss come from posterior sampling; `ci_low`/`ci_high` are 95% credible intervals.
- **ARPU (Bayesian):** Normal-Inverse-Gamma posterior for the mean revenue per visitor, built from `visitors`, `revenue_sum` and `revenue_sum_squares`.
- **Sequential (mSPRT):** mixture sequential probability ratio test (Johari et al. 2017) on the difference in means, giving an e-value and a Bonferroni-adjusted always-valid p-value `min(1, n_comparisons/e_value)` (so `p_value <= alpha` exactly when `significant`), plus an always-valid confidence interval. A comparison is significant when `e_value >= e_threshold`. The mixing parameter is `sequential_tau`.
- **Multiple variants:** in sequential mode alpha is Bonferroni-corrected: `alpha_per_comparison = alpha / number_of_treatments` and `e_threshold = 1 / alpha_per_comparison`. Bayesian mode applies no correction. Both are reported in `analysis_settings.multiple_comparisons`.
- **SRM:** chi-square goodness-of-fit of the visitor counts against an equal split. Failure when `p < 0.001`.

## Agent playbook

1. Run `ablift analyze --input ...` with a JSON, CSV, or XLSX file.
2. For CSV/XLSX, format data with positional columns (see CSV/XLSX input).
3. Read `recommendation.action`, `decision_confidence`, and `risk_flags`.
4. If `action=continue_collecting_data`, re-run on the cumulative data later. With `--method sequential` this is always safe, however often you look.
5. If `action=investigate_data_quality`, resolve SRM or tracking issues before deciding to ship.
6. For planning questions, run `ablift duration`.

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

`analysis_settings` includes `multiple_comparisons` (`correction`, `n_comparisons`, and for sequential `alpha_per_comparison`), `sequential_tau`, `priors`, `decision_policy` and `input_interpretation`.

Each entry in `comparisons` has:
- `control_rate`, `treatment_rate`, `absolute_lift`, `relative_lift`
- `ci_low`, `ci_high`: interval on the **absolute** lift scale. Bayesian: 95% credible interval. Sequential: always-valid confidence interval.
- Bayesian only: `probability_beats_control`, `expected_loss`, `relative_ci_low`, `relative_ci_high`
- Sequential only: `p_value` (always-valid and Bonferroni-adjusted, `min(1, n_comparisons/e_value)`; `p_value <= alpha` exactly when `significant`), `e_value`, `e_threshold`, `significant`

`recommendation` contains:
- `action` (`ship_<variant>`, `continue_collecting_data`, `do_not_ship`, `investigate_data_quality`, `stop_and_rollback`)
- `rationale`
- `decision_confidence` (0 to 1)
- `next_best_action`
- `risk_flags`

How the sequential recommendation is chosen:
- `ship_<variant>`: the best significantly positive variant, even if others are not significant. `decision_confidence = 1 - p_value` of the winner. If another variant is significantly worse, `other_variant_significantly_worse` is added to `risk_flags`.
- `stop_and_rollback`: only if every treatment is significantly worse. `decision_confidence = 1 - p_value` of the losers (the largest of them).
- `continue_collecting_data`: otherwise. `decision_confidence = 1 - p_value` of the leading comparison (the smallest adjusted p-value), so higher means closer to significance; it stays below `1 - alpha` until a decision is reached. `some_variants_significantly_worse` is added if at least one variant is significantly worse.

Applies to both methods, checked first:
- SRM failure gives `investigate_data_quality`, flag `srm_detected`, `decision_confidence = 1 - SRM p-value`.
- Guardrail failure gives `do_not_ship`, flag `guardrail_failure`, `decision_confidence = 1.0` (a deterministic rule, not a statistical test).

If no explicit Bayesian decision policy is provided, `recommendation` is `null` and `analysis_settings.decision_policy` shows that no automated decision policy was applied.

## Limitations

- ARPU uses a normal approximation of the mean. Heavy-tailed revenue may need winsorising upstream.
- SRM assumes an equal traffic split across variants.
- mSPRT guarantees assume you re-run the analysis on cumulative data as it arrives. The tool keeps no state between looks.
- No variance reduction (CUPED) yet.
- Duration planning is conversion-rate only.

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
