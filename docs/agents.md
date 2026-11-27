# Using ablift from agents and automation

Machine-readable contracts, a step-by-step playbook, exit codes, and the full meaning of every output field.

## Schema introspection

Agents and CI pipelines can discover all supported input fields, enums, constraints, and output contracts directly as JSON Schema (Draft 2020-12) without scraping documentation:

```bash
ablift schema analyze-input
ablift schema analyze-output
ablift schema duration-input
ablift schema duration-output
ablift schema power-output
ablift schema --all
ablift schema analyze-output --output analyze-output.schema.json
```

## Agent playbook

1. Inspect contracts programmatically with `ablift schema --all` (or `ablift schema analyze-input` / `analyze-output` / `duration-input` / `duration-output` / `power-output`).
2. Run `ablift analyze --input ...` with a JSON, CSV, or XLSX file, or `--input -` to pipe JSON/CSV over stdin without temporary files.
3. For CSV/XLSX (or piped CSV with `--input-format csv`), format data with positional columns (see [CSV/XLSX input](inputs.md#csvxlsx-input)).
4. Inspect `srm.passed`, `guardrails_passed`, top-level `risk_flags`, and `comparisons` (plus `bayesian_arms` in Bayesian mode).
5. If `srm_detected` or `covariate_imbalance` is in `risk_flags`, resolve tracking or randomization issues before acting on lift estimates.
6. If you want an opt-in rule-based decision summary (`ship_<variant>`, `continue_collecting_data`, `do_not_ship`, `stop_and_rollback`, `investigate_data_quality`), pass `--recommend` (or set `decision_policy.enabled = true`) and read `recommendation`. With `--method sequential`, re-running on cumulative data is always safe, however often you look.
7. For planning questions, run `ablift duration` (flags, file, or `--input -`). When the sample size is already fixed, run `ablift power` to get the power for a given lift, or the smallest detectable lift.

When aggregating raw exposure and event tables in SQL before piping into `ablift`, start from the [reference SQL query](sql.md).

Before running analyses in CI or an agent sandbox, check the environment:

```bash
ablift doctor --strict        # exit 1 if any check fails
ablift doctor --json-output   # machine-readable results
```

Exit codes:
- `0`: success
- `1`: analysis or input error (a clean `Error: ...` message on stderr, no traceback)
- `2`: CLI usage error (bad flag value, missing file, missing required duration assumptions)

Common errors and fixes:
- `Exactly one variant must have is_control=true`: mark one and only one control row.
- `conversions cannot exceed visitors`: the metric is a repeated-event count rather than a binary outcome. Redefine it as binary per unit.
- `Column '...' looks like revenue, not conversions`: add a `revenue_sum_squares` column (or use row-level data) for ARPU.
- `Unknown duration column(s)`: use only the named columns listed in [Duration planning](duration.md).

## Agent output contract

Output JSON starts with `"schema_version": "1"` (for `analyze`, `duration`, and `power`).

`analysis_settings` includes `multiple_comparisons` (`correction`, `n_comparisons`, and for sequential/fixed `alpha_per_comparison`), `sequential_tau`, `priors`, `decision_policy`, `input_interpretation`, and `config_source`.

`analysis_settings.config_source` is the absolute path of the `pyproject.toml` whose `[tool.ablift]` table was applied, or `null` when there was none (no `pyproject.toml` in the current directory, no `[tool.ablift]` table, or `--no-config`). `ablift analyze` reads that file from the directory it runs in, so pass `--no-config` when you want results that depend only on the input and flags.

`srm` includes `passed`, `p_value`, `observed`, `expected`, `expected_proportions` (normalized to sum to `1.0`), and `reason`.

`risk_flags` is a top-level list of strings that is **always present** (empty when nothing was flagged), even when `recommendation` is `null` (e.g. `srm_detected`, `guardrail_failure`, `some_variants_guardrail_failure`, `covariate_imbalance`, `heavy_tailed_revenue`, `sparse_conversions`).

Each entry in `comparisons` has:
- `control_rate`, `treatment_rate`, `absolute_lift`, `relative_lift`
- `ci_low`, `ci_high`: interval on the **absolute** lift scale. Bayesian: 95% credible interval. Sequential: always-valid confidence interval. Fixed: Bonferroni-adjusted `(1 - alpha)` frequentist confidence interval (Newcombe hybrid score for `conversion_rate`, Welch t for `arpu`).
- Bayesian only: `probability_beats_control`, `expected_loss` (pairwise vs. control), `probability_best` (probability this arm is best overall across control and all treatments), `multi_arm_expected_loss` (`E[max_j(theta_j) - theta_i]`), `relative_ci_low`, `relative_ci_high`. For 2-variant tests, `probability_best == probability_beats_control` and `multi_arm_expected_loss == expected_loss`.
- Sequential only: `p_value` (always-valid and Bonferroni-adjusted, `min(1, n_comparisons/e_value)`; `p_value <= alpha` exactly when `significant`), `e_value`, `e_threshold`, `significant`
- Fixed only: `p_value` (Bonferroni-adjusted, `min(1, raw_p_value * n_comparisons)`; `p_value <= alpha` exactly when `significant`), `raw_p_value`, `test_statistic` (`z` for `conversion_rate` or `null` on Fisher's exact sparse fallback, Welch `t` for `arpu`), `degrees_of_freedom` (Welch-Satterthwaite df for `arpu`, `null` for `conversion_rate`), `significant`

`bayesian_arms` (Bayesian mode only; `null` for other methods) is a list covering **every** arm (control and all treatments, in input order) with `variant`, `is_control`, `probability_best` (sums to `1.0` across all arms), and `multi_arm_expected_loss`.

`guardrails` is a list of evaluated guardrail results. Each entry has `name`, `direction`, `control`, `treatment`, `relative_change`, `max_relative_change`, `passed`, `reason`, plus:
- `mode`: `"statistical"` for per-variant statistical guardrails, `"threshold"` for legacy scalar guardrails
- `variant`: treatment variant name for statistical guardrails, or `null` for legacy global threshold guardrails
- `comparison`: full `ComparisonResult` statistical estimate (lift, interval, p-value / posterior probability) for statistical guardrails, or `null` for threshold guardrails

`secondary_metrics` is a list of informational metric readouts (empty list `[]` when omitted), each with `name`, `metric`, `comparisons`, and `bayesian_arms` (Bayesian mode only; `null` otherwise). Secondary metrics never block `recommendation`.

`recommendation` is `null` by default across all methods (`bayesian`, `sequential`, and `fixed`). When opted in via `--recommend` or `decision_policy.enabled = true`, it contains:
- `action` (`ship_<variant>`, `continue_collecting_data`, `do_not_ship`, `investigate_data_quality`, `stop_and_rollback`)
- `rationale`
- `decision_confidence` (0 to 1)
- `next_best_action`
- `risk_flags` (mirrors top-level `risk_flags`)

How the Bayesian recommendation is chosen (when opted in):
- `ship_<variant>`: evaluates every eligible treatment against `absolute_lift > 0`, `probability_beats_control >= bayes_prob_beats_control`, and `expected_loss <= max_expected_loss`; among qualifying winners, picks the Bayes-optimal arm minimizing `multi_arm_expected_loss` (tie-breaking on higher `absolute_lift`), so a noisy arm with a slightly higher point estimate cannot overshadow a qualifying winner.
- `do_not_ship`: only if every eligible treatment has `absolute_lift < 0` and `probability_beats_control <= 1 - bayes_prob_beats_control`.
- `continue_collecting_data`: otherwise, with `decision_confidence` set to the leading eligible treatment's `probability_beats_control`.

How the sequential recommendation is chosen (when opted in):
- `ship_<variant>`: the best significantly positive eligible variant, even if others are not significant. `decision_confidence = 1 - p_value` of the winner. If another eligible variant is significantly worse, `other_variant_significantly_worse` is added to `risk_flags`.
- `stop_and_rollback`: only if every eligible treatment is significantly worse. `decision_confidence = 1 - p_value` of the losers (the largest of them).
- `continue_collecting_data`: otherwise. `decision_confidence = 1 - p_value` of the leading eligible comparison (the smallest adjusted p-value), so higher means closer to significance; it stays below `1 - alpha` until a decision is reached. `some_variants_significantly_worse` is added if at least one eligible variant is significantly worse.

How the fixed-horizon recommendation is chosen (when opted in):
- `ship_<variant>`: the best significantly positive eligible variant (`p_value <= alpha`). `decision_confidence = 1 - p_value` of the winner. Adds `other_variant_significantly_worse` if another eligible treatment is significantly worse.
- `do_not_ship`: if every eligible treatment is significantly worse (`decision_confidence = 1 - max(p_value)`), or if no eligible treatment reached significance at the planned look (`decision_confidence = 1 - min(p_value)`, reminding callers not to peek repeatedly with `--method fixed`).

Applies to all methods when recommendations are opted in, checked first:
- SRM failure gives `investigate_data_quality`, flag `srm_detected`, `decision_confidence = 1 - SRM p-value`.
- Legacy global threshold guardrail failure, or per-variant statistical guardrail failures on **all** treatments, gives `do_not_ship`, flag `guardrail_failure`, `decision_confidence = 1.0`.
- Per-variant statistical guardrail failure on a **subset** of treatments excludes only those failing treatment(s) from shipping; another winning treatment can still receive `ship_<variant>` with `other_variant_guardrail_failure` in `risk_flags` (or `some_variants_guardrail_failure` if remaining treatments are inconclusive).
