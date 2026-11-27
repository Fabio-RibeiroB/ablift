# Duration and power

`ablift duration` estimates how many visitors, and how many days, a test needs before it can reach a decision. [`ablift power`](#power-and-minimum-detectable-effect) answers the reverse questions for a sample size you already have.

```bash
# Sequential conversion rate: mSPRT sample size
ablift duration --method sequential --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3

# Classic fixed-horizon (only valid if you look once, at the end)
ablift duration --method fixed --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3

# Unequal traffic splits (e.g. 90/10 canary ramp or 80/10/10 holdout; control first)
ablift duration --method fixed --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --expected-split 0.9,0.1
ablift duration --method sequential --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3 --expected-split 80,10,10

# Bayesian assurance (simulation-based)
ablift duration --method bayesian --baseline-rate 0.04 --relative-mde 0.08 --daily-traffic 60000 --max-days 60

# ARPU duration planning (pass --baseline-sd or --baseline-cv)
ablift duration --metric arpu --method sequential --baseline-mean 12.5 --baseline-sd 25.0 --relative-mde 0.05 --daily-traffic 50000
ablift duration --metric arpu --method fixed --baseline-mean 12.5 --baseline-cv 2.0 --relative-mde 0.05 --daily-traffic 50000
ablift duration --metric arpu --method bayesian --baseline-mean 12.5 --baseline-sd 25.0 --relative-mde 0.1 --daily-traffic 50000 --max-expected-loss 0.05

# From a CSV/XLSX/JSON file or stdin (the file's `method` field counts; --method overrides it)
ablift duration --input examples/duration_inputs.csv
ablift duration --input examples/duration_inputs.xlsx --sheet Sheet1
ablift duration --input examples/duration_arpu.csv
echo '{"method": "sequential", "baseline_rate": 0.04, "relative_mde": 0.05, "daily_traffic": 50000}' | ablift duration --input -
cat examples/duration_inputs.csv | ablift duration --input - --input-format csv
```

Methods (there is no default: pass `--method` or a `method` input field, otherwise it is a usage error, exit code 2):
- `sequential`: smallest sample size at which the always-valid mSPRT test reaches the target power for every treatment arm. This is a conservative upper bound, since the test may stop earlier if the true effect exceeds the MDE. Diagnostics include `fixed_horizon_n_per_variant` and `sequential_inflation` (typically about 2x).
- `fixed`: classic fixed-horizon sample size (two-proportion z test for `conversion_rate`, two-sample z test for `arpu`). Only valid if you analyze once. The output reports `"method": "fixed_horizon"`.
- `bayesian`: simulation-based assurance, the probability of reaching the Bayesian decision thresholds within `--max-days`. Simulations are vectorised and the earliest qualifying day is found by doubling then bisecting, so even a 365-day horizon plans in about a second (Beta-Binomial for `conversion_rate`, Normal-Inverse-Gamma for `arpu`).

Both `sequential` and `fixed` apply a Bonferroni correction across `n_variants - 1` comparisons.

Inputs:
- `--metric` chooses `conversion_rate` (default) or `arpu`.
- For `conversion_rate`: `--baseline-rate`, `--relative-mde`, and `--daily-traffic` are required.
- For `arpu`: `--baseline-mean`, either `--baseline-sd` or `--baseline-cv` (`sd = |mean| * cv`, not both), `--relative-mde`, and `--daily-traffic` are required.
- Required assumptions can come from flags or `--input`. The command never prompts; missing values are a usage error (exit code 2).
- `--method` is required, as a flag or a `method` field in `--input`. Use `sequential` if you will check results as they come in, `fixed` for a single planned look, `bayesian` to plan a Bayesian analysis. Pick the method you will analyze with, since `analyze` defaults to `bayesian`.
- Optional: `--n-variants` (default 2, including control), `--expected-split` (comma-separated positive weights in variant order, control first, e.g. `0.9,0.1` or `80,10,10`; must have length `n_variants` and are normalized to sum to 1), `--alpha` (0.05), `--power` (0.8), `--sequential-tau` (defaults to 10% of the baseline rate or mean), `--prob-threshold`, `--max-expected-loss`, `--assurance-target`, `--max-days`, `--input-format {json,csv}`, `--output`.
- `--input` takes a CSV/XLSX/JSON file or `-` for stdin with **named** fields/columns: `metric`, `method`, `baseline_rate`, `baseline_mean`, `baseline_sd`, `baseline_cv`, `relative_mde`, `daily_traffic`, `n_variants`, `expected_split` (JSON array of positive numbers like `[0.9, 0.1]` or comma-separated string like `"0.9,0.1"` in CSV/XLSX), `alpha`, `power`, `sequential_tau`, `prob_threshold`, `max_expected_loss`, `assurance_target`, `max_days`. Unknown columns or fields are an error. CLI flags override input values.
- Settings the chosen metric or method does not use are a usage error rather than being silently ignored: `--baseline-rate` is rejected for `arpu`; `--baseline-mean`, `--baseline-sd`, and `--baseline-cv` are rejected for `conversion_rate`; `--alpha`, `--power` and `--sequential-tau` do not apply to `bayesian`; `--prob-threshold`, `--max-expected-loss`, `--assurance-target` and `--max-days` apply only to `bayesian`; `--sequential-tau` applies only to `sequential`.
- Output (`ablift schema duration-output`) reports `estimated_days` (`n_total / daily_traffic` for frequentist modes), `n_by_variant` (per-arm sample sizes in variant order), `n_total`, and `n_per_variant` (control arm sample size, equal to every arm under an equal split). With unequal splits, the plan sizes for the worst-case treatment arm so every treatment reaches target power.

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

## Power and minimum detectable effect

`ablift power` takes a fixed sample size (`--n-per-variant` for an equal split, or `--n-total` with optional `--expected-split`) and solves for whichever quantity you leave out:

- give `--relative-mde` to get the **power**: the chance the test detects a lift of that size;
- leave it out to get the **minimum detectable effect (MDE)**: the smallest relative lift detectable at `--power` (default `0.8`).

```bash
# What power do 10,000 users per variant give for a +10% lift on an 8% baseline?
ablift power --method fixed --baseline-rate 0.08 --n-per-variant 10000 --relative-mde 0.1

# What is the smallest lift 10,000 users per variant can detect with 80% power?
ablift power --method fixed --baseline-rate 0.08 --n-per-variant 10000

# Unequal traffic split: pass total users with --n-total and --expected-split (control first)
ablift power --method fixed --baseline-rate 0.08 --n-total 20000 --expected-split 0.9,0.1 --relative-mde 0.1
ablift power --method fixed --baseline-rate 0.08 --n-total 100000 --n-variants 3 --expected-split 80,10,10

# ARPU, 90% power
ablift power --method fixed --metric arpu --baseline-mean 12.5 --baseline-sd 25 --n-per-variant 20000 --power 0.9

# The same questions for the always-valid sequential test
ablift power --method sequential --baseline-rate 0.04 --n-per-variant 50000 --relative-mde 0.1
```

- `--method` is required (no default). `--method fixed` is the classic single-look two-sided z-test, so results match standard power calculators and statsmodels' `NormalIndPower`. `--method sequential` uses the mSPRT, which needs a larger effect for the same power but lets you look as often as you like.
- `--n-variants` applies the same Bonferroni correction as `analyze` and `duration`.
- For unequal splits, pass `--n-total` together with `--expected-split` (comma-separated positive weights in variant order, control first; `--n-per-variant` and `--n-total` are mutually exclusive). Power and MDE are evaluated for the worst-case treatment arm so every treatment meets the reported guarantee.
- The formulas are shared with `duration`: feeding `duration`'s `n_per_variant` (for equal splits) or `n_total` with the same `--expected-split` back into `power` returns the planned power and MDE.
- Output (`ablift schema power-output`) reports `solved_for` (`power` or `relative_mde`), `power`, `relative_mde`, `absolute_mde`, `n_per_variant` (control arm size), `n_by_variant`, `n_total`, and the `assumptions` used.
