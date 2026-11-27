# Duration planning

`ablift duration` estimates how many visitors, and how many days, a test needs before it can reach a decision.

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
- `bayesian`: simulation-based assurance, the probability of reaching the Bayesian decision thresholds within `--max-days`. Simulations are vectorised and the earliest qualifying day is found by doubling then bisecting, so even a 365-day horizon plans in about a second (Beta-Binomial for `conversion_rate`, Normal-Inverse-Gamma for `arpu`).

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
