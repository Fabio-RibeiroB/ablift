# bayestest-cli Skill

Use this skill when a user asks for Bayesian or sequential A/B/n analysis, recommendations, or experiment reporting.

## Purpose

Run `bayestest` CLI to produce statistically grounded decisions for experiments using:
- Bayesian (`method=bayesian`)
- Frequentist sequential (`method=frequentist_sequential`)
- Primary metric: `conversion_rate` or `arpu`
- Guardrails and SRM diagnostics

## When to use

Use this skill if the user asks to:
- Decide a winner for an A/B or A/B/n test
- Evaluate probability to win (Bayesian)
- Evaluate sequential significance across looks
- Include business guardrails in decisions
- Generate JSON and markdown reports for agent/user review

## Preconditions

Run from repo root: `/home/fabio/bayestest`.

Environment setup:
```bash
source $HOME/.local/bin/env
uv venv .venv
uv pip install -e .
```

Pre-run checklist for agents:
- `bayestest doctor --strict` passes
- Python is `>=3.9`
- Input has exactly one control variant
- Variant rows include `name`, `visitors`, `conversions`
- If `primary_metric=arpu`, include `revenue_sum` and `revenue_sum_squares`
- For CSV/XLSX commands, mapping JSON is present and column names are verified

## Input modes

1. JSON payload mode
```bash
bayestest analyze --input input.json --output output.json --report report.md
```

2. CSV/XLSX mode with mapping
```bash
bayestest analyze-file --input data.csv --mapping mapping.json --output output.json --report report.md
bayestest analyze-file --input data.xlsx --mapping mapping.json --sheet Sheet1 --output output.json --report report.md
```

3. Pasted-text quick analysis (conversation input)
```bash
bayestest analyze-text --text \"Variant A: 100 conversions out of 2000 visitors\nVariant B: 125 conversions out of 2000 visitors\"
```

4. Duration planning (conversation, CLI args, CSV, Excel)
```bash
bayestest duration --prompt-text \"Traffic: 50000 visitors/day\nBaseline: 4%\nMDE: 5%\nLooks: 10\"
bayestest duration --method frequentist --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3 --max-looks 10
bayestest duration --method bayesian --baseline-rate 0.04 --relative-mde 0.05 --daily-traffic 50000 --n-variants 3 --max-days 60
bayestest duration --input duration_inputs.csv --mapping duration_mapping.json
bayestest duration --input duration_inputs.xlsx --mapping duration_mapping.json --sheet Sheet1
```

Generate templates:
```bash
bayestest example-input
bayestest example-mapping
bayestest example-duration-mapping
bayestest example-duration-prompt
```

## Required data contract

- Exactly one control variant (`is_control=true`)
- For all variants: `name`, `visitors`, `conversions`
- If `primary_metric=arpu`: also `revenue_sum`, `revenue_sum_squares`

## Decision interpretation

Read `recommendation` in output JSON:
- `action`: final recommendation (`ship_*`, `continue_collecting_data`, `do_not_ship`, `investigate_data_quality`, `stop_and_rollback`)
- `decision_confidence`: confidence score for action
- `risk_flags`: blockers/warnings (e.g., `srm_detected`, `guardrail_failure`)
- `next_best_action`: concrete follow-up for agent/user

Always inspect:
- `srm.passed` before trusting winner decisions
- `guardrails_passed` before rollout
- per-treatment `comparisons` for lift, intervals, p-values or P(win)

## Agent response format

When returning results to user, include:
1. Recommended action and confidence
2. Top treatment(s) with lift and uncertainty
3. Guardrail and SRM status
4. Clear next action (ship, continue, rollback, or investigate)
5. Input mode used (`conversation`, `csv`, or `excel`) so users can reproduce the run

## Safety rules

- Do not recommend shipping when SRM fails.
- Do not recommend shipping when guardrails fail.
- If sequential method is not significant at current look, recommend collecting more data.
- If data contract is invalid, report exact missing/invalid fields and request corrected input.
