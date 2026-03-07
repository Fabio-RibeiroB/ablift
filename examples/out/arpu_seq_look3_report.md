# A/B Decision Report: checkout_flow

## Summary
- Method: `frequentist_sequential`
- Control: `control`
- Guardrails passed: `True`
- Recommendation: `continue_collecting_data`
- Rationale: Sequential significance boundary not reached yet.
- Decision confidence: 0.5000
- Next best action: Wait for more information fraction or sample size.
- Risk flags: `none`

## Data Quality
- SRM passed: `True`
- SRM p-value: 0.950498
- SRM reason: No strong SRM evidence.

## Variant Comparisons
- Treatment: `v1` vs Control: `control`
  - Metric: `arpu`
  - Control ARPU: 1.750000
  - Treatment ARPU: 1.866667
  - Absolute lift: 0.116667
  - Relative lift: 6.667%
  - Interval: [-0.028950, 0.262283]
  - p-value: 0.004144
  - Alpha spent: 0.000346
  - Significant now: `False`

## Guardrails
No guardrails provided.
