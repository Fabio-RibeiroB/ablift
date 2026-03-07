# A/B Decision Report: pricing_page

## Summary
- Method: `bayesian`
- Control: `control`
- Guardrails passed: `True`
- Recommendation: `ship_v1`
- Rationale: v1 passes Bayesian thresholds (P(win)=0.999, expected_loss=0.000008).
- Decision confidence: 0.9994
- Next best action: Roll out gradually and monitor guardrails.
- Risk flags: `none`

## Data Quality
- SRM passed: `True`
- SRM p-value: 0.996166
- SRM reason: No strong SRM evidence.

## Variant Comparisons
- Treatment: `v1` vs Control: `control`
  - Metric: `arpu`
  - Control ARPU: 2.200000
  - Treatment ARPU: 2.350000
  - Absolute lift: 0.150000
  - Relative lift: 6.818%
  - Interval: [0.026049, 0.111946]
  - P(treatment > control): 0.9994
  - Expected loss: 0.000008
- Treatment: `v2` vs Control: `control`
  - Metric: `arpu`
  - Control ARPU: 2.200000
  - Treatment ARPU: 2.310000
  - Absolute lift: 0.110000
  - Relative lift: 5.000%
  - Interval: [0.008732, 0.092605]
  - P(treatment > control): 0.9913
  - Expected loss: 0.000127

## Guardrails
No guardrails provided.
