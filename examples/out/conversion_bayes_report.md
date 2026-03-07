# A/B Decision Report: homepage_cta

## Summary
- Method: `bayesian`
- Control: `control`
- Guardrails passed: `True`
- Recommendation: `ship_v2`
- Rationale: v2 passes Bayesian thresholds (P(win)=0.988, expected_loss=0.000005).
- Decision confidence: 0.9882
- Next best action: Roll out gradually and monitor guardrails.
- Risk flags: `none`

## Data Quality
- SRM passed: `True`
- SRM p-value: 0.996166
- SRM reason: No strong SRM evidence.

## Variant Comparisons
- Treatment: `v1` vs Control: `control`
  - Metric: `conversion_rate`
  - Control CR: 4.000%
  - Treatment CR: 4.160%
  - Absolute lift: 0.160%
  - Relative lift: 4.000%
  - Interval: [-0.020737, 0.105364]
  - P(treatment > control): 0.8972
  - Expected loss: 0.000061
- Treatment: `v2` vs Control: `control`
  - Metric: `conversion_rate`
  - Control CR: 4.000%
  - Treatment CR: 4.280%
  - Absolute lift: 0.280%
  - Relative lift: 7.000%
  - Interval: [0.008740, 0.135480]
  - P(treatment > control): 0.9882
  - Expected loss: 0.000005

## Guardrails
- `p95_latency_ms`: pass=`True`
  - direction: `decrease`
  - control: 420
  - treatment: 430
  - relative change: 2.381%
  - allowed: 5.000%
  - reason: Within allowed increase
