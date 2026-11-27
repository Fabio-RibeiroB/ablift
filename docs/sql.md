# Reference SQL

A portable ANSI SQL query that turns raw exposure and event tables into `ablift`'s aggregated CSV input for `arpu` and `conversion_rate`, plus an optional CUPED variant for JSON input.

## Expected tables

The queries below assume two warehouse tables:

- `exposures(user_id, variant, exposed_at)` — raw assignment or exposure logs for one experiment. A user may appear more than once (duplicate exposure rows) or in more than one variant (mixed-variant contamination). Filter to a single experiment in `clean_units` if your table stores multiple experiments.
- `orders(user_id, revenue, ordered_at)` — raw outcome events. A user may have zero, one, or many orders.

## Aggregated CSV query

This query outputs `ablift`'s 5-column positional aggregated CSV layout (`name, visitors, conversions, revenue_sum, revenue_sum_squares`; see [Aggregated input](inputs.md#aggregated-input-one-row-per-variant)). The same CSV works for both `arpu` (default) and `conversion_rate` (with `--primary-metric conversion_rate`).

```sql
WITH clean_units AS (
    -- One row per randomization unit (user_id).
    -- GROUP BY collapses duplicate exposure rows in the same variant.
    -- HAVING COUNT(DISTINCT variant) = 1 excludes units exposed to multiple variants.
    SELECT
        user_id,
        MIN(variant) AS variant,
        MIN(exposed_at) AS first_exposed_at
    FROM exposures
    GROUP BY user_id
    HAVING COUNT(DISTINCT variant) = 1
),
unit_metrics AS (
    -- Roll up events to one row per unit BEFORE computing sums of squares.
    -- LEFT JOIN + COALESCE keeps exposed users with zero orders in the denominator
    -- with 0 conversions and 0.0 revenue.
    SELECT
        u.user_id,
        u.variant,
        CASE WHEN COUNT(o.user_id) > 0 THEN 1 ELSE 0 END AS converted,
        COALESCE(SUM(o.revenue), 0.0) AS unit_revenue
    FROM clean_units u
    LEFT JOIN orders o
        ON o.user_id = u.user_id
        AND o.ordered_at >= u.first_exposed_at
    GROUP BY u.user_id, u.variant
),
variant_aggregates AS (
    -- Aggregate per-unit metrics into ablift's positional CSV columns.
    SELECT
        variant AS name,
        COUNT(*) AS visitors,
        SUM(converted) AS conversions,
        SUM(unit_revenue) AS revenue_sum,
        SUM(unit_revenue * unit_revenue) AS revenue_sum_squares
    FROM unit_metrics
    GROUP BY variant
)
SELECT
    name,
    visitors,
    conversions,
    revenue_sum,
    revenue_sum_squares
FROM variant_aggregates
ORDER BY
    CASE WHEN name = 'control' THEN 0 ELSE 1 END,
    name;
```

Without an `is_control` column, `ablift` treats the first row as control. The `ORDER BY` places `'control'` first; if your control arm has another name, either adjust the `ORDER BY` or append an `is_control` boolean column at the end.

## Piping CSV into `ablift analyze`

Pipe the query's CSV output (including the header row) straight into `ablift analyze --input - --input-format csv`:

<!-- docs-test: skip -->
```bash
# ARPU (default for the 5-column layout)
psql "$DATABASE_URL" --csv -f query.sql \
  | ablift analyze --input - --input-format csv --method sequential

# Conversion rate from the same 5-column query
psql "$DATABASE_URL" --csv -f query.sql \
  | ablift analyze --input - --input-format csv --primary-metric conversion_rate --method sequential

# SQLite (with CSV headers)
sqlite3 -header -csv experiment.db < query.sql \
  | ablift analyze --input - --input-format csv

# DuckDB
duckdb -csv experiment.duckdb < query.sql \
  | ablift analyze --input - --input-format csv
```

## Optional CUPED variant

For [CUPED variance reduction](inputs.md#json-input) on `arpu` or `conversion_rate`, add a strictly pre-experiment covariate `X` (for example, each user's revenue before `first_exposed_at`) and compute three additional per-variant sums: `covariate_sum`, `covariate_sum_squares`, and `outcome_covariate_sum`.

```sql
WITH clean_units AS (
    SELECT
        user_id,
        MIN(variant) AS variant,
        MIN(exposed_at) AS first_exposed_at
    FROM exposures
    GROUP BY user_id
    HAVING COUNT(DISTINCT variant) = 1
),
unit_metrics AS (
    -- Split orders strictly at first_exposed_at using conditional aggregation
    -- on a single LEFT JOIN to avoid join fan-out:
    --   ordered_at <  first_exposed_at -> pre-experiment covariate X
    --   ordered_at >= first_exposed_at -> post-exposure outcome Y
    SELECT
        u.user_id,
        u.variant,
        CASE
            WHEN SUM(CASE WHEN o.ordered_at >= u.first_exposed_at THEN 1 ELSE 0 END) > 0
            THEN 1 ELSE 0
        END AS converted,
        COALESCE(
            SUM(CASE WHEN o.ordered_at >= u.first_exposed_at THEN o.revenue ELSE 0.0 END),
            0.0
        ) AS unit_revenue,
        COALESCE(
            SUM(CASE WHEN o.ordered_at < u.first_exposed_at THEN o.revenue ELSE 0.0 END),
            0.0
        ) AS covariate
    FROM clean_units u
    LEFT JOIN orders o
        ON o.user_id = u.user_id
    GROUP BY u.user_id, u.variant
),
variant_aggregates AS (
    SELECT
        variant AS name,
        COUNT(*) AS visitors,
        SUM(converted) AS conversions,
        SUM(unit_revenue) AS revenue_sum,
        SUM(unit_revenue * unit_revenue) AS revenue_sum_squares,
        SUM(covariate) AS covariate_sum,
        SUM(covariate * covariate) AS covariate_sum_squares,
        -- For primary_metric = "arpu", Y is unit_revenue.
        -- For primary_metric = "conversion_rate", use SUM(converted * covariate) instead.
        SUM(unit_revenue * covariate) AS outcome_covariate_sum
    FROM unit_metrics
    GROUP BY variant
)
SELECT
    name,
    visitors,
    conversions,
    revenue_sum,
    revenue_sum_squares,
    covariate_sum,
    covariate_sum_squares,
    outcome_covariate_sum
FROM variant_aggregates
ORDER BY
    CASE WHEN name = 'control' THEN 0 ELSE 1 END,
    name;
```

Aggregated CUPED requires **JSON** input (for row-level CSV/XLSX input, pass `--covariate-column` instead; see [Inputs](inputs.md#json-input)). Map each row returned by the query to an entry in `variants`:

```json
{
  "experiment_name": "checkout_v2",
  "method": "sequential",
  "primary_metric": "arpu",
  "variants": [
    {
      "name": "control",
      "is_control": true,
      "visitors": 10000,
      "conversions": 550,
      "revenue_sum": 22000.0,
      "revenue_sum_squares": 150000.0,
      "covariate_sum": 19500.0,
      "covariate_sum_squares": 132000.0,
      "outcome_covariate_sum": 98000.0
    },
    {
      "name": "treatment",
      "is_control": false,
      "visitors": 10000,
      "conversions": 590,
      "revenue_sum": 23600.0,
      "revenue_sum_squares": 164000.0,
      "covariate_sum": 19620.0,
      "covariate_sum_squares": 133100.0,
      "outcome_covariate_sum": 103400.0
    }
  ]
}
```

## Common traps

- **`INNER JOIN` drops non-purchasers.** Joining `exposures` to `orders` with `INNER JOIN` (or filtering on `orders` columns in `WHERE` instead of `LEFT JOIN ... ON`) discards every exposed user with zero orders. That turns ARPU into average revenue per purchaser (closer to AOV), breaks visitor counts, and corrupts the SRM check whenever conversion rates differ between arms.
- **Event-level sums of squares are wrong.** Squaring individual order amounts before rolling up to one row per unit computes $\sum_j r_j^2$ instead of $(\sum_j r_j)^2$. For a user with multiple orders $r_1, r_2$, dropping the cross-term $2 r_1 r_2$ underestimates variance and produces intervals that are too narrow. Always aggregate per unit in `unit_metrics` before squaring.
- **Duplicate or mixed-variant exposures.** Raw exposure logs often record multiple rows for the same user, which inflates `visitors` if not deduplicated. Users exposed to more than one variant break randomization independence and should be excluded with `HAVING COUNT(DISTINCT variant) = 1` (investigate assignment logging if the excluded share is large).
