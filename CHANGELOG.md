# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.5.1] - 2026-11-22

### Added
- MIT `LICENSE` and license metadata in the package.
- Release workflow that builds, smoke-tests and publishes to PyPI on a `v*` tag (trusted publishing).

### Changed
- Sequential `continue_collecting_data` now reports `decision_confidence = 1 - p_value` of the leading comparison, on the same scale as ship/rollback (previously the raw p-value, where lower meant stronger).
- `duration` rejects settings the chosen method does not use (e.g. `--alpha`/`--power` with `--method bayesian`) instead of silently ignoring them. The bundled `examples/duration_inputs.xlsx` now uses Bayesian settings.

### Fixed
- A control-flag column placed before the value columns gave a confusing numeric error, and between visitors and revenue columns was silently read as conversions. It is now rejected with a message saying the flag must be the last column.

## [0.5.0] - 2026-11-14

### Fixed
- Aggregated ARPU CSV/XLSX inputs (`name`, `visitors`, `conversions`, `revenue_sum`, `revenue_sum_squares`) were silently analysed as conversion rate.
- Aggregated revenue-only input failed validation.
- Row-level ARPU crashed during analysis.
- `make demo` used the removed `--mapping` flag.
- Malformed input produced Python tracebacks instead of clean errors (including JSON arrays, unknown XLSX sheets and non-numeric cells, which now name the offending column).
- Sequential tests declared significance on 1-2 users when an arm had 0% or 100% conversion; the standard error now uses the Agresti-Caffo adjustment, and ARPU arms with fewer than 2 users report no evidence.
- Sequential results could emit `-Infinity` (invalid JSON) for intervals that cannot be estimated; these are now `null`.
- SRM p-value used a Wilson-Hilferty approximation (0.95 for a perfect split); it is now the exact chi-square tail.
- Aggregated tables with a repeated variant name were misread as row-level data; they are now rejected with a layout error.
- A control column with no `true` values was misread as a metric column; columns whose header mentions "control" are now always treated as the control flag.
- A numeric 0/1 metric column (e.g. a tiny `revenue_sum_squares`) could be mistaken for the control flag; numeric control flags now require a header mentioning "control".
- Row-level outcomes written as `1.0`/`0.0` were counted as failures; numeric tokens equal to 1 now count as successes.
- JSON conversion-rate input missing `conversions` silently defaulted to 0; it is now rejected (ARPU input may still omit it).
- Fractional counts (e.g. `n_variants` = 2.5 or `visitors` = 1000.5) were silently truncated; they are now rejected.
- Sequential `p_value` is now Bonferroni-adjusted, so `p_value <= alpha` exactly when `significant`.
- CSVs saved by Excel with a UTF-8 byte-order mark had a corrupted first header, and blank trailing rows in CSV/XLSX broke parsing.

### Changed
- Sequential testing now uses mSPRT with always-valid confidence intervals (replacing O'Brien-Fleming).
- Bonferroni correction across treatments in sequential mode.
- Sequential comparisons report an always-valid p-value, and `decision_confidence` derives from it.
- `ci_low`/`ci_high` are on the absolute-lift scale for both methods, with Bayesian `relative_ci_low`/`relative_ci_high`.
- Duration planning methods are now `sequential` (mSPRT sample size, default), `fixed` and `bayesian`, with Bonferroni correction.
- `duration` no longer prompts interactively.
- Duration CSV columns are validated.
- CSV/XLSX columns are inferred by position, and `input_interpretation.column_roles` reports the mapping.
- JSON output includes `schema_version`.

### Removed
- `--mapping`, `--max-looks` and `--prompt-text` options.
- `analyze-file` and `example-duration-prompt` commands.
- `look_index`, `max_looks` and `information_fraction` inputs.
- `arpu_approximate`.
