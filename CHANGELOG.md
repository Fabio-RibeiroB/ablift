# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Very large `--samples` crashed with a Python `MemoryError` traceback (200M samples needs 1.5 GB per arm). Bayesian runs now reject `samples × variants` above 20,000,000 with a message giving the maximum, and any out-of-memory condition becomes a clean `Error:` with exit code 1.
- `--bootstrap-samples` had no upper bound: 2,000,000 resamples of 40,000 rows ran for minutes with no output. Requests above 1,000,000,000 resampled rows are now a usage error (exit 2) that states the maximum for the data.

### Changed
- `duration --method bayesian` is about 25x faster: simulations are vectorised, and the earliest day reaching the assurance target is found by doubling then bisecting instead of scanning every day. A 1% MDE over 365 days went from 37 s to 1.5 s; results agree with the previous scan within simulation noise.

### Documentation
- README rewritten as a front page with a demo image rendered from a real run; reference material moved to `docs/` (inputs, methods, duration planning, agents).
- `tests/test_docs.py`, also run as a pre-commit hook, fails when a CLI command or option is undocumented, a documented example stops running, a link breaks, the install line does not match the release state, or the demo image is stale (`make docs`).
- Added `CONTRIBUTING.md`, issue and pull request templates, Dependabot for GitHub Actions and `CITATION.cff`.

### Fixed
- Fixed-horizon conversion intervals used the Wald formula, which collapses to `[0, 0]` at 0% or 100% conversion (claiming certainty from e.g. 0/10 vs 0/10) and undercovers for rare events. They now use Newcombe's hybrid score interval.
- Fixed-horizon conversion p-values used the normal approximation even for sparse tables, so 0/50 vs 4/50 was called significant (p ≈ 0.04). When any expected cell count is below 5, Fisher's exact test is used instead (p ≈ 0.12), and the `sparse_conversions` risk flag is raised.

## [0.5.0] - Unreleased

First published as pre-release `0.5.0a1` for early testing (`pip install --pre ablift`).

### Added
- MIT `LICENSE` and license metadata in the package.
- Release workflow that builds, smoke-tests and publishes to PyPI on a `v*` tag (trusted publishing).

### Changed
- Sequential testing now uses mSPRT with always-valid confidence intervals (replacing O'Brien-Fleming).
- Bonferroni correction across treatments in sequential mode.
- Sequential comparisons report an always-valid p-value, and `decision_confidence` reports `1 - p_value` on the same 0–1 scale across ship, rollback, and continue decisions.
- `ci_low`/`ci_high` are on the absolute-lift scale for both methods, with Bayesian `relative_ci_low`/`relative_ci_high`.
- Duration planning methods are now `sequential` (mSPRT sample size, default), `fixed` and `bayesian`, with Bonferroni correction.
- `duration` no longer prompts interactively.
- Duration CSV columns are validated, and `duration` rejects settings the chosen method does not use (e.g. `--alpha`/`--power` with `--method bayesian`) instead of silently ignoring them (including input-file columns unless `--method` overrides the file's `method`). The bundled `examples/duration_inputs.xlsx` uses Bayesian settings.
- CSV/XLSX columns are inferred by position, and `input_interpretation.column_roles` reports the mapping.
- JSON output includes `schema_version`.

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
- A control-flag column placed before the value columns gave a confusing numeric error, and between visitors and revenue columns was silently read as conversions; it is now rejected with a message saying the flag must be the last column.
- A numeric 0/1 metric column (e.g. a tiny `revenue_sum_squares`) could be mistaken for the control flag; numeric control flags now require a header mentioning "control".
- Row-level outcomes written as `1.0`/`0.0` were counted as failures; numeric tokens equal to 1 now count as successes.
- JSON conversion-rate input missing `conversions` silently defaulted to 0; it is now rejected (ARPU input may still omit it).
- Fractional counts (e.g. `n_variants` = 2.5 or `visitors` = 1000.5) were silently truncated; they are now rejected.
- Sequential `p_value` is now Bonferroni-adjusted, so `p_value <= alpha` exactly when `significant`.
- CSVs saved by Excel with a UTF-8 byte-order mark had a corrupted first header, and blank trailing rows in CSV/XLSX broke parsing.

### Removed
- `--mapping`, `--max-looks` and `--prompt-text` options.
- `analyze-file` and `example-duration-prompt` commands.
- `look_index`, `max_looks` and `information_fraction` inputs.
- `arpu_approximate`.
