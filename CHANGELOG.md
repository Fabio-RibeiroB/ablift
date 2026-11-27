# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries are assembled from `changes/` fragments with towncrier; see [changes/README.md](changes/README.md).

<!-- towncrier release notes start -->

## [0.5.0a3] - 2026-11-27

Third pre-release (`pip install --pre ablift`). **Breaking:** recommendations are now opt-in (`--recommend`), `duration` and `power` require `--method`, and method-specific settings are rejected or reported as `null` when the method does not use them. See Changed.

### Added
- `ablift duration` and `ablift power` now accept `--expected-split` (and an `expected_split` field or column in duration `--input` files) to plan sample sizes, run duration, power, and MDE under unequal traffic allocations (such as 90/10 rollouts) across all methods and both `conversion_rate` and `arpu` metrics. Duration and power outputs now report `n_by_variant` and `n_total` alongside `n_per_variant`, and `ablift power` adds `--n-total` so unequal-split plans round-trip cleanly. ([#36](https://github.com/Fabio-RibeiroB/ablift/issues/36))
- `ablift analyze` now reports which `pyproject.toml` it took defaults from in `analysis_settings.config_source` (an absolute path, or `null` when no `[tool.ablift]` table was applied), and a new `--no-config` flag ignores that file entirely so results depend only on the input and flags.
- `ablift analyze` output now has a top-level `risk_flags` list that is always present, even when no recommendation is requested. It reports data-quality and experiment risks such as `srm_detected`, `guardrail_failure`, `covariate_imbalance`, `heavy_tailed_revenue` and `sparse_conversions`.

### Changed
- `ablift analyze` now leaves `recommendation` as `null` by default for every method. Before, `sequential` and `fixed` always produced one. To get a recommendation, pass `--recommend` or set `decision_policy.enabled = true`. The `--enable-recommendation/--no-recommendation` flags are renamed to `--recommend/--no-recommend`. `analyze` now rejects `--sequential-tau` unless the method is `sequential`, and rejects Bayesian-only flags (`--samples`, `--prob-threshold`, `--max-expected-loss`) for `sequential` as well as `fixed`. `ablift example-input` no longer includes a `decision_policy`, and the Markdown report is now titled "A/B Analysis Report".
- `ablift analyze` now reports `analysis_settings.samples` and `analysis_settings.random_seed` as `null` when no posterior sampling happened (`sequential` and `fixed`; `random_seed` stays set when a row-level `--bootstrap-samples` CI used it). JSON input is now held to the same rules as the CLI flags: `samples` or `random_seed` with a `sequential`/`fixed` method, or `sequential_tau` with `bayesian`, is an error instead of being silently ignored. `[tool.ablift]` project defaults for these fields only apply to the method that uses them, so they never block `--method sequential` or `--method fixed`.
- `ablift duration` and `ablift power` no longer pick a method for you. Before, `duration` defaulted to `sequential` and `power` to `fixed`, while `analyze` defaults to `bayesian`, so running them in a row without `--method` silently switched frameworks. Now pass `--method sequential` if you'll check results as they come in, or `--method fixed` for a single planned look (`duration` also takes `bayesian`); for `duration`, a `method` field in the `--input` file counts too. Leaving it out is a usage error (exit code 2). `analyze` keeps its `bayesian` default.

### Documentation
- Added `docs/sql.md` with a verified ANSI SQL reference query (and optional CUPED variant) for aggregating raw exposure and order tables into `ablift`'s aggregated CSV and JSON inputs, along with common pitfalls to avoid. ([#42](https://github.com/Fabio-RibeiroB/ablift/issues/42))

## [0.5.0a2] - 2026-11-27

Second pre-release (`pip install --pre ablift`).

### Added
- `ablift power`: power analysis for a fixed sample size. Give `--relative-mde` to get the power, or leave it out to get the minimum detectable effect at `--power` (default 0.8). It supports conversion rate and ARPU, the `fixed` (default) and `sequential` methods, and Bonferroni across `--n-variants`. Fixed-method power matches statsmodels `NormalIndPower`, and `duration`'s sample size round-trips to the planned power and MDE. The output contract is `ablift schema power-output`.

### Fixed
- `ablift doctor` now enforces minimum dependency versions (`click>=8.0`, `numpy>=1.22`, `openpyxl>=3.1.0`) instead of passing whenever a package is merely installed.
- Malformed `--expected-split` CLI flag values now exit with code `2` (usage error) rather than `1`.
- `ablift analyze` now rejects unknown top-level JSON fields (matching `ablift schema analyze-input` instead of silently ignoring typos), and no longer reports `decision_policy.enabled: false` from `pyproject.toml` on `--method sequential` or `--method fixed` runs.
- Sparse fixed-horizon conversion comparisons using Fisher's exact test now report `test_statistic: null` (and label the test as Fisher's exact in Markdown reports) instead of emitting the unused normal z-statistic.
- `scripts/render_demo.py` now pins both console width and height so `docs/demo.svg` renders deterministically even under `TERM=dumb`.
- Very large `--samples` crashed with a Python `MemoryError` traceback (200M samples needs 1.5 GB per arm). Bayesian runs now reject `samples × variants` above 20,000,000 with a message giving the maximum, and any out-of-memory condition becomes a clean `Error:` with exit code 1.
- `--bootstrap-samples` had no upper bound: 2,000,000 resamples of 40,000 rows ran for minutes with no output. Requests above 1,000,000,000 resampled rows are now a usage error (exit 2) that states the maximum for the data.
- Fixed-horizon conversion intervals used the Wald formula, which collapses to `[0, 0]` at 0% or 100% conversion (claiming certainty from e.g. 0/10 vs 0/10) and undercovers for rare events. They now use Newcombe's hybrid score interval.
- Fixed-horizon conversion p-values used the normal approximation even for sparse tables, so 0/50 vs 4/50 was called significant (p ≈ 0.04). When any expected cell count is below 5, Fisher's exact test is used instead (p ≈ 0.12), and the `sparse_conversions` risk flag is raised.

### Changed
- `duration --method bayesian` is about 25x faster: simulations are vectorised, and the earliest day reaching the assurance target is found by doubling then bisecting instead of scanning every day. A 1% MDE over 365 days went from 37 s to 1.5 s; results agree with the previous scan within simulation noise.

### Documentation
- README rewritten as a front page with a demo image rendered from a real run; reference material moved to `docs/` (inputs, methods, duration planning, agents).
- `tests/test_docs.py`, also run as a pre-commit hook, fails when a CLI command or option is undocumented, a documented example stops running, a link breaks, the install line does not match the release state, or the demo image is stale (`make docs`).
- Added `CONTRIBUTING.md`, issue and pull request templates, Dependabot for GitHub Actions and `CITATION.cff`.

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
