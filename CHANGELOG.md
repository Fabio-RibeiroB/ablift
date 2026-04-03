# Changelog

## 0.4.0 - 2026-04-03

- Renamed the project, package, CLI, and GitHub repository from `bayestest` to `ablift`.
- Unified `analyze` across JSON, CSV, and XLSX inputs with clearer help examples and input diagnostics.
- Made Bayesian recommendations explicit and policy-driven, with reusable project defaults under `[tool.ablift]`.
- Raised the supported Python baseline to 3.11 and adopted a `pyproject.toml`-first configuration flow.
- Added Ruff, Ty, and pre-commit hooks, including `pytest` on `pre-push`.

## 0.3.0 - 2026-03-07

- Added ARPU modeling for Bayesian and sequential workflows.
- Added SRM check and agent-oriented recommendation fields.
- Added multi-variant (`A/B/n`) coverage with winner recommendation.
- Added CSV/XLSX connectors with mapping-based ingestion.
- Added demo assets and run script under `examples/` and `scripts/run_demo.sh`.
- Added test coverage for ARPU, SRM, connectors, and multi-variant behavior.
- Added uv-first docs and packaging metadata in `pyproject.toml`.
