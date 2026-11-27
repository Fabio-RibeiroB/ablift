# Agent Notes

`ablift` is a CLI for Bayesian and sequential A/B/n experiment analysis, designed to be called as a tool by AI agents.

## Setup
```bash
uv sync   # default groups include test + lint
```

## Commands
```bash
uv run pytest             # tests (also runs on push via pre-commit)
uv run ruff check .       # lint
uv run ruff format --check .
uv run ty check           # type check
uv run ablift doctor --strict # self-check
make demo                 # run example demo
```

## Rules
- Use `uv` for all Python commands.
- CLI uses Click; entry point is `ablift/cli.py`.
- Read `[tool.ablift]` from `pyproject.toml` for default config.

## Releasing
1. Bump `version` in `pyproject.toml`, run `uv lock`, and move `CHANGELOG.md` entries under a new `## [X.Y.Z] - date` heading.
2. Merge to `main`, then push a matching tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `.github/workflows/release.yml` checks the tag matches the version, tests, builds, smoke-tests the wheel, publishes to PyPI (trusted publishing, environment `pypi`) and creates the GitHub release from the changelog section.
4. Pre-releases: use a PEP 440 version such as `0.5.1a1` or `0.6.0rc1` and tag `v0.5.1a1`. The workflow marks the GitHub release as a pre-release; `pip install ablift` ignores it unless `--pre` is passed.
