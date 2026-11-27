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
make docs                 # re-render docs/demo.svg from real CLI output
```

## Rules
- Use `uv` for all Python commands.
- All code must pass `make lint` (ruff check, ruff format, ty) and `make test` before it is committed. The git hooks enforce this (`uv run pre-commit install --hook-type pre-commit --hook-type pre-push`, which also covers worktrees), CI checks every PR, and `main` only accepts PRs whose checks pass. Never skip hooks with `--no-verify`.
- CLI uses Click; entry point is `ablift/cli.py`.
- Read `[tool.ablift]` from `pyproject.toml` for default config.

<important if="you change CLI commands, options, output, examples, or the package version">
Update README.md / docs/ in the same change. `tests/test_docs.py` (also a pre-commit hook) fails on undocumented options, documented commands that no longer run, broken links, a stale install line, or a stale `docs/demo.svg` (fix with `make docs`).
</important>

## Releasing
1. Bump `version` in `pyproject.toml`, run `uv lock`, and move `CHANGELOG.md` entries under a new `## [X.Y.Z] - date` heading.
2. Merge to `main`, then push a matching tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `.github/workflows/release.yml` checks the tag matches the version, tests, builds, smoke-tests the wheel, publishes to PyPI (trusted publishing, environment `pypi`) and creates the GitHub release from the changelog section.
4. Pre-releases: use a PEP 440 version such as `0.5.0a1` or `1.0.0rc1` and tag `v0.5.0a1`. The workflow marks the GitHub release as a pre-release; `pip install ablift` ignores it unless `--pre` is passed. Keep pre-releases sparing during `0.x`.
