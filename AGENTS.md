# Agent Notes

`ablift` is a statistical analysis and planning CLI for Bayesian, sequential (mSPRT), and fixed-horizon A/B/n experiments, designed to be called as a tool by humans, pipelines, and AI agents.

## Scope & Product Philosophy
- **Analysis tool first:** `ablift analyze` returns statistical estimates (`comparisons`, `bayesian_arms`), data-quality checks (`srm`, `guardrails`, top-level `risk_flags`), and `secondary_metrics`.
- **Opt-in recommendations:** Automated ship/wait/rollback `recommendation` output is off by default across all methods (`bayesian`, `sequential`, `fixed`) and only produced when callers opt in via `--recommend` or `decision_policy.enabled = true`.

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
- All code must pass `make lint` (ruff check, ruff format, ty) and `make test` before it is committed. The git hooks enforce this. Git shares them across all worktrees, so install them once from the main checkout (`uv run pre-commit install --hook-type pre-commit --hook-type pre-push`), never from a worktree. CI checks every PR, and `main` only accepts PRs whose checks pass. Never skip hooks with `--no-verify`.
- Several agents work on this repo in parallel. Follow "Working in parallel" in `CONTRIBUTING.md`: claim the issue and open a draft PR first, work in a worktree and leave the main checkout clean on `main`, and merge only your own PRs.
- CLI uses Click; entry point is `ablift/cli.py`.
- Read `[tool.ablift]` from `pyproject.toml` for default config.
- Never edit `CHANGELOG.md` in a feature PR. Add a fragment `changes/<issue>.<type>.md` (`added`, `changed`, `fixed`, `removed`, `docs`); CI requires one when `ablift/` changes. Check locally with `make changelog-check` (also a pre-push hook). Internal-only changes: `no-changelog` PR label, or `SKIP_CHANGELOG=1` locally. See `changes/README.md`.

<important if="you change CLI commands, options, output, examples, or the package version">
Update README.md / docs/ in the same change. `tests/test_docs.py` (also a pre-commit hook) fails on undocumented options, documented commands that no longer run, broken links, a stale install line, or a stale `docs/demo.svg` (fix with `make docs`).
</important>

## Releasing
1. Bump `version` in `pyproject.toml`, run `uv lock`, then `uv run towncrier build --version X.Y.Z --yes` to move `changes/` fragments into `CHANGELOG.md`.
2. Merge to `main`, then push a matching tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `.github/workflows/release.yml` checks the tag matches the version, tests, builds, smoke-tests the wheel, publishes to PyPI (trusted publishing, environment `pypi`) and creates the GitHub release from the changelog section.
4. Pre-releases: use a PEP 440 version such as `0.5.0a1` or `1.0.0rc1` and tag `v0.5.0a1`. The workflow marks the GitHub release as a pre-release; `pip install ablift` ignores it unless `--pre` is passed. Keep pre-releases sparing during `0.x`.
