# Contributing

Thanks for helping improve `ablift`. Bug reports with a minimal input file are the most useful contribution of all.

## Set up

```bash
uv sync                       # Python 3.11+, installs the test and lint groups
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Before opening a pull request

```bash
make lint      # ruff and ty
make test      # full test suite
make demo      # every bundled example, end to end
```

- **Statistics changes** need a test against an independent reference (scipy, statsmodels or a textbook value), not just the code's own helpers. Bug fixes start with a failing regression test.
- **CLI changes** must keep the docs current. `tests/test_docs.py` fails when:
  - an `ablift` command or option is not mentioned in `README.md` or `docs/`;
  - a documented `ablift ...` example no longer runs;
  - a link or heading anchor is broken, or a `docs/` page is not linked from the README;
  - the README install command does not match the release state (pre-release or stable);
  - `docs/demo.svg` no longer matches real CLI output (run `make docs`).
  The same check runs as a pre-commit hook whenever docs, the CLI or examples change.
- A documented command that is not meant to run (for example one with placeholder paths) goes in a block preceded by `<!-- docs-test: skip -->`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`fix(engine): ...`, `docs(readme): ...`). Describe user-facing changes in a changelog fragment, `changes/<issue>.<type>.md`, rather than editing `CHANGELOG.md`; see [changes/README.md](changes/README.md). CI requires a fragment when `ablift/` changes.

## Working in parallel

Several people and AI agents often work on `ablift` at the same time. These rules keep that from producing conflicts or broken checkouts.

**Before you start**
- Claim the work on GitHub, not in chat: assign yourself or comment on the issue, then open a draft PR early so others can see the branch and the files it touches. Check `gh pr list` before editing a file someone else may be changing.
- Work in a git worktree, one per branch. Keep the main checkout on a clean, up-to-date `main`. Never leave staged or uncommitted work there.
- Install the git hooks once, from the main checkout: `uv run pre-commit install --hook-type pre-commit --hook-type pre-push`. Git shares hooks across all worktrees, so installing them from a worktree, especially a temporary one, points them at that worktree's `.venv` and breaks them everywhere once it is deleted.

**While you work**
- Keep PRs small and short-lived. Squash-merge once checks pass; long-lived branches are the main source of conflicts.
- Treat these as hot files and keep your edits to them small and focused: `ablift/cli.py`, `ablift/engine.py`, `ablift/schemas.py`, `README.md`, `docs/demo.svg`, `pyproject.toml`. Rebase onto `main` right before merging. Whoever merges second resolves the conflicts and keeps both sides.
- Change the output contract in one place. A PR that adds, removes or renames an `analyze`, `duration` or `power` output field updates `ablift/schemas.py` and `docs/agents.md` in the same PR.
- Test on every supported Python version. Local runs usually use the newest Python, but CI also runs 3.11, so avoid 3.12+-only APIs. Floating-point results can differ slightly between versions (`statistics.NormalDist` by about 1e-5), so don't put test assertions exactly on a boundary.

**Before you merge**
- Breaking changes (CLI flags, output fields, defaults) get a `changed` or `removed` changelog fragment, a `!` in the PR title (`feat(cli)!: ...`), and a heads-up to anyone else with open work, because they break other branches' tests.
- PRs that change statistical behaviour need an independent review before merge: a second reviewer (person or a different model) checks the maths against an independent reference such as scipy, statsmodels or a closed form, and mutates the implementation to confirm the tests catch it.
- The PR's owner merges it. Don't merge someone else's PR, don't merge with checks pending, and don't bypass branch protection.

## Releasing

See the Releasing section of [AGENTS.md](AGENTS.md): bump the version, update the changelog, merge, then push a `vX.Y.Z` tag.
