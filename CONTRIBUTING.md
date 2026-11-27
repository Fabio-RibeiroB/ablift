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

## Pull requests

- Keep pull requests small and focused, and rebase onto `main` before asking for review. `ablift/cli.py`, `ablift/engine.py` and `ablift/schemas.py` change often, so long-lived branches conflict quickly.
- A change that adds, removes or renames an `analyze`, `duration` or `power` output field updates `ablift/schemas.py` and `docs/agents.md` in the same pull request.
- Mark breaking changes (CLI flags, output fields, defaults) with a `!` in the title (`feat(cli)!: ...`) and a `changed` or `removed` changelog fragment.
- CI runs Python 3.11 to 3.13. Avoid 3.12+-only APIs, and don't put test assertions exactly on a floating-point boundary, because results can differ slightly between versions.

## Releasing

See the Releasing section of [AGENTS.md](AGENTS.md): bump the version, update the changelog, merge, then push a `vX.Y.Z` tag.
