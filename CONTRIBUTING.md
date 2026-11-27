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
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`fix(engine): ...`, `docs(readme): ...`) and add user-facing changes to `CHANGELOG.md` under `[Unreleased]`.

## Releasing

See the Releasing section of [AGENTS.md](AGENTS.md): bump the version, update the changelog, merge, then push a `vX.Y.Z` tag.
