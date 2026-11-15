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
