# Agent Notes

`ablift` is a CLI for Bayesian and sequential A/B/n experiment analysis.

## Setup
```bash
uv sync --all-groups
```

## Commands
```bash
uv run pytest          # tests (also runs on push via pre-commit)
ablift doctor --strict # self-check
```

## Rules
- Use `uv` for all Python commands.
- Read `[tool.ablift]` from `pyproject.toml` for default config.
