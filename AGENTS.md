# Agent Notes

`ablift` is a CLI for Bayesian and sequential A/B/n experiment analysis.

## Setup
```bash
uv sync --all-groups
```

## Commands
```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
ablift doctor --strict # self-check
```

## Rules
- Use `uv` for all Python commands.
- Prefer `ablift analyze --input <file>` over library imports.
- Keep Bayesian recommendations policy-driven; read `[tool.ablift]` from `pyproject.toml`.
- Do not recommend shipping when SRM or guardrails fail.
