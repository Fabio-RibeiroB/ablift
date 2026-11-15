.PHONY: demo test lint

demo:
	bash scripts/run_demo.sh

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run ty check
