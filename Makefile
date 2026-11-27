.PHONY: demo docs test lint changelog-check

demo:
	bash scripts/run_demo.sh

docs:
	uv run python scripts/render_demo.py

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run ty check

changelog-check:
	bash scripts/check_changelog_fragment.sh origin/main
