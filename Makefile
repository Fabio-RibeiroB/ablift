.PHONY: demo test

demo:
	bash scripts/run_demo.sh

test:
	uv run pytest
