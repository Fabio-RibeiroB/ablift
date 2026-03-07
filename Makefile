.PHONY: demo test

demo:
	./scripts/run_demo.sh

test:
	.venv/bin/python -m unittest discover -s tests -v
