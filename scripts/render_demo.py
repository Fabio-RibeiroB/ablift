"""Render docs/demo.svg, the README's terminal screenshot, from a real ablift run.

Run `make docs` after changing CLI output. tests/test_docs.py fails if the committed
image no longer matches what this script renders.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import contextmanager
from pathlib import Path

from click.testing import CliRunner
from rich.console import Console
from rich.json import JSON
from rich.text import Text

from ablift.cli import cli

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "demo.svg"
INPUT = "examples/checkout_test.csv"
ARGS = ["analyze", "--input", INPUT, "--method", "sequential"]
JQ_FILTER = ".recommendation | {action, decision_confidence, rationale}"


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _jq_equivalent(result: dict) -> dict:
    """Python equivalent of JQ_FILTER, so the image shows exactly what jq would print."""
    rec = result["recommendation"]
    return {k: rec[k] for k in ("action", "decision_confidence", "rationale")}


def _prompt(command: str, continuation: str | None = None) -> Text:
    line = Text("$ ", style="bold green")
    line.append(command, style="bold")
    if continuation:
        line.append(" \\\n    " + continuation, style="bold")
    return line


def render_svg() -> str:
    with _cwd(ROOT):
        run = CliRunner().invoke(cli, ARGS, catch_exceptions=False)
    if run.exit_code != 0:
        raise RuntimeError(f"ablift {' '.join(ARGS)} failed:\n{run.output}")

    console = Console(record=True, width=84, file=io.StringIO(), force_terminal=True)
    console.print(_prompt(f"cat {INPUT}"))
    console.print((ROOT / INPUT).read_text(encoding="utf-8").rstrip(), highlight=False)
    console.print()
    console.print(_prompt(f"ablift {' '.join(ARGS)}", f"| jq '{JQ_FILTER}'"))
    console.print(JSON(json.dumps(_jq_equivalent(json.loads(run.output))), indent=2))
    return console.export_svg(title="ablift")


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(render_svg(), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
