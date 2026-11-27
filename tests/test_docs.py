"""Guardrails that keep README.md and docs/ in step with the code.

These run in CI and in the pre-commit hook. When one fails, fix the docs (or run
`make docs` to re-render the demo image) rather than loosening the check.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pytest

from ablift.cli import cli

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOC_FILES = [README, *sorted((ROOT / "docs").glob("*.md"))]
SKIP_MARKER = "<!-- docs-test: skip -->"


def _all_docs_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in DOC_FILES)


def _bash_blocks(path: Path) -> list[str]:
    """Fenced ```bash blocks, except those directly preceded by the skip marker."""
    text = path.read_text(encoding="utf-8")
    blocks = []
    for match in re.finditer(r"```bash\n(.*?)```", text, re.S):
        before = text[: match.start()].rstrip().splitlines()
        if before and before[-1].strip() == SKIP_MARKER:
            continue
        blocks.append(match.group(1))
    return blocks


def _commands(block: str) -> list[str]:
    """Shell commands in a block that invoke ablift, with line continuations joined."""
    joined = re.sub(r"\\\n\s*", " ", block)
    commands = []
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and re.search(r"(^|\|\s*)ablift\b", line):
            commands.append(re.sub(r"\s+#.*$", "", line))
    return commands


DOC_COMMANDS = [
    pytest.param(cmd, id=f"{path.relative_to(ROOT)}: {cmd[:60]}")
    for path in DOC_FILES
    for block in _bash_blocks(path)
    for cmd in _commands(block)
]


def _run(command: str, workdir: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(
        ["bash", "-o", "pipefail", "-c", command],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_every_documented_ablift_command_runs(tmp_path: Path) -> None:
    """Every `ablift ...` example in README.md and docs/ must succeed as written."""
    (tmp_path / "examples").symlink_to(ROOT / "examples")
    commands = [p.values[0] for p in DOC_COMMANDS]
    assert commands, "no runnable ablift examples found in the docs"
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda c: (c, _run(c, tmp_path)), commands))
    failures = [
        f"$ {cmd}\n  exit {res.returncode}: {res.stderr.strip()[-400:]}"
        for cmd, res in results
        if res.returncode != 0
    ]
    assert not failures, (
        "Documented commands fail. Fix the example, or put "
        f"'{SKIP_MARKER}' on the line before a block that is not meant to run:\n\n"
        + "\n\n".join(failures)
    )


def _documented_flags(command: click.Command) -> list[str]:
    flags = []
    for param in command.params:
        if isinstance(param, click.Option) and "--help" not in param.opts:
            flags += [o for o in [*param.opts, *param.secondary_opts] if o.startswith("--")]
    return flags


def test_every_command_and_option_is_documented() -> None:
    text = _all_docs_text()
    assert isinstance(cli, click.Group)
    missing = []
    for name, command in sorted(cli.commands.items()):
        if command.hidden:
            continue
        if f"ablift {name}" not in text:
            missing.append(f"command: ablift {name}")
        missing += [
            f"option: ablift {name} {flag}"
            for flag in _documented_flags(command)
            if not re.search(rf"{re.escape(flag)}(?![\w-])", text)
        ]
    assert not missing, "Undocumented CLI surface (add it to README.md or docs/):\n" + "\n".join(
        missing
    )


def test_readme_lists_every_command_and_method() -> None:
    readme = README.read_text(encoding="utf-8")
    assert isinstance(cli, click.Group)
    for name, command in cli.commands.items():
        if not command.hidden:
            assert f"`ablift {name}`" in readme, f"README command table is missing ablift {name}"
    method = next(p for p in cli.commands["analyze"].params if p.name == "method")
    assert isinstance(method.type, click.Choice)
    for choice in method.type.choices:
        assert f"`{choice}`" in readme, f"README feature table is missing --method {choice}"


def _slug(heading: str) -> str:
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


REPO_URL = re.compile(
    r"https://(?:github\.com/Fabio-RibeiroB/ablift/blob/main|"
    r"raw\.githubusercontent\.com/Fabio-RibeiroB/ablift/main)/"
)


def test_readme_links_work_on_pypi() -> None:
    """PyPI shows README.md alone, so its links and images must be absolute URLs."""
    relative = [
        t
        for t in re.findall(r"\]\(([^)\s]+)\)", README.read_text(encoding="utf-8"))
        if not re.match(r"https?://|mailto:|#", t)
    ]
    assert not relative, (
        "README.md has relative links that break on PyPI; use "
        "https://github.com/Fabio-RibeiroB/ablift/blob/main/<path> (or raw.githubusercontent "
        f"for images): {relative}"
    )


def test_links_and_anchors_resolve() -> None:
    broken = []
    for path in DOC_FILES:
        for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")):
            local = REPO_URL.match(target)
            if local:  # links to this repo's main branch must point at real files
                target = str((ROOT / target[local.end() :]).relative_to(path.parent, walk_up=True))
            elif re.match(r"https?://|mailto:", target):
                continue
            file_part, _, anchor = target.partition("#")
            dest = (path.parent / file_part).resolve() if file_part else path
            if not dest.exists():
                broken.append(f"{path.relative_to(ROOT)}: {target} (missing file)")
            elif anchor and dest.suffix == ".md":
                headings = re.findall(r"^#+ (.+)$", dest.read_text(encoding="utf-8"), re.M)
                if anchor not in {_slug(h) for h in headings}:
                    broken.append(f"{path.relative_to(ROOT)}: {target} (missing heading)")
    assert not broken, "Broken links:\n" + "\n".join(broken)


def test_every_docs_page_is_linked_from_readme() -> None:
    readme = README.read_text(encoding="utf-8")
    orphans = [p.name for p in DOC_FILES[1:] if f"docs/{p.name}" not in readme]
    assert not orphans, f"docs pages not linked from README: {orphans}"


def test_install_instructions_match_release_state() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    install = README.read_text(encoding="utf-8").split("## Install", 1)[1].split("\n## ", 1)[0]
    is_prerelease = re.search(r"(a|b|rc|\.dev)\d+$", version) is not None
    allows_pre = "--prerelease=allow" in install or "--pre " in install
    if is_prerelease:
        assert allows_pre, f"{version} is a pre-release: README install must pass --pre"
    else:
        assert not allows_pre, f"{version} is stable: drop the pre-release install flags"


def test_demo_image_matches_current_cli_output() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from render_demo import OUTPUT, render_svg
    finally:
        sys.path.pop(0)
    assert OUTPUT.read_text(encoding="utf-8") == render_svg(), (
        "docs/demo.svg is stale; run `make docs`"
    )


def test_no_generated_artifacts_are_tracked() -> None:
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    junk = re.compile(r"(^|/)(\.coverage|__pycache__|\.pytest_cache|.*\.egg-info|dist|build)(/|$)")
    assert not [f for f in tracked if junk.search(f)]
