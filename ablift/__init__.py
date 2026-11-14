from importlib.metadata import PackageNotFoundError, version

from .engine import analyze, parse_payload
from .reporting import build_markdown_report

try:
    __version__ = version("ablift")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled checkout
    __version__ = "0.0.0"

__all__ = ["analyze", "parse_payload", "build_markdown_report", "__version__"]
