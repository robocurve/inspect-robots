"""Mirror Inspect AI's ``.env`` auto-loading because the core must stay dependency-free."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path


def read_dotenv(path: Path) -> dict[str, str]:
    """Return supported key-value pairs, or an empty mapping when the file is unreadable."""
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    parsed: dict[str, str] = {}
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ")
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[key] = value
    return parsed


def init_dotenv(environ: MutableMapping[str, str], path: Path | None = None) -> None:
    """Add file values only for keys absent from the supplied environment mapping."""
    dotenv_path = Path(".env") if path is None else path
    for key, value in read_dotenv(dotenv_path).items():
        environ.setdefault(key, value)
