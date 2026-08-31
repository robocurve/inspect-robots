"""Mirror Inspect AI's ``.env`` auto-loading because the core must stay dependency-free."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from pathlib import Path


def _closing_quote(value: str) -> int:
    """Return the first unescaped matching quote, or -1 when none exists."""
    backslashes = 0
    for index, char in enumerate(value[1:], start=1):
        if char == "\\":
            backslashes += 1
            continue
        if char == value[0] and backslashes % 2 == 0:
            return index
        backslashes = 0
    return -1


def read_dotenv(path: Path) -> dict[str, str]:
    """Return supported key-value pairs, or an empty mapping when the file is unreadable."""
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
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
        if value[:1] in {"'", '"'} and (closing := _closing_quote(value)) != -1:
            # python-dotenv semantics for quoting and comments only;
            # backslashes remain literal. A quoted value ends at the first
            # unescaped matching quote and keeps "#" literally; anything after
            # the closing quote (typically a comment) is ignored. An unmatched
            # opening quote falls through and is kept literally. Consequently,
            # a quoted value ending in a lone backslash keeps its surrounding quotes.
            value = value[1:closing]
        else:
            # python-dotenv semantics for quoting and comments only;
            # backslashes remain literal. An unquoted value ends at the first
            # whitespace-preceded "#".
            value = re.split(r"\s#", value, maxsplit=1)[0].rstrip()
        parsed[key] = value
    return parsed


def init_dotenv(environ: MutableMapping[str, str], path: Path | None = None) -> None:
    """Add file values only for keys absent from the supplied environment mapping."""
    dotenv_path = Path(".env") if path is None else path
    for key, value in read_dotenv(dotenv_path).items():
        environ.setdefault(key, value)
