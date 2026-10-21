"""Links in the Markdown docs must resolve on GitHub.

The guides in ``docs/`` are read directly on GitHub, so nothing else catches a
link broken by a renamed page, heading or symbol. This checks relative links,
``#anchor`` fragments against GitHub's heading slugs, that a linked API symbol
is defined in the source file it points at, and that absolute links into this
repository name a path that exists. Code blocks and inline code are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPO_URL = re.compile(
    r"https://github\.com/robocurve/inspect-robots/(?:blob|tree)/main/([^)\s#?]*)"
)
LINK = re.compile(r"!?\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(```|~~~)")
INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)+?\1")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
SYMBOL = re.compile(r"^`([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?`$")
FORBIDDEN = {
    "](/api/": "site-only API link",
    "][inspect_robots": "mkdocstrings autoref",
    ":::": "Docusaurus admonition fence; use a GitHub alert",
}
REF_DEFINITION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s")


def doc_pages() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def prose_lines(text: str) -> list[str]:
    """``text`` split into lines, fenced blocks blanked (line numbers kept)."""
    lines, fence = [], None
    for line in text.split("\n"):
        marker = FENCE.match(line.lstrip())
        if fence is None and marker:
            fence = marker.group(1)
            lines.append("")
        elif fence is not None:
            if line.lstrip().startswith(fence):
                fence = None
            lines.append("")
        else:
            lines.append(line)
    return lines


def slug(heading: str) -> str:
    """GitHub's anchor for a heading (before duplicate suffixes)."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links -> text
    text = re.sub(r"<[^>]+>", "", text).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    result = set()
    for line in prose_lines(path.read_text(encoding="utf-8")):
        match = HEADING.match(line)
        if not match:
            continue
        base = slug(match.group(1))
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def defines(source: Path, name: str) -> bool:
    pattern = rf"^(?:class|(?:async\s+)?def)\s+{name}\b|^{name}\s*[:=]"
    return re.search(pattern, source.read_text(encoding="utf-8"), re.M) is not None


def page_problems(page: Path, root: Path = ROOT) -> list[str]:
    problems = []
    rel = page.relative_to(root)
    for number, raw in enumerate(prose_lines(page.read_text(encoding="utf-8")), 1):
        where = f"{rel}:{number}"
        if REF_DEFINITION.match(raw):
            problems.append(f"{where}: reference-style link definition; use an inline link")
        spans = [m.span() for m in INLINE_CODE.finditer(raw)]
        line = INLINE_CODE.sub(lambda m: "x" * len(m.group(0)), raw)
        for needle, why in FORBIDDEN.items():
            if needle in line:
                problems.append(f"{where}: {why}")
        for match in LINK.finditer(raw):
            if any(start <= match.start() < end for start, end in spans):
                continue  # the link syntax is itself inside inline code
            text, target = match.group(1), match.group(2)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, fragment = target.partition("#")
            resolved = (page.parent / path_part).resolve() if path_part else page
            if not resolved.exists():
                problems.append(f"{where}: {target} does not exist")
                continue
            if fragment and resolved.suffix == ".md" and fragment not in anchors(resolved):
                problems.append(f"{where}: no heading for #{fragment} in {path_part or rel.name}")
            symbol = SYMBOL.match(text)
            if symbol and resolved.suffix == ".py" and not defines(resolved, symbol.group(1)):
                problems.append(f"{where}: {symbol.group(1)} is not defined in {path_part}")
    return problems


@pytest.mark.parametrize("page", doc_pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_doc_page_links_resolve(page: Path) -> None:
    assert page_problems(page) == []


@pytest.mark.parametrize(
    "page", [ROOT / "README.md", *doc_pages()], ids=lambda p: str(p.relative_to(ROOT))
)
def test_repo_urls_name_existing_paths(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    missing = [m.group(1) for m in REPO_URL.finditer(text) if not (ROOT / m.group(1)).exists()]
    assert missing == []


def test_checker_catches_broken_links(tmp_path: Path) -> None:
    (tmp_path / "other.md").write_text("# Real heading\n", encoding="utf-8")
    (tmp_path / "mod.py").write_text("class Real:\n    pass\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "\n".join(
            [
                "# Page",
                "[ok](other.md#real-heading) [ok](#page) [`Real`](mod.py)",
                "[gone](missing.md)",
                "[bad anchor](other.md#nope)",
                "[`Fake`](mod.py)",
                "[api](/api/#inspect_robots.x)",
                "`[in code](missing.md)`",
                "```",
                "[in fence](missing.md)",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    problems = page_problems(page, root=tmp_path)
    assert [p.split(": ", 1)[1] for p in problems] == [
        "missing.md does not exist",
        "no heading for #nope in other.md",
        "Fake is not defined in mod.py",
        "site-only API link",
        "/api/#inspect_robots.x does not exist",
    ]


def test_slug_matches_github() -> None:
    assert slug("Speaking policy notes (`--speak`)") == "speaking-policy-notes---speak"
    assert slug("Several rigs on one host") == "several-rigs-on-one-host"


def test_symbol_check_sees_the_api_links() -> None:
    # Guards against the symbol check silently matching nothing again.
    checked = sum(
        1
        for page in doc_pages()
        for raw in prose_lines(page.read_text(encoding="utf-8"))
        for match in LINK.finditer(raw)
        if SYMBOL.match(match.group(1)) and match.group(2).endswith(".py")
    )
    assert checked >= 60
