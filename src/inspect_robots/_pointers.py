"""Resolve portable log pointers without allowing reads outside the log directory.

Both the transcript summarizer and HTML/CLI wire viewers consume paths stored
in logs as foreign text.  Keeping their traversal guard in this leaf module
also avoids an ``_html`` → ``_summarize`` → ``cli`` → ``_html`` import cycle.
"""

from __future__ import annotations

from pathlib import Path


def resolve_log_pointer(log_path: Path, pointer: object) -> Path | None:
    """Resolve a non-empty relative pointer beneath the saved log's directory."""
    if not isinstance(pointer, str) or not pointer:
        return None
    try:
        root = log_path.parent.resolve()
        target = (root / pointer).resolve()
    except OSError:
        return None
    if not target.is_relative_to(root):
        return None
    return target


def derive_blob_dir(log_path: Path, target: Path) -> Path | None:
    """Derive a wire sidecar's run-scoped blobs dir, refusing to leave the log dir.

    ``target`` is a ``resolve_log_pointer`` result for ``wire/<run>/<trial>/calls.jsonl``;
    a shallower hand-crafted pointer would put ``parent.parent`` above the
    guarded root, so the derivation re-checks containment.
    """
    blob_dir = target.parent.parent / "blobs"
    if not blob_dir.is_relative_to(log_path.parent.resolve()):
        return None
    return blob_dir
