"""Resolve metric camera depth and render it for LLM observation messages."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots.types import Observation
from inspect_robots_agent._png import png_data_url


def _render_depth(depth: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Render metric depth as grayscale with near pixels brighter than far pixels.

    Requires at least one valid (finite, positive) pixel; ``depth_parts``
    guarantees this via its <1%-valid guard before calling.
    """
    valid = np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    lo, hi = (float(value) for value in np.percentile(values, (2, 98)))
    if lo == hi:
        hi = lo + 1e-3

    rendered = np.zeros(depth.shape, dtype=np.uint8)
    scaled = 255.0 - (depth[valid] - lo) * (255.0 - 64.0) / (hi - lo)
    rendered[valid] = np.clip(scaled, 64.0, 255.0).round().astype(np.uint8)
    return rendered


def depth_parts(
    name: str,
    entry: npt.NDArray[np.float64] | str,
    step_label: str,
) -> list[dict[str, Any]]:
    """Build a metric depth label and optional inline grayscale PNG.

    Array entries must be 2-D; ``resolve_depth`` guarantees this (a str entry
    is resolution-failure text and passes through as a plain line).
    """
    if isinstance(entry, str):
        return [{"type": "text", "text": entry}]

    valid = np.isfinite(entry) & (entry > 0)
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = valid_count / entry.size if entry.size else 0.0
    # Floored, not rounded: 0.9% valid must not print as "1% valid" beside
    # the no-usable-depth message, and 99.5% must not claim "100% valid".
    valid_pct = int(100 * valid_fraction)
    if valid_fraction < 0.01:
        return [
            {
                "type": "text",
                "text": f"depth {name!r}: no usable depth this frame ({valid_pct}% valid)",
            }
        ]

    lo, hi = (float(value) for value in np.percentile(entry[valid], (2, 98)))
    if lo == hi:
        hi = lo + 1e-3
    suffix = f" ({step_label})" if step_label else ""
    text = (
        f"depth {name!r}{suffix}: bright {lo:.2f} m -> dim {hi:.2f} m "
        f"(2nd-98th pctl), {valid_pct}% valid"
    )
    center = (entry.shape[0] // 2, entry.shape[1] // 2)
    if valid[center]:
        text += f", center {entry[center]:.2f} m"
    text += ":"
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": png_data_url(_render_depth(entry))}},
    ]


def resolve_depth(observation: Observation) -> dict[str, npt.NDArray[np.float64] | str]:
    """Resolve each camera's depth entry once and preserve failures as text."""
    resolved: dict[str, npt.NDArray[np.float64] | str] = {}
    for name in observation.images:
        key = f"{name}_depth"
        if key not in observation.extra:
            continue
        try:
            value = observation.extra[key]
            if callable(value):
                value = value()
            depth = np.asarray(value, dtype=np.float64)
            if depth.ndim != 2:
                raise ValueError(f"expected a 2-D array, got shape {depth.shape}")
        except Exception as exc:
            resolved[name] = f"depth {name!r} unavailable: {exc}"
        else:
            resolved[name] = depth
    return resolved
