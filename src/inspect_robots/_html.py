"""Render evaluation logs as dependency-free, self-contained HTML documents."""

from __future__ import annotations

import base64
import html
import json
import math
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from inspect_robots._pngenc import png_data_url
from inspect_robots._pointers import derive_blob_dir, read_jsonl_prefix, resolve_log_pointer
from inspect_robots.frames import _safe
from inspect_robots.log import EvalLog, SceneResult

_STATUS_DISPLAY = {"started": "running", "success": "completed"}
_JSON_STRING_LIMIT = 2048
# \d{1,12}, not \d+: a hostile multi-thousand-digit "step" must miss the
# regex and degrade rather than trip int()'s conversion-length limit.
_FRAME_LABEL_RE = re.compile(r"camera '(?P<name>.*)' \(step (?P<step>\d{1,12})\):")
_FRAME_PLACEHOLDER = "[image omitted: streamed camera frame]"
_FRAME_MAX_SIDE = 448
_VIDEO_BUDGET_BYTES = 30_000_000
_BLOB_SENTINEL_RE = re.compile(r"\$blob:([^\s]+)")
_BLOB_SHA_RE = re.compile(r"[0-9a-f]{64}")
_CAMERA_FRAME_RE = re.compile(r"^(.+)_(\d{6,})\.npy$")
_MISSING = object()

_FRAME_CLICK_SCRIPT = """document.addEventListener('click', (event) => {
  const cell = event.target.closest('.frame-cell');
  if (cell) cell.classList.toggle('wide');
});"""

_FLIPBOOK_SCRIPT = """document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.run-media[data-trial]').forEach((block) => {
    const trial = block.dataset.trial;
    const tabs = Array.from(block.querySelectorAll('[data-camera-tab]'));
    const panels = Array.from(block.querySelectorAll('[data-camera-panel]'));
    const transcript = block.closest('details.transcript');
    let timer = null;
    let active = tabs.find((tab) => tab.classList.contains('active'))?.dataset.cameraTab;

    const syncVideo = () => {
      const video = block.querySelector('.video-panel video');
      if (!video) return;
      if (!transcript || transcript.open) void video.play().catch(() => {});
      else video.pause();
    };

    const sourceFrames = (camera) => Array.from(document.querySelectorAll(
      `[data-trial="${CSS.escape(trial)}"][data-camera="${CSS.escape(camera)}"]`
    )).filter((element) => element.matches('img.frame'));
    const storageKey = (camera) =>
      `inspect-robots:flipbook:${location.pathname}:${trial}:${camera}`;
    const tabKey = `inspect-robots:flipbook-tab:${location.pathname}:${trial}`;
    try {
      const storedTab = sessionStorage.getItem(tabKey);
      if (storedTab && tabs.some((tab) => tab.dataset.cameraTab === storedTab)) {
        active = storedTab;
      }
    } catch (_error) { /* Storage can be disabled without disabling playback. */ }
    const readState = (camera) => {
      try { return JSON.parse(sessionStorage.getItem(storageKey(camera))) || {}; }
      catch (_error) { return {}; }
    };
    const writeState = (camera, state) => {
      try { sessionStorage.setItem(storageKey(camera), JSON.stringify(state)); }
      catch (_error) { /* Storage can be disabled without disabling playback. */ }
    };
    const stopTimer = () => {
      if (timer !== null) window.clearInterval(timer);
      timer = null;
    };
    const updateFlipbook = (panel, camera, index, paused) => {
      const frames = sourceFrames(camera);
      if (!frames.length) return;
      const position = ((index % frames.length) + frames.length) % frames.length;
      const player = panel.querySelector('.flipbook-player');
      const scrubber = panel.querySelector('[data-scrubber]');
      const step = panel.querySelector('[data-step-label]');
      const toggle = panel.querySelector('[data-play-toggle]');
      player.src = frames[position].src;
      player.alt = frames[position].alt;
      scrubber.max = String(frames.length - 1);
      scrubber.value = String(position);
      step.textContent = `step ${frames[position].dataset.step}`;
      toggle.textContent = paused ? 'Play' : 'Pause';
      panel.dataset.position = String(position);
      panel.dataset.paused = String(paused);
      writeState(camera, {index: position, paused});
    };
    const startFlipbook = (panel, camera) => {
      stopTimer();
      const state = readState(camera);
      const index = Number.isInteger(state.index) ? state.index : 0;
      const paused = state.paused === true;
      updateFlipbook(panel, camera, index, paused);
      if (!paused) timer = window.setInterval(() => {
        updateFlipbook(panel, camera, Number(panel.dataset.position || 0) + 1, false);
      }, 250);
    };
    const activate = (camera) => {
      active = camera;
      try { sessionStorage.setItem(tabKey, camera); }
      catch (_error) { /* Storage can be disabled without disabling playback. */ }
      stopTimer();
      const expanded = !transcript || transcript.open;
      tabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.cameraTab === camera));
      panels.forEach((panel) => {
        const selected = panel.dataset.cameraPanel === camera;
        panel.hidden = !selected;
        const video = panel.querySelector('video');
        if (video && (!selected || !expanded)) video.pause();
        if (video && selected && expanded) void video.play().catch(() => {});
        if (selected && expanded && panel.classList.contains('flipbook-panel')) {
          startFlipbook(panel, camera);
        }
      });
    };
    tabs.forEach((tab) => tab.addEventListener('click', () => activate(tab.dataset.cameraTab)));
    panels.filter((panel) => panel.classList.contains('flipbook-panel')).forEach((panel) => {
      const camera = panel.dataset.cameraPanel;
      panel.querySelector('[data-play-toggle]').addEventListener('click', () => {
        const paused = panel.dataset.paused !== 'true';
        updateFlipbook(panel, camera, Number(panel.dataset.position || 0), paused);
        if (camera === active) paused ? stopTimer() : startFlipbook(panel, camera);
      });
      panel.querySelector('[data-scrubber]').addEventListener('input', (event) => {
        updateFlipbook(panel, camera, Number(event.target.value), true);
        if (camera === active) stopTimer();
      });
    });
    if (transcript) transcript.addEventListener('toggle', () => {
      activate(active);
      syncVideo();
    });
    if (active) activate(active);
    syncVideo();
  });
});"""


@dataclass
class _FrameBudget:
    """Shared mutable accounting for frame payloads in one document."""

    limit: int
    embedded: int = 0
    payload_bytes: int = 0
    truncated: bool = False
    cache: dict[tuple[str, str, int], str] | None = None


@dataclass(frozen=True)
class _FrameReference:
    """Identify one transcript placeholder and its exact stored-frame key."""

    parts: list[object]
    label_index: int
    placeholder_index: int
    trial_prefix: str
    camera: str
    step: int


@dataclass(frozen=True)
class _FrameContext:
    """The filesystem correlation state for one trial transcript."""

    frames_dir: Path
    trial_prefix: str
    budget: _FrameBudget
    rendered: list[tuple[str, int]] | None = None


@dataclass
class _Turn:
    """Retain original transcript messages and frame references for one turn."""

    messages: list[object]
    references: tuple[_FrameReference, ...]
    preamble: bool = False
    feedback: list[dict[str, Any]] | None = None


@dataclass
class _VideoBudget:
    """Track base64 MP4 characters spent by one report in document order.

    ``truncated`` is sticky like the frame budget's: the first trial composite
    that overflows denies every later one, and later trials skip their encode
    entirely rather than paying ffmpeg cost for a discarded payload.
    """

    limit: int = _VIDEO_BUDGET_BYTES
    encoded: int = 0
    warned: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class _VideoContext:
    """Carry one render pass's MP4 eligibility, encoder path, rate, and budget."""

    enabled: bool
    ffmpeg: str | None
    fps: float
    budget: _VideoBudget


@dataclass
class _WireContext:
    """Track one trial's blob directory, budget, and emitted image anchors."""

    blob_dir: Path
    trial_id: str
    budget: _FrameBudget
    emitted: set[str]
    denied: set[str]
    elide_media: bool = False


_STYLES = """
:root {
  color-scheme: light dark;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #20242b;
  --muted: #68707d;
  --line: #dfe3e8;
  --green: #19723b;
  --green-bg: #e9f6ed;
  --red: #a12a2a;
  --red-bg: #fbecec;
  --grey: #626a75;
  --grey-bg: #eef0f2;
  --neutral: #45546a;
  --neutral-bg: #edf1f6;
  --user: #3178c6;
  --assistant: #7a55b5;
  --tool: #36866a;
  --system: #7a828d;
  --amber: #8a5700;
  --amber-line: #d69b2d;
  --amber-bg: #fff5d9;
}
@media (prefers-color-scheme: light) {
  :root { color-scheme: light; }
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111419;
    --panel: #191d24;
    --text: #e7eaf0;
    --muted: #a4acb8;
    --line: #343a45;
    --green: #7ed99a;
    --green-bg: #193b27;
    --red: #ff9b9b;
    --red-bg: #492323;
    --grey: #c0c5cd;
    --grey-bg: #343943;
    --neutral: #b9c9df;
    --neutral-bg: #293342;
    --user: #73b7ff;
    --assistant: #bd9bed;
    --tool: #79c9ab;
    --system: #adb4be;
    --amber: #ffd484;
    --amber-line: #b77a16;
    --amber-bg: #3b2d12;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1120px, calc(100% - 32px)); margin: 0 auto 64px; }
header { border-bottom: 1px solid var(--line); background: var(--panel); }
.header-inner { width: min(1120px, calc(100% - 32px)); margin: auto; padding: 28px 0 22px; }
.header-top { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
h1 { margin: 0; font-size: 24px; font-weight: 650; min-width: 0; overflow-wrap: anywhere; }
h2 { margin: 0; font-size: 18px; font-weight: 650; min-width: 0; overflow-wrap: anywhere; }
h3 { margin: 24px 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; }
.meta { color: var(--muted); margin-top: 8px; display: flex; gap: 16px; flex-wrap: wrap; }
.badge, .chip { display: inline-block; border-radius: 999px; padding: 2px 9px; font-size: 12px; }
.status-completed { color: var(--green); background: var(--green-bg); }
.status-running { color: var(--amber); background: var(--amber-bg); }
.status-error { color: var(--red); background: var(--red-bg); }
.status-cancelled { color: var(--grey); background: var(--grey-bg); }
.status-neutral { color: var(--neutral); background: var(--neutral-bg); }
.chip { color: var(--muted); background: var(--grey-bg); }
.running-banner {
  margin: 18px 0 0; padding: 11px 14px; color: var(--amber);
  background: var(--amber-bg); border: 1px solid var(--amber-line); border-radius: 7px;
  font-weight: 650;
}
.spec-strip {
  margin: 24px 0; padding: 18px 20px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px;
}
dl {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: 14px 24px; margin: 0;
}
dt { color: var(--muted); font-size: 12px; }
dd { margin: 2px 0 0; overflow-wrap: anywhere; }
.metrics {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
  gap: 12px; margin-bottom: 28px;
}
.stat {
  padding: 14px 16px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px;
}
.stat-name { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.stat-value { margin-top: 3px; font-size: 19px; font-weight: 620; }
.scene {
  margin: 18px 0; padding: 22px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 9px;
}
.scene-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.instruction, .error { margin: 12px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.error { color: var(--red); }
.score-row { display: flex; flex-wrap: wrap; gap: 7px; margin: 8px 0 0; }
.score-chip {
  border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; font-size: 12px;
  overflow-wrap: anywhere;
}
details { border-top: 1px solid var(--line); margin-top: 18px; padding-top: 12px; }
summary { cursor: pointer; color: var(--muted); font-weight: 600; }
.conversation { margin-top: 14px; }
.turn { margin: 18px 0 24px; }
.turn + .turn { border-top: 1px solid var(--line); padding-top: 18px; }
.turn-step {
  margin: 0 0 8px; color: var(--muted); font-size: 12px;
  font-weight: 750; letter-spacing: .07em; text-transform: uppercase;
}
.message { margin: 13px 0; padding: 2px 0 2px 13px; border-left: 3px solid var(--system); }
.message.user { border-color: var(--user); }
.message.assistant { border-color: var(--assistant); }
.message.tool { border-color: var(--tool); margin-left: 20px; }
.role { color: var(--muted); font-size: 12px; font-weight: 650; text-transform: uppercase; }
.content { margin-top: 3px; white-space: pre-wrap; overflow-wrap: anywhere; }
img.frame {
  display: block; max-width: 100%; height: auto; margin: 6px 0;
  border: 1px solid var(--line); border-radius: 6px;
}
.frame-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px; margin: 6px 0;
}
.frame-cell { margin: 0; min-width: 0; }
.frame-cell figcaption {
  font-size: 12px; color: var(--muted); margin-bottom: 2px; overflow-wrap: anywhere;
}
.frame-cell img.frame { width: 100%; max-width: 448px; margin: 0; cursor: zoom-in; }
.frame-cell.wide { grid-column: 1 / -1; }
.frame-cell.wide img.frame { max-width: none; cursor: zoom-out; }
.system-message {
  margin: 13px 0; padding: 0 0 0 13px; border: 0;
  border-left: 3px solid var(--system);
}
.system-message summary { color: var(--muted); }
.call {
  margin-top: 8px; overflow-wrap: anywhere;
  font: 13px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
}
.pretty-call { margin: 10px 0; padding: 10px 12px; background: var(--bg); border-radius: 6px; }
.call-name { font-weight: 700; margin-bottom: 7px; }
.call-chips, .feedback-row { display: flex; gap: 6px; flex-wrap: wrap; }
.call-chip {
  border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; font-size: 12px;
}
.call-key { color: var(--muted); margin-right: 4px; }
.feedback-row { margin: 8px 0; }
.feedback-chip {
  padding: 5px 9px; border-radius: 6px; color: var(--tool); background: var(--green-bg);
  overflow-wrap: anywhere;
}
.feedback-source {
  font-size: 10px; font-weight: 750; text-transform: uppercase; margin-right: 6px;
}
.raw-transcript { border-top: 0; margin: 12px 0 0; padding-top: 0; }
.raw-transcript .message { margin: 9px 0; }
.agent-note {
  margin: 10px 0 6px; padding: 9px 11px; color: var(--amber);
  background: var(--amber-bg); border-left: 3px solid var(--amber-line);
}
.grader-note {
  margin: 8px 0; padding: 9px 11px; color: var(--muted);
  background: var(--bg); border: 1px solid var(--line); overflow-wrap: anywhere;
}
.note-label {
  display: block; font-size: 10px; font-weight: 750;
  letter-spacing: .09em; text-transform: uppercase;
}
pre {
  padding: 14px; overflow: auto; background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px;
  font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
}
.none { color: var(--muted); margin: 28px 0; }
.wire-static, .wire-change { margin: 10px 0; }
.wire-change { color: var(--muted); }
.wire-media { margin: 8px 0; }
.wire-blob { display: block; }
.wire-broken { color: var(--red); }
.wire-placeholder { color: var(--muted); }
.run-media { margin: 14px 0 20px; padding: 12px; background: var(--bg); border-radius: 7px; }
.run-media-head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-weight: 650; }
.camera-order { color: var(--muted); font-size: 12px; margin: 4px 0 8px; }
.camera-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; }
.camera-tab, .flipbook-controls button {
  border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px;
  color: var(--text); background: var(--panel); cursor: pointer;
}
.camera-tab.active { color: var(--user); border-color: var(--user); }
.camera-panel video, .flipbook-player {
  display: block; width: 100%; max-height: 70vh; object-fit: contain; background: #000;
  border: 1px solid var(--line); border-radius: 6px;
}
.flipbook-controls { display: flex; gap: 9px; align-items: center; margin-top: 8px; }
.flipbook-controls input { flex: 1; }
.step-label { min-width: 62px; color: var(--muted); font-size: 12px; }
""".strip()


def _display_status(status: str) -> str:
    """Return the stable human-facing form of a persisted status value."""
    return _STATUS_DISPLAY.get(status, status)


def _chat_content(content: object) -> str | None:
    """Render text from an OpenAI-style content value, collapsing media parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        else:
            parts.append("[image]")
    return "\n".join(parts)


def _is_chat_transcript(transcript: object) -> bool:
    """Recognize a non-empty list of role-bearing message dictionaries."""
    return (
        isinstance(transcript, list)
        and bool(transcript)
        and all(isinstance(message, dict) and "role" in message for message in transcript)
    )


def _escape(value: object) -> str:
    """Escape one foreign value at its HTML interpolation boundary."""
    return html.escape(str(value), quote=True)


def _number(value: int | float | None) -> str:
    """Format numeric log values compactly before their interpolation boundary."""
    return "n/a" if value is None else f"{value:.4g}"


def _value(value: object) -> str:
    """Format a scalar or structured spec value without HTML escaping it."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _definition(label: object, value: object) -> str:
    """Render one escaped definition pair."""
    return f"<div><dt>{_escape(label)}</dt><dd>{_escape(value)}</dd></div>"


def _status_class(status: str) -> str:
    """Map a persisted status to one of the fixed badge color classes."""
    displayed = _display_status(status)
    if displayed == "completed":
        return "status-completed"
    if displayed == "running":
        return "status-running"
    if status == "error":
        return "status-error"
    if status == "cancelled":
        return "status-cancelled"
    return "status-neutral"


def _status_badge(status: str) -> str:
    """Render a status with a fixed class and escaped display label."""
    return f'<span class="badge {_status_class(status)}">{_escape(_display_status(status))}</span>'


def _agent_notes(name: str, arguments: object) -> list[str]:
    """Extract non-empty agent notes from supported tool argument shapes."""
    parsed: object = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, dict):
        return []

    keys = ["note"]
    if name == "done":
        keys.append("summary")
    elif name == "give_up":
        keys.append("reason")
    notes: list[str] = []
    for key in keys:
        note = parsed.get(key)
        if isinstance(note, str) and note.strip():
            notes.append(note)
    return notes


def _render_tool_call(raw_call: object) -> str:
    """Render one tolerant OpenAI-style tool call as notes and argument chips."""
    if not isinstance(raw_call, dict):
        return ""
    function = raw_call.get("function")
    if not isinstance(function, dict):
        return ""
    name = str(function.get("name", "unknown"))
    arguments = function.get("arguments", "")
    notes = "".join(
        f'<div class="agent-note"><span class="note-label">agent note</span>{_escape(note)}</div>'
        for note in _agent_notes(name, arguments)
    )
    parsed: object = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return f"{notes}{_raw_tool_call(name, arguments)}"
    if not isinstance(parsed, dict):
        return f"{notes}{_raw_tool_call(name, arguments)}"

    excluded = {"note"}
    if name == "done":
        excluded.add("summary")
    elif name == "give_up":
        excluded.add("reason")
    hindsight = parsed.get("hindsight")
    hindsight_block = (
        '<div class="agent-note"><span class="note-label">hindsight</span>'
        f"{_escape(hindsight)}</div>"
        if isinstance(hindsight, str) and hindsight.strip()
        else ""
    )
    chips: list[str] = []
    for key, value in parsed.items():
        if key in excluded or key == "hindsight":
            continue
        if isinstance(value, dict) and value and all(_is_number(item) for item in value.values()):
            chips.extend(
                _call_chip(item_key, f"{float(item_value):.4f}")
                for item_key, item_value in value.items()
            )
        elif isinstance(value, list):
            chips.append(_call_chip(key, ", ".join(str(item) for item in value)))
        elif value is None or isinstance(value, (str, int, float, bool)):
            chips.append(_call_chip(key, value))
        else:
            return f"{notes}{_raw_tool_call(name, arguments)}"
    return (
        f'{notes}<div class="pretty-call"><div class="call-name">{_escape(name)}</div>'
        f'<div class="call-chips">{"".join(chips)}</div>{hindsight_block}</div>'
    )


def _is_number(value: object) -> bool:
    """Accept real JSON numbers while excluding booleans from numeric chips."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _call_chip(key: object, value: object) -> str:
    """Render one escaped key and value argument chip."""
    shown = (
        json.dumps(value, ensure_ascii=False) if value is None or isinstance(value, bool) else value
    )
    return (
        '<span class="call-chip"><span class="call-key">'
        f"{_escape(key)}</span>{_escape(shown)}</span>"
    )


def _raw_tool_call(name: str, arguments: object) -> str:
    """Render the legacy raw call form for malformed or unsupported arguments."""
    shown = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return f'<div class="call">{_escape(name)}({_escape(shown)})</div>'


def _load_frame(frame_ctx: _FrameContext, name: str, step: int) -> npt.NDArray[np.uint8] | None:
    """Load one exact-match stored frame, degrading every invalid artifact to ``None``."""
    if frame_ctx.budget.truncated:
        return None
    path = frame_ctx.frames_dir / f"{frame_ctx.trial_prefix}_{_safe(name)}_{step:06d}.npy"
    if not path.exists():
        return None
    try:
        array = cast("npt.NDArray[Any]", np.load(path, allow_pickle=False))
        if array.dtype != np.uint8 or array.size == 0:
            return None
        if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in {1, 3, 4})):
            return None
    except Exception:
        return None
    longest = max(array.shape[0], array.shape[1])
    if longest > _FRAME_MAX_SIDE:
        stride = math.ceil(longest / _FRAME_MAX_SIDE)
        array = array[::stride, ::stride]
    return cast("npt.NDArray[np.uint8]", array)


def _frame_image(frame_ctx: _FrameContext, name: str, step: int) -> str | None:
    """Render one correlated frame if it is valid and fits the shared budget."""
    budget = frame_ctx.budget
    if budget.cache is not None:
        source = budget.cache.get((frame_ctx.trial_prefix, name, step))
        if source is None:
            return None
        return _frame_markup(source, frame_ctx, name, step)
    array = _load_frame(frame_ctx, name, step)
    if array is None:
        return None
    source = png_data_url(array)
    payload_size = len(source.partition(",")[2])
    if budget.limit and budget.payload_bytes + payload_size > budget.limit:
        budget.truncated = True
        return None
    budget.payload_bytes += payload_size
    budget.embedded += 1
    return _frame_markup(source, frame_ctx, name, step)


def _frame_markup(source: str, frame_ctx: _FrameContext, name: str, step: int) -> str:
    """Return the shared image markup for one already encoded frame."""
    if frame_ctx.rendered is not None:
        frame_ctx.rendered.append((name, step))
    return (
        f'<img class="frame" loading="lazy" alt="camera {_escape(name)} step {step}" '
        f'data-camera="{_escape(name)}" data-step="{step}" '
        f'data-trial="{_escape(frame_ctx.trial_prefix)}" '
        f'src="{source}">'
    )


def _frame_references(transcript: object, trial_prefix: str) -> tuple[_FrameReference, ...]:
    """Enumerate renderable user placeholders in stable transcript order."""
    if not _is_chat_transcript(transcript):
        return ()
    references: list[_FrameReference] = []
    for raw_message in cast(list[object], transcript):
        if not isinstance(raw_message, dict) or raw_message.get("role") != "user":
            continue
        raw_content = raw_message.get("content")
        if not isinstance(raw_content, list):
            continue
        parts = cast(list[object], raw_content)
        pending: tuple[str, int, int] | None = None
        for index, part in enumerate(parts):
            if not (isinstance(part, dict) and part.get("type") == "text"):
                continue
            raw_text = part.get("text", "")
            if not isinstance(raw_text, str):
                continue
            label = _FRAME_LABEL_RE.fullmatch(raw_text)
            if label is not None:
                pending = (label.group("name"), int(label.group("step")), index)
            elif raw_text == _FRAME_PLACEHOLDER and pending is not None:
                camera, step, label_index = pending
                references.append(
                    _FrameReference(
                        parts,
                        label_index,
                        index,
                        trial_prefix,
                        camera,
                        step,
                    )
                )
                pending = None
    return tuple(references)


def _render_chat_transcript(
    transcript: list[object],
    frame_ctx: _FrameContext | None = None,
    *,
    trial_prefix: str = "",
    operator_messages: Sequence[dict[str, Any]] = (),
) -> str:
    """Render original chat messages as observation-led turns with one raw transcript each."""
    rendered, _residual = _render_chat_and_residual(
        transcript,
        frame_ctx,
        trial_prefix=trial_prefix,
        operator_messages=operator_messages,
    )
    return rendered


def _render_chat_and_residual(
    transcript: list[object],
    frame_ctx: _FrameContext | None,
    *,
    trial_prefix: str,
    operator_messages: Sequence[dict[str, Any]],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Render turns and return only structured feedback that could not be placed."""
    prefix = frame_ctx.trial_prefix if frame_ctx is not None else trial_prefix
    turns = _group_turns(transcript, _frame_references(transcript, prefix))
    _place_feedback(turns, operator_messages)
    rendered = (
        '<div class="conversation">'
        + "".join(_render_turn(turn, frame_ctx) for turn in turns)
        + "</div>"
    )
    return rendered, _unplaced_feedback(turns, operator_messages)


def _group_turns(transcript: list[object], references: Sequence[_FrameReference]) -> list[_Turn]:
    """Group exact message objects at list-shaped user observations."""
    turns: list[_Turn] = []
    current: _Turn | None = None
    for message in transcript:
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), list)
        ):
            parts = cast(list[object], message.get("content"))
            current = _Turn(
                messages=[message],
                references=tuple(reference for reference in references if reference.parts is parts),
                feedback=[],
            )
            turns.append(current)
            continue
        if current is None:
            current = _Turn(messages=[], references=(), preamble=True, feedback=[])
            turns.append(current)
        current.messages.append(message)
    return turns


def _place_feedback(turns: Sequence[_Turn], messages: Sequence[dict[str, Any]]) -> None:
    """Attach each structured message to the latest eligible frame-step turn."""
    stepped = [
        (turn.references[0].step, index, turn)
        for index, turn in enumerate(turns)
        if not turn.preamble and turn.references
    ]
    for message in messages:
        raw_t = message.get("t")
        if not _is_number(raw_t):
            continue
        message_step = cast(int | float, raw_t)
        candidates = [candidate for candidate in stepped if candidate[0] <= message_step]
        if not candidates:
            continue
        turn = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]
        cast(list[dict[str, Any]], turn.feedback).append(message)


def _unplaced_feedback(
    turns: Sequence[_Turn], messages: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Return structured messages not attached by identity to any turn."""
    placed = {id(message) for turn in turns for message in (turn.feedback or [])}
    return tuple(message for message in messages if id(message) not in placed)


def _render_turn(turn: _Turn, frame_ctx: _FrameContext | None) -> str:
    """Render one human-readable turn followed by its complete raw exchange."""
    header = (
        ""
        if turn.preamble or not turn.references
        else f'<div class="turn-step">step {turn.references[0].step}</div>'
    )
    frames = _render_turn_frames(turn.references, frame_ctx)
    feedback = _render_feedback_chips(turn.feedback or [])
    visible = "".join(_render_visible_message(message) for message in turn.messages)
    pov = _render_pov(turn.messages)
    return f'<section class="turn">{header}{frames}{feedback}{visible}{pov}</section>'


def _render_turn_frames(
    references: Sequence[_FrameReference], frame_ctx: _FrameContext | None
) -> str:
    """Render correlated frames in reference order with camera-only captions."""
    if frame_ctx is None:
        return ""
    cells: list[str] = []
    for reference in references:
        image = _frame_image(frame_ctx, reference.camera, reference.step)
        if image is None:
            continue
        cells.append(
            '<figure class="frame-cell">'
            f"<figcaption>{_escape(reference.camera)}</figcaption>{image}</figure>"
        )
    return "" if not cells else f'<div class="frame-row">{"".join(cells)}</div>'


def _render_feedback_chips(messages: Sequence[dict[str, Any]]) -> str:
    """Render structured operator messages with their source labels."""
    if not messages:
        return ""
    chips = "".join(
        '<span class="feedback-chip"><span class="feedback-source">'
        f"{_escape(message.get('source', 'operator'))}</span>"
        f"{_escape(message.get('text', ''))}</span>"
        for message in messages
    )
    return f'<div class="feedback-row">{chips}</div>'


def _render_visible_message(raw_message: object) -> str:
    """Render only human-layer prose and pretty calls from one original message."""
    if not isinstance(raw_message, dict):
        return ""
    role = str(raw_message.get("role", "unknown"))
    content_value = raw_message.get("content")
    content = _chat_content(content_value)
    if role == "user" and isinstance(content_value, list):
        return ""
    if role == "tool":
        return ""
    body = "" if content is None else f'<div class="content">{_escape(content)}</div>'
    calls = ""
    tool_calls = raw_message.get("tool_calls")
    if isinstance(tool_calls, list):
        calls = "".join(_render_tool_call(raw_call) for raw_call in tool_calls)
    if role == "system":
        return f'<details class="system-message"><summary>system</summary>{body}</details>'
    role_class = role if role in {"user", "assistant"} else "unknown"
    return (
        f'<div class="message {role_class}"><div class="role">{_escape(role)}</div>'
        f"{body}{calls}</div>"
    )


def _render_pov(messages: Sequence[object]) -> str:
    """Render one native dropdown containing the turn's verbatim wire-level text."""
    raw = "".join(_render_raw_message(message) for message in messages)
    return f'<details class="raw-transcript"><summary>Raw transcript</summary>{raw}</details>'


def _render_raw_message(raw_message: object) -> str:
    """Render one original chat message without filtering its text or tool results."""
    if not isinstance(raw_message, dict):
        return ""
    role = str(raw_message.get("role", "unknown"))
    content_value = raw_message.get("content")
    content: str | None
    if isinstance(content_value, list):
        # Non-text parts leave a marker so nothing silently vanishes from the
        # page: the raw transcript is the turn's complete exchange.
        content = "\n".join(
            str(part.get("text", ""))
            if isinstance(part, dict) and part.get("type") == "text"
            else "[image]"
            for part in content_value
        )
    else:
        content = _chat_content(content_value)
    body = "" if content is None else f'<div class="content">{_escape(content)}</div>'
    calls = ""
    tool_calls = raw_message.get("tool_calls")
    if isinstance(tool_calls, list):
        for raw_call in tool_calls:
            if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
                continue
            function = cast(dict[str, object], raw_call["function"])
            calls += _raw_tool_call(
                str(function.get("name", "unknown")), function.get("arguments", "")
            )
    role_class = role if role in {"user", "assistant", "tool"} else "unknown"
    return (
        f'<div class="message {role_class}"><div class="role">{_escape(role)}</div>'
        f"{body}{calls}</div>"
    )


def _elide_json_values(value: Any) -> Any:
    """Recursively bound long JSON string values before serialization."""
    if isinstance(value, str):
        if len(value) <= _JSON_STRING_LIMIT:
            return value
        omitted = len(value) - _JSON_STRING_LIMIT
        return value[:_JSON_STRING_LIMIT] + f"[... {omitted} chars truncated]"
    if isinstance(value, list):
        return [_elide_json_values(item) for item in value]
    if isinstance(value, tuple):
        return [_elide_json_values(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _elide_json_values(item) for key, item in value.items()}
    return value


def _render_transcript(
    transcript: object,
    frame_ctx: _FrameContext | None = None,
    *,
    trial_prefix: str = "",
) -> str:
    """Render chat-shaped records conversationally and all others as bounded JSON."""
    if _is_chat_transcript(transcript):
        return _render_chat_transcript(
            cast(list[object], transcript), frame_ctx, trial_prefix=trial_prefix
        )
    # Escaping happens on the dumped text below, so raw non-ASCII is safe and
    # far more readable than \uXXXX escapes.
    dumped = json.dumps(
        _elide_json_values(transcript), indent=2, sort_keys=True, ensure_ascii=False
    )
    return f"<pre>{_escape(dumped)}</pre>"


def _score_chips(values: Mapping[str, float], *, prefix: str = "") -> str:
    """Render sorted score values as compact escaped chips."""
    return "".join(
        f'<span class="score-chip">{_escape(prefix + name)}={_escape(_number(value))}</span>'
        for name, value in sorted(values.items())
    )


def _trial_frame_context(
    frame_ctx: _FrameContext | None,
    scene_id: str,
    trial: int,
    rendered: list[tuple[str, int]] | None = None,
) -> _FrameContext | None:
    """Specialize a document frame context to one scene trial."""
    if frame_ctx is None:
        return None
    return _FrameContext(
        frame_ctx.frames_dir,
        _safe(f"{scene_id}-e{trial}"),
        frame_ctx.budget,
        rendered,
    )


def _document_frame_references(log: EvalLog) -> tuple[_FrameReference, ...]:
    """Enumerate correlated references across scenes and trials in document order."""
    references: list[_FrameReference] = []
    for scene in log.samples:
        for trial, transcript in enumerate(scene.policy_transcripts):
            references.extend(_frame_references(transcript, _safe(f"{scene.scene_id}-e{trial}")))
    return tuple(references)


def _prime_live_frame_cache(log: EvalLog, frame_ctx: _FrameContext) -> None:
    """Allocate the shared live budget to valid frames from newest to oldest."""
    budget = frame_ctx.budget
    cache = cast(dict[tuple[str, str, int], str], budget.cache)
    attempted: set[tuple[str, str, int]] = set()
    for reference in reversed(_document_frame_references(log)):
        key = (reference.trial_prefix, reference.camera, reference.step)
        if key in attempted:
            continue
        attempted.add(key)
        trial_ctx = _FrameContext(frame_ctx.frames_dir, reference.trial_prefix, budget)
        array = _load_frame(trial_ctx, reference.camera, reference.step)
        if array is None:
            continue
        source = png_data_url(array)
        payload_size = len(source.partition(",")[2])
        if budget.limit and budget.payload_bytes + payload_size > budget.limit:
            budget.truncated = True
            break
        budget.payload_bytes += payload_size
        budget.embedded += 1
        cache[key] = source


def _render_trial_transcript(
    transcript: object,
    frame_ctx: _FrameContext | None,
    scene_id: str,
    trial: int,
    operator_messages: Sequence[dict[str, Any]],
) -> tuple[str, tuple[dict[str, Any], ...], list[tuple[str, int]]]:
    """Render one trial and return its residual feedback and embedded frame keys."""
    trial_prefix = _safe(f"{scene_id}-e{trial}")
    rendered_frames: list[tuple[str, int]] = []
    trial_ctx = _trial_frame_context(frame_ctx, scene_id, trial, rendered_frames)
    if _is_chat_transcript(transcript):
        document, residual = _render_chat_and_residual(
            cast(list[object], transcript),
            trial_ctx,
            trial_prefix=trial_prefix,
            operator_messages=operator_messages,
        )
        return document, residual, rendered_frames
    return (
        _render_transcript(transcript, trial_ctx, trial_prefix=trial_prefix),
        tuple(operator_messages),
        rendered_frames,
    )


def _load_wire_rows(
    scene: SceneResult, trial: int, log_path: Path | None
) -> tuple[list[dict[str, Any]], Path] | None:
    """Load one guarded wire sidecar and derive its run-scoped blob directory."""
    if log_path is None or trial >= len(scene.trial_metadata):
        return None
    target = resolve_log_pointer(log_path, scene.trial_metadata[trial].get("wire_capture"))
    if target is None:
        return None
    loaded = read_jsonl_prefix(target)
    if loaded is None:
        return None
    blob_dir = derive_blob_dir(log_path, target)
    if blob_dir is None:
        return None
    return loaded, blob_dir


def _wire_blob_tokens(value: object) -> list[str]:
    """Return every blob token suffix found recursively in a captured value."""
    if isinstance(value, str):
        return [match.group(1) for match in _BLOB_SENTINEL_RE.finditer(value)]
    if isinstance(value, list):
        return [token for item in value for token in _wire_blob_tokens(item)]
    if isinstance(value, dict):
        return [token for item in value.values() for token in _wire_blob_tokens(item)]
    return []


def _render_wire_blob(token: str, context: _WireContext) -> str:
    """Render one validated blob reference as an embed, link, elision, or error."""
    if _BLOB_SHA_RE.fullmatch(token) is None:
        return (
            f'<span class="wire-broken">{_escape(f"[broken blob reference: $blob:{token}]")}</span>'
        )
    if context.elide_media:
        return '<span class="wire-placeholder">[blob elided: live page]</span>'
    anchor = f"wire-{_safe(context.trial_id)}-{token[:12]}"
    if token in context.emitted:
        return f'<a href="#{_escape(anchor)}">[blob {_escape(token[:12])}]</a>'
    if token in context.denied:
        return '<span class="wire-placeholder">[blob elided: media budget]</span>'

    # ``token`` has passed the exact lowercase-sha guard above.  Only now may
    # foreign JSONL text participate in a filesystem path.
    path = context.blob_dir / f"{token}.png"
    try:
        raw = path.read_bytes()
    except OSError:
        return (
            f'<span class="wire-broken">{_escape(f"[broken blob reference: $blob:{token}]")}</span>'
        )
    encoded = base64.b64encode(raw).decode("ascii")
    budget = context.budget
    if budget.truncated or (budget.limit and budget.payload_bytes + len(encoded) > budget.limit):
        budget.truncated = True
        context.denied.add(token)
        return '<span class="wire-placeholder">[blob elided: media budget]</span>'
    budget.payload_bytes += len(encoded)
    budget.embedded += 1
    context.emitted.add(token)
    return (
        f'<span class="wire-blob" id="{_escape(anchor)}">'
        '<img class="frame" loading="lazy" '
        f'alt="wire blob {_escape(token[:12])}" src="data:image/png;base64,{encoded}"></span>'
    )


def _render_wire_value(value: object, context: _WireContext) -> str:
    """Render captured JSON verbatim enough for forensic keys, then its blob media."""
    dumped = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    media = "".join(
        f'<div class="wire-media">{_render_wire_blob(token, context)}</div>'
        for token in _wire_blob_tokens(value)
    )
    return f"<pre>{_escape(dumped)}</pre>{media}"


def _wire_messages(request: dict[str, Any]) -> list[object]:
    """Return the messages-shaped sequence used by any supported wire."""
    messages = request.get("messages", request.get("input"))
    return cast(list[object], messages) if isinstance(messages, list) else []


def _wire_effort(request: dict[str, Any]) -> object:
    """Extract the supported top-level or nested effort parameter."""
    if "reasoning_effort" in request:
        return request["reasoning_effort"]
    reasoning = request.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        return reasoning["effort"]
    output_config = request.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        return output_config["effort"]
    return "n/a"


def _wire_request(row: dict[str, Any]) -> dict[str, Any]:
    """Return a captured request mapping, degrading malformed rows to empty."""
    request = row.get("request")
    return cast(dict[str, Any], request) if isinstance(request, dict) else {}


def _render_wire_call(
    row: dict[str, Any],
    previous_messages: list[object] | None,
    first_tools: object,
    first_system: object,
    context: _WireContext,
) -> str:
    """Render one attempt row with static-field change notes and message deltas."""
    request = _wire_request(row)
    messages = _wire_messages(request)
    if previous_messages is None:
        changed: list[tuple[int, object]] = []
        new_messages = messages
    else:
        changed = [
            (index, message)
            for index, message in enumerate(messages[: len(previous_messages)])
            if message != previous_messages[index]
        ]
        new_messages = messages[len(previous_messages) :]

    notes: list[str] = []
    if request.get("tools", _MISSING) != first_tools:
        notes.append("tools changed from the trial's first call")
    if request.get("system", _MISSING) != first_system:
        notes.append("system changed from the trial's first call")
    notes.extend(f"message {index} changed as sent" for index, _ in changed)
    changes = "".join(f'<p class="wire-change">{_escape(note)}</p>' for note in notes)
    changed_messages = "".join(
        (
            f'<div class="wire-change">message {index} current as-sent form</div>'
            f"{_render_wire_value(message, context)}"
        )
        for index, message in changed
    )
    additions = "".join(_render_wire_value(message, context) for message in new_messages)
    if not additions and not changed_messages:
        additions = '<p class="wire-change">no new messages</p>'

    tools = request.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0
    params = "".join(
        (
            _definition("model", request.get("model", "n/a")),
            _definition("effort", _wire_effort(request)),
            _definition("temperature", request.get("temperature", "n/a")),
            _definition("tool count", tool_count),
        )
    )
    call = row.get("call", "?")
    attempt = row.get("attempt", "?")
    endpoint = row.get("endpoint", "?")
    status = row.get("status")
    duration = row.get("duration_s")
    shown_status = "null" if status is None else status
    shown_duration = (
        f"{_number(duration)} s"
        if isinstance(duration, (int, float)) and not isinstance(duration, bool)
        else "n/a"
    )
    summary = (
        f"call {_escape(call)} attempt {_escape(attempt)} · {_escape(endpoint)} · "
        f"status {_escape(shown_status)} · {_escape(shown_duration)}"
    )
    return (
        f'<details class="wire-call"><summary>{summary}</summary>'
        f"<dl>{params}</dl>{changes}{changed_messages}{additions}</details>"
    )


def _render_trial_wire(
    scene: SceneResult,
    trial: int,
    log_path: Path | None,
    budget: _FrameBudget,
    *,
    elide_media: bool = False,
) -> str:
    """Render one trial's guarded wire capture, or nothing when unavailable."""
    loaded = _load_wire_rows(scene, trial, log_path)
    if loaded is None:
        return ""
    rows, blob_dir = loaded
    first_request = _wire_request(rows[0])
    first_tools = first_request.get("tools", _MISSING)
    first_system = first_request.get("system", _MISSING)
    context = _WireContext(
        blob_dir=blob_dir,
        trial_id=f"{scene.scene_id}-e{trial}",
        budget=budget,
        emitted=set(),
        denied=set(),
        elide_media=elide_media,
    )
    static = ""
    if first_tools is not _MISSING:
        static += (
            '<div class="wire-static"><strong>Tools (trial initial)</strong>'
            f"{_render_wire_value(first_tools, context)}</div>"
        )
    if first_system is not _MISSING:
        static += (
            '<div class="wire-static"><strong>System (trial initial)</strong>'
            f"{_render_wire_value(first_system, context)}</div>"
        )

    calls: list[str] = []
    previous: list[object] | None = None
    for row in rows:
        calls.append(_render_wire_call(row, previous, first_tools, first_system, context))
        previous = _wire_messages(_wire_request(row))
    return (
        '<details class="wire"><summary>'
        f"Trial {trial} Wire</summary>{static}{''.join(calls)}</details>"
    )


def _trial_camera_streams(frames_dir: Path, trial_prefix: str) -> dict[str, list[tuple[int, Path]]]:
    """Enumerate one trial's camera streams by stripping its known filename prefix."""
    streams: dict[str, list[tuple[int, Path]]] = {}
    marker = f"{trial_prefix}_"
    for path in sorted(frames_dir.glob(f"{trial_prefix}_*.npy")):
        remainder = path.name[len(marker) :]
        match = _CAMERA_FRAME_RE.fullmatch(remainder)
        if match is None:
            continue
        streams.setdefault(match.group(1), []).append((int(match.group(2)), path))
    for frames in streams.values():
        frames.sort()
    return dict(sorted(streams.items()))


def _warn_video_degrade(context: _VideoContext, reason: str) -> None:
    """Write at most one encoder-degrade warning for a complete render pass."""
    if context.budget.warned:
        return
    print(f"warning: report video unavailable ({reason}); using flipbook", file=sys.stderr)
    context.budget.warned = True


def _render_trial_media(
    frames_dir: Path | None,
    trial_prefix: str,
    rendered_frames: Sequence[tuple[str, int]],
    context: _VideoContext,
) -> str:
    """Render one composite MP4 when eligible and flipbook panels otherwise."""
    flipbook: dict[str, list[int]] = {}
    for camera, step in rendered_frames:
        flipbook.setdefault(camera, []).append(step)
    display_names = {_safe(camera): camera for camera in flipbook}

    available_streams = (
        _trial_camera_streams(frames_dir, trial_prefix)
        if context.enabled and frames_dir is not None
        else {}
    )
    streams = available_streams if context.ffmpeg is not None else {}
    if context.enabled and context.ffmpeg is None and (available_streams or flipbook):
        _warn_video_degrade(context, "ffmpeg not found")
    rendered_order: dict[str, int] = {}
    for index, (camera, _step) in enumerate(rendered_frames):
        rendered_order.setdefault(camera, index)
    default_order = len(rendered_frames)
    ordered_streams = sorted(
        streams.items(),
        key=lambda item: (
            rendered_order.get(display_names.get(item[0], item[0]), default_order),
            item[0],
        ),
    )
    fallback_cameras = tuple(
        dict.fromkeys(
            [display_names.get(key, key) for key, _frames in streams.items()] + list(flipbook)
        )
    )
    if ordered_streams and not context.budget.truncated:
        from inspect_robots._video import _encode_composite_mp4

        ffmpeg = cast(str, context.ffmpeg)
        encoded = _encode_composite_mp4(ordered_streams, context.fps, ffmpeg)
        if encoded is None:
            _warn_video_degrade(context, f"encode failed for {trial_prefix} composite")
        else:
            video, survivors = encoded
            payload = base64.b64encode(video).decode("ascii")
            if not (
                context.budget.limit
                and context.budget.encoded + len(payload) > context.budget.limit
            ):
                context.budget.encoded += len(payload)
                camera_order = " · ".join(display_names.get(key, key) for key in survivors)
                return (
                    f'<div class="run-media" data-trial="{_escape(trial_prefix)}">'
                    '<div class="run-media-head">Run video</div>'
                    f'<div class="camera-order">{_escape(camera_order)}</div>'
                    '<div class="camera-panel video-panel">'
                    '<video controls muted loop preload="metadata" '
                    f'src="data:video/mp4;base64,{payload}"></video></div></div>'
                )
            context.budget.truncated = True

    # Sticky, first-wins: once one trial composite overflows, later trials
    # degrade without paying an encode for a discarded payload.
    reason = (
        "video budget"
        if context.budget.truncated
        else "no ffmpeg"
        if context.enabled and context.ffmpeg is None
        else None
    )
    return _render_flipbook_media(trial_prefix, flipbook, fallback_cameras, reason)


def _render_flipbook_media(
    trial_prefix: str,
    flipbook: Mapping[str, Sequence[int]],
    cameras: Sequence[str],
    reason: str | None,
) -> str:
    """Render the existing per-camera tabbed flipbook fallback."""
    panels = [
        (camera, panel)
        for camera in cameras
        if (panel := _flipbook_panel(camera, flipbook.get(camera, ())))
    ]
    if not panels:
        return ""
    tabs = "".join(
        f'<button type="button" class="camera-tab{" active" if index == 0 else ""}" '
        f'data-camera-tab="{_escape(camera)}">{_escape(camera)}</button>'
        for index, (camera, _panel) in enumerate(panels)
    )
    bodies = "".join(
        f'<div class="camera-panel flipbook-panel" data-camera-panel="{_escape(camera)}"'
        f"{'' if index == 0 else ' hidden'}>{panel}</div>"
        for index, (camera, panel) in enumerate(panels)
    )
    reason_chip = "" if reason is None else f'<span class="chip">{_escape(reason)}</span>'
    return (
        f'<div class="run-media" data-trial="{_escape(trial_prefix)}">'
        f'<div class="run-media-head">Run video{reason_chip}</div>'
        f'<div class="camera-tabs">{tabs}</div>{bodies}</div>'
    )


def _flipbook_panel(camera: str, steps: Sequence[int]) -> str:
    """Render controls for a camera only when at least two source frames exist."""
    if len(steps) < 2:
        return ""
    return (
        f'<img class="flipbook-player" alt="camera {_escape(camera)} flipbook">'
        '<div class="flipbook-controls"><button type="button" data-play-toggle>Pause</button>'
        '<input type="range" min="0" value="0" data-scrubber '
        f'max="{len(steps) - 1}" aria-label="{_escape(camera)} frame">'
        f'<span class="step-label" data-step-label>step {steps[0]}</span></div>'
    )


def _scene_section(
    scene: SceneResult,
    *,
    open_transcript: bool,
    budget: _FrameBudget,
    video_context: _VideoContext,
    log_path: Path | None,
    frames_dir: Path | None,
    frame_ctx: _FrameContext | None = None,
    scores_pending: bool = False,
    wire_media_elided: bool = False,
    log_started: bool = False,
) -> str:
    """Render one complete scene card and its available trial transcripts."""
    instruction = (
        ""
        if scene.instruction is None
        else f'<p class="instruction">{_escape(scene.instruction)}</p>'
    )
    error = "" if scene.error is None else f'<p class="error">{_escape(scene.error)}</p>'

    reduced = _score_chips(scene.reduced)
    reduced_block = (
        "" if not reduced else f'<h3>Reduced scores</h3><div class="score-row">{reduced}</div>'
    )
    epoch_chips = "".join(
        (
            f'<span class="score-chip">{_escape(f"trial {index} pending")}</span>'
            if scores_pending and not epoch
            else _score_chips(epoch, prefix=f"trial {index} ")
        )
        for index, epoch in enumerate(scene.epochs)
    )
    epoch_block = (
        ""
        if not epoch_chips
        else f'<h3>Trial scores</h3><div class="score-row">{epoch_chips}</div>'
    )
    reasons = "".join(
        '<span class="score-chip">n/a</span>'
        if reason is None
        else f'<span class="score-chip">{_escape(reason)}</span>'
        for reason in scene.termination_reasons
    )
    reasons_block = (
        "" if not reasons else f'<h3>Termination reasons</h3><div class="score-row">{reasons}</div>'
    )
    judgements = "".join(
        '<span class="score-chip">n/a</span>'
        if judgement is None
        else f'<span class="score-chip">{_escape(judgement)}</span>'
        for judgement in scene.operator_judgements
    )
    judgements_block = (
        ""
        if not judgements
        else f'<h3>Operator judgements</h3><div class="score-row">{judgements}</div>'
    )
    notes = "".join(
        (
            f'<div class="grader-note"><span class="note-label">trial {index}</span>'
            f"{_escape(note)}</div>"
        )
        for index, note in enumerate(scene.operator_notes)
        if note is not None
    )
    # Unlike the chip rows above, an all-``None`` tuple emits nothing at all:
    # ``notes`` is empty exactly when no trial carried a note, and most will not.
    notes_block = "" if not notes else f"<h3>Grader notes</h3>{notes}"
    transcript_blocks: list[str] = []
    residual: list[tuple[int, dict[str, Any]]] = []
    for trial, transcript in enumerate(scene.policy_transcripts):
        messages = scene.operator_messages[trial] if trial < len(scene.operator_messages) else ()
        if transcript is None:
            residual.extend((trial, message) for message in messages)
            continue
        rendered, unplaced, rendered_frames = _render_trial_transcript(
            transcript,
            frame_ctx,
            scene.scene_id,
            trial,
            messages,
        )
        residual.extend((trial, message) for message in unplaced)
        trial_prefix = _safe(f"{scene.scene_id}-e{trial}")
        media = _render_trial_media(frames_dir, trial_prefix, rendered_frames, video_context)
        transcript_blocks.append(
            f'<details class="transcript"{" open" if open_transcript else ""}>'
            f"<summary>Trial {trial} transcript</summary>{media}{rendered}</details>"
        )
    for trial in range(len(scene.policy_transcripts), len(scene.operator_messages)):
        residual.extend((trial, message) for message in scene.operator_messages[trial])
    transcripts = "".join(transcript_blocks)
    feedback = "".join(
        (
            f'<div class="grader-note"><span class="note-label">trial {trial}, '
            f"step {_escape(message.get('t', '?'))}, "
            f"{_escape(message.get('source', 'operator'))}</span>"
            f"{_escape(message.get('text', ''))}</div>"
        )
        for trial, message in residual
    )
    feedback_block = "" if not feedback else f"<h3>Operator feedback</h3>{feedback}"
    wires = "".join(
        _render_trial_wire(
            scene,
            trial,
            log_path,
            budget,
            elide_media=wire_media_elided,
        )
        for trial in range(len(scene.epochs))
    )
    scene_status = scene.status
    has_live_slot = any(isinstance(metadata.get("live"), dict) for metadata in scene.trial_metadata)
    if log_started and has_live_slot:
        scene_status = "started"
    return (
        '<section class="scene">'
        f'<div class="scene-head"><h2>{_escape(scene.scene_id)}</h2>'
        f"{_status_badge(scene_status)}</div>{instruction}{error}{reduced_block}{epoch_block}"
        f"{reasons_block}{judgements_block}{notes_block}{transcripts}{feedback_block}"
        f"{wires}</section>"
    )


def render_html(
    log: EvalLog,
    *,
    title: str,
    log_path: Path | None = None,
    frames_dir: Path | None = None,
    frames_budget_bytes: int = 50_000_000,
    live_frames_budget_bytes: int | None = None,
    wire_media_elided: bool = False,
    refresh_seconds: int | None = None,
    no_video: bool = False,
    serve_pass: bool = False,
    video_budget_bytes: int = _VIDEO_BUDGET_BYTES,
) -> str:
    """Return one self-contained HTML document describing the complete evaluation log."""
    git = (
        'git <span class="chip">unknown</span>'
        if log.eval.git_commit is None
        else f"git {_escape(log.eval.git_commit)}"
    )
    definitions = [
        _definition("policy", log.eval.policy),
        _definition("embodiment", log.eval.embodiment),
    ]
    definitions.extend(
        _definition(key, _value(value)) for key, value in sorted(log.eval.policy_config.items())
    )
    if log.eval.seed is not None:
        definitions.append(_definition("seed", log.eval.seed))
    if log.eval.max_seconds is not None:
        definitions.append(_definition("max seconds", log.eval.max_seconds))
        if log.eval.max_steps is not None:
            definitions.append(_definition("resolved max steps", log.eval.max_steps))
    elif log.eval.max_steps is not None:
        definitions.append(_definition("max steps", log.eval.max_steps))
    if log.stats.mean_inference_latency_s is not None:
        definitions.append(
            _definition(
                "mean inference latency", f"{_number(log.stats.mean_inference_latency_s)} s"
            )
        )
    definitions.extend(
        [
            _definition("duration", f"{_number(log.stats.duration_s)} s"),
            _definition("total steps", log.stats.total_steps),
        ]
    )

    metric_tiles = "".join(
        '<div class="stat">'
        f'<div class="stat-name">{_escape(name)}</div>'
        f'<div class="stat-value">{_escape(_number(value))}</div></div>'
        for name, value in sorted(log.results.metrics.items())
    )
    metric_tiles += "".join(
        '<div class="stat">'
        f'<div class="stat-name">{label}</div>'
        f'<div class="stat-value">{_escape(value)}</div></div>'
        for label, value in (
            ("scenes", log.results.total_scenes),
            ("trials", log.results.total_trials),
            ("errored", log.results.errored_trials),
        )
    )

    transcript_count = sum(
        transcript is not None for scene in log.samples for transcript in scene.policy_transcripts
    )
    effective_budget = frames_budget_bytes
    if live_frames_budget_bytes is not None:
        finite_limits = [
            limit for limit in (frames_budget_bytes, live_frames_budget_bytes) if limit != 0
        ]
        effective_budget = min(finite_limits) if finite_limits else 0
    budget = _FrameBudget(
        limit=effective_budget,
        cache={} if live_frames_budget_bytes is not None else None,
    )
    frame_ctx = None if frames_dir is None else _FrameContext(frames_dir, "", budget)
    if frame_ctx is not None and live_frames_budget_bytes is not None:
        _prime_live_frame_cache(log, frame_ctx)
    video_eligible = (
        frame_ctx is not None and log.status != "started" and not serve_pass and not no_video
    )
    ffmpeg = shutil.which("ffmpeg") if video_eligible else None
    if video_eligible:
        from inspect_robots._video import default_fps

        fps, _source = default_fps(log.eval.embodiment_info)
    else:
        fps = 0.0
    video_context = _VideoContext(
        enabled=video_eligible,
        ffmpeg=ffmpeg,
        fps=fps,
        budget=_VideoBudget(limit=video_budget_bytes),
    )
    scenes = "".join(
        _scene_section(
            scene,
            open_transcript=transcript_count == 1,
            budget=budget,
            video_context=video_context,
            log_path=log_path,
            frames_dir=frames_dir,
            frame_ctx=frame_ctx,
            scores_pending=log.status == "started",
            wire_media_elided=wire_media_elided,
            log_started=log.status == "started",
        )
        for scene in log.samples
    )
    no_transcripts = (
        '<p class="none">no policy transcripts recorded</p>' if transcript_count == 0 else ""
    )
    frames_chip = (
        ""
        if not budget.truncated
        else '<span class="chip">embedded media truncated at '
        f"{effective_budget / 1_000_000:g} MB ({budget.embedded} embedded)</span>"
    )
    meta_tail = (
        f"<span>inspect-robots {_escape(log.eval.inspect_robots_version)}</span>"
        f"<span>{git}</span>{frames_chip}"
    )
    refresh = (
        ""
        if refresh_seconds is None
        else f'<meta http-equiv="refresh" content="{_escape(refresh_seconds)}">\n'
    )
    last_update: str | None = None
    for scene in reversed(log.samples):
        for metadata in reversed(scene.trial_metadata):
            live = metadata.get("live")
            if isinstance(live, dict) and isinstance(live.get("updated_at"), str):
                updated_at = live["updated_at"]
                time_tail = updated_at.partition("T")[2]
                last_update = time_tail[:8] if time_tail else updated_at
                break
        if last_update is not None:
            break
    running_banner = ""
    if refresh_seconds is not None:
        update = "" if last_update is None else f" · last update {_escape(last_update)}"
        running_banner = (
            '<div class="running-banner">RUNNING — refreshes every '
            f"{_escape(refresh_seconds)}s{update}</div>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
{refresh}<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_STYLES}</style>
<script>{_FRAME_CLICK_SCRIPT}</script>
<script>{_FLIPBOOK_SCRIPT}</script>
</head>
<body>
<header><div class="header-inner">
  <div class="header-top"><h1>{_escape(log.eval.task)}</h1>{_status_badge(log.status)}</div>
  <div class="meta"><span>{_escape(log.eval.created)}</span>
  {meta_tail}</div>
  {running_banner}
</div></header>
<main>
  <section class="spec-strip"><dl>{"".join(definitions)}</dl></section>
  <section class="metrics">{metric_tiles}</section>
  {scenes}
  {no_transcripts}
</main>
</body>
</html>
"""
    return document
