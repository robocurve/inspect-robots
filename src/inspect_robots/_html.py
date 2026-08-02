"""Render evaluation logs as dependency-free HTML report documents."""

from __future__ import annotations

import base64
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from inspect_robots._pngenc import png_data_url
from inspect_robots._pointers import derive_blob_dir, read_jsonl_prefix, resolve_log_pointer
from inspect_robots._video import default_fps
from inspect_robots.frames import _safe
from inspect_robots.log import EvalLog, SceneResult

_STATUS_DISPLAY = {"success": "completed"}
_JSON_STRING_LIMIT = 2048
# \d{1,12}, not \d+: a hostile multi-thousand-digit "step" must miss the
# regex and degrade rather than trip int()'s conversion-length limit.
_FRAME_LABEL_RE = re.compile(r"camera '(?P<name>.*)' \(step (?P<step>\d{1,12})\):")
_FRAME_PLACEHOLDER = "[image omitted: streamed camera frame]"
_FRAME_MAX_SIDE = 448
_BLOB_SENTINEL_RE = re.compile(r"\$blob:([^\s]+)")
_BLOB_SHA_RE = re.compile(r"[0-9a-f]{64}")
_MISSING = object()

_VIEW_SCRIPT = """document.addEventListener('click', (event) => {
  const cell = event.target.closest('.frame-cell');
  if (cell) cell.classList.toggle('wide');
});
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".trial.has-player").forEach(trial => {
    const player = trial.querySelector(".player");
    const videos = Array.from(player.querySelectorAll("video"));
    const playpause = player.querySelector(".playpause");
    const scrub = player.querySelector(".scrub");
    const clock = player.querySelector(".clock");
    const follow = player.querySelector(".follow");
    const rail = trial.querySelector(".transcript");
    const fps = Number(player.dataset.fps);
    let master = null, settled = 0, frameRequest = null, autoScrolling = false;

    function setFollow(on) {
      follow.classList.toggle("on", on);
      follow.textContent = on ? "Follow" : "Follow off";
    }
    function disableFollow() {
      if (!autoScrolling) setFollow(false);
    }
    function update(time) {
      if (!master) return;
      const step = Math.round(time * fps);
      scrub.value = String(Math.round(1000 * time / master.duration));
      clock.textContent = `step ${step} · ${time.toFixed(1)}s`;
      if (!rail) return;
      let live = null;
      rail.querySelectorAll(".message[data-step]").forEach(message => {
        if (Number(message.dataset.step) <= step) live = message;
      });
      rail.querySelectorAll(".message.live").forEach(message => {
        if (message !== live) message.classList.remove("live");
      });
      if (live) {
        live.classList.add("live");
        if (follow.classList.contains("on")) {
          autoScrolling = true;
          rail.scrollTop = live.offsetTop - rail.offsetTop - 16;
          requestAnimationFrame(() => { autoScrolling = false; });
        }
      }
    }
    function seek(time) {
      videos.forEach(video => {
        if (video.readyState >= 1 && Number.isFinite(video.duration)) {
          video.currentTime = Math.min(time, video.duration);
        }
      });
      update(Math.min(time, master.duration));
    }
    function animate() {
      update(master.currentTime);
      if (!master.paused && !master.ended) {
        frameRequest = requestAnimationFrame(animate);
      }
    }
    function pauseAll() {
      videos.forEach(video => video.pause());
      playpause.textContent = "Play";
      if (frameRequest !== null) cancelAnimationFrame(frameRequest);
      frameRequest = null;
      update(master.currentTime);
    }
    function elect() {
      const playable = videos.filter(video => video.dataset.playable === "yes");
      if (!playable.length) return;
      master = playable.reduce((longest, video) =>
        video.duration > longest.duration ? video : longest
      );
      playpause.disabled = false;
      scrub.disabled = false;
      follow.disabled = false;
      master.addEventListener("ended", pauseAll);
      update(0);
    }
    function settle(video, playable) {
      if (video.dataset.settled) return;
      video.dataset.settled = "yes";
      if (playable && Number.isFinite(video.duration)) video.dataset.playable = "yes";
      settled += 1;
      if (settled === videos.length) elect();
    }
    videos.forEach(video => {
      if (video.readyState >= 1) settle(video, true);
      else {
        video.addEventListener("loadedmetadata", () => settle(video, true), { once: true });
        video.addEventListener("error", () => settle(video, false), { once: true });
      }
    });
    playpause.addEventListener("click", () => {
      if (!master) return;
      if (!master.paused && !master.ended) {
        pauseAll();
        return;
      }
      seek(master.ended ? 0 : master.currentTime);
      videos.forEach(video => {
        const started = video.play();
        if (started) started.catch(() => {});
      });
      playpause.textContent = "Pause";
      frameRequest = requestAnimationFrame(animate);
    });
    scrub.addEventListener("input", () => {
      if (master) seek(Number(scrub.value) * master.duration / 1000);
    });
    follow.addEventListener("click", () => setFollow(!follow.classList.contains("on")));
    if (rail) {
      rail.addEventListener("scroll", disableFollow);
      rail.addEventListener("wheel", () => setFollow(false));
      rail.addEventListener("touchstart", () => setFollow(false));
      rail.addEventListener("click", event => {
        const message = event.target.closest(".message[data-step]");
        if (!message || event.target.closest(".frame-cell") || event.target.closest("a")) return;
        if (getSelection().toString() || !master) return;
        seek(Number(message.dataset.step) / fps);
      });
    }
  });
});"""


@dataclass
class _FrameBudget:
    """Shared mutable accounting for frame payloads in one document."""

    limit: int
    embedded: int = 0
    payload_bytes: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class _FrameContext:
    """The filesystem correlation state for one trial transcript."""

    frames_dir: Path
    trial_prefix: str
    budget: _FrameBudget


@dataclass
class _WireContext:
    """Track one trial's blob directory, budget, and emitted image anchors."""

    blob_dir: Path
    trial_id: str
    budget: _FrameBudget
    emitted: set[str]
    denied: set[str]


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
.status-error { color: var(--red); background: var(--red-bg); }
.status-cancelled { color: var(--grey); background: var(--grey-bg); }
.status-neutral { color: var(--neutral); background: var(--neutral-bg); }
.chip { color: var(--muted); background: var(--grey-bg); }
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
.trial { position: relative; }
.player {
  position: sticky; top: 0; z-index: 2; margin-top: 18px; padding: 12px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
}
.cams { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
.cam { margin: 0; min-width: 0; }
.cam figcaption { margin-bottom: 4px; color: var(--muted); font-size: 12px; }
.cam video { display: block; width: 100%; max-height: 42vh; background: #000; }
.controls { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.controls button { font: inherit; }
.scrub { min-width: 100px; flex: 1; }
.clock { min-width: 112px; color: var(--muted); font-variant-numeric: tabular-nums; }
.follow.on { font-weight: 700; }
.trial.has-player .transcript { max-height: 60vh; overflow-y: auto; }
.conversation { margin-top: 14px; }
.message {
  margin: 13px 0; padding: 2px 0 2px 13px; border-left: 3px solid var(--system);
  scroll-margin-top: 16px;
}
.message.live { background: var(--grey-bg); }
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


def _number(value: int | float) -> str:
    """Format numeric log values compactly before their interpolation boundary."""
    return f"{value:.4g}"


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
    """Render one tolerant OpenAI-style tool call, or nothing when malformed."""
    if not isinstance(raw_call, dict):
        return ""
    function = raw_call.get("function")
    if not isinstance(function, dict):
        return ""
    name = str(function.get("name", "unknown"))
    arguments = function.get("arguments", "")
    shown_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    notes = "".join(
        f'<div class="agent-note"><span class="note-label">agent note</span>{_escape(note)}</div>'
        for note in _agent_notes(name, arguments)
    )
    return f'{notes}<div class="call">{_escape(name)}({_escape(shown_arguments)})</div>'


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
    array = _load_frame(frame_ctx, name, step)
    if array is None:
        return None
    source = png_data_url(array)
    payload_size = len(source.partition(",")[2])
    budget = frame_ctx.budget
    if budget.limit and budget.payload_bytes + payload_size > budget.limit:
        budget.truncated = True
        return None
    budget.payload_bytes += payload_size
    budget.embedded += 1
    return (
        f'<img class="frame" loading="lazy" alt="camera {_escape(name)} step {step}" '
        f'src="{source}">'
    )


def _render_frame_parts(parts: list[object], frame_ctx: _FrameContext) -> str:
    """Render user content parts as text runs split only by successful frame embeds."""
    runs: list[str] = []
    buffered: list[str] = []
    pending: tuple[str, int, int] | None = None
    embedded = 0
    row_open = False
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            raw_text = part.get("text", "")
            part_text = str(raw_text)
            if isinstance(raw_text, str):
                label = _FRAME_LABEL_RE.fullmatch(raw_text)
                if label is not None:
                    label_index = len(buffered)
                    buffered.append(part_text)
                    pending = (
                        label.group("name"),
                        int(label.group("step")),
                        label_index,
                    )
                    continue
                elif raw_text == _FRAME_PLACEHOLDER and pending is not None:
                    name, step, label_index = pending
                    pending = None
                    image = _frame_image(frame_ctx, name, step)
                    if image is not None:
                        caption = buffered[label_index][:-1]
                        del buffered[label_index]
                        joined = "\n".join(buffered)
                        if joined:
                            if row_open:
                                runs.append("</div>")
                                row_open = False
                            runs.append(f'<div class="content">{_escape(joined)}</div>')
                        if not row_open:
                            runs.append('<div class="frame-row">')
                            row_open = True
                        runs.append(
                            '<figure class="frame-cell">'
                            f"<figcaption>{_escape(caption)}</figcaption>"
                            f"{image}</figure>"
                        )
                        buffered = []
                        embedded += 1
                        continue
            buffered.append(part_text)
        else:
            buffered.append("[image]")
    if embedded == 0 or buffered:
        joined = "\n".join(buffered)
        if embedded == 0 or joined:
            if row_open:
                runs.append("</div>")
                row_open = False
            runs.append(f'<div class="content">{_escape(joined)}</div>')
    if row_open:
        runs.append("</div>")
    return "".join(runs)


def _message_step(raw_message: object) -> int | None:
    """Return the minimum camera-label step in one user message, if any."""
    if not isinstance(raw_message, dict) or raw_message.get("role") != "user":
        return None
    content = raw_message.get("content")
    if not isinstance(content, list):
        return None
    steps: list[int] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str):
            continue
        match = _FRAME_LABEL_RE.fullmatch(text)
        if match is not None:
            steps.append(int(match.group("step")))
    return min(steps) if steps else None


def _render_message(
    raw_message: object,
    frame_ctx: _FrameContext | None = None,
    *,
    anchor: int = 0,
) -> str:
    """Render one tolerant chat message without trusting its role or content."""
    if not isinstance(raw_message, dict):
        return ""
    role = str(raw_message["role"])
    content = _chat_content(raw_message.get("content"))
    if role == "system":
        body = "" if content is None else f'<div class="content">{_escape(content)}</div>'
        return f'<details class="system-message"><summary>system</summary>{body}</details>'

    role_class = role if role in {"user", "assistant", "tool"} else "unknown"
    if frame_ctx is not None and role == "user" and isinstance(raw_message.get("content"), list):
        body = _render_frame_parts(cast(list[object], raw_message["content"]), frame_ctx)
    else:
        body = "" if content is None else f'<div class="content">{_escape(content)}</div>'
    if role == "tool":
        return (
            f'<div class="message {role_class}" data-step="{anchor}">'
            f'<div class="role">{_escape(role)}</div>{body}</div>'
        )
    calls = ""
    tool_calls = raw_message.get("tool_calls")
    if isinstance(tool_calls, list):
        calls = "".join(_render_tool_call(raw_call) for raw_call in tool_calls)
    return (
        f'<div class="message {role_class}" data-step="{anchor}">'
        f'<div class="role">{_escape(role)}</div>{body}{calls}</div>'
    )


def _render_chat_transcript(
    transcript: list[object], frame_ctx: _FrameContext | None = None
) -> str:
    """Render a defensive role-oriented conversation."""
    rendered: list[str] = []
    anchor = 0
    for message in transcript:
        step = _message_step(message)
        if step is not None:
            anchor = step
        rendered.append(_render_message(message, frame_ctx, anchor=anchor))
    return '<div class="conversation">' + "".join(rendered) + "</div>"


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


def _render_transcript(transcript: object, frame_ctx: _FrameContext | None = None) -> str:
    """Render chat-shaped records conversationally and all others as bounded JSON."""
    if _is_chat_transcript(transcript):
        return _render_chat_transcript(cast(list[object], transcript), frame_ctx)
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
    frame_ctx: _FrameContext | None, scene_id: str, trial: int
) -> _FrameContext | None:
    """Specialize a document frame context to one scene trial."""
    if frame_ctx is None:
        return None
    return _FrameContext(
        frame_ctx.frames_dir,
        _safe(f"{scene_id}-e{trial}"),
        frame_ctx.budget,
    )


def _render_trial_transcript(
    transcript: object,
    frame_ctx: _FrameContext | None,
    scene_id: str,
    trial: int,
) -> str:
    """Render one transcript with its enumerate-index frame correlation context."""
    return _render_transcript(transcript, _trial_frame_context(frame_ctx, scene_id, trial))


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


def _render_player(media: Sequence[tuple[str, str]], fps: float) -> str:
    """Render one synchronized camera player from already-relative media links."""
    cameras = "".join(
        '<figure class="cam">'
        f"<figcaption>{_escape(label)}</figcaption>"
        f'<video src="{_escape(href)}" preload="metadata" muted playsinline></video>'
        "</figure>"
        for label, href in media
    )
    return (
        f'<section class="player" data-fps="{fps:g}">'
        f'<div class="cams">{cameras}</div>'
        '<div class="controls">'
        '<button class="playpause" type="button" disabled>Play</button>'
        '<input class="scrub" type="range" min="0" max="1000" value="0" step="1" disabled>'
        '<span class="clock">step 0 · 0.0s</span>'
        '<button class="follow on" type="button" disabled>Follow</button>'
        "</div></section>"
    )


def _scene_section(
    scene: SceneResult,
    *,
    open_transcript: bool,
    budget: _FrameBudget,
    log_path: Path | None,
    fps: float,
    trial_media: Mapping[str, Sequence[tuple[str, str]]] | None,
    frame_ctx: _FrameContext | None = None,
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
        _score_chips(epoch, prefix=f"trial {index} ") for index, epoch in enumerate(scene.epochs)
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

    trials: list[str] = []
    trial_count = max(len(scene.epochs), len(scene.policy_transcripts))
    for trial in range(trial_count):
        prefix = _safe(f"{scene.scene_id}-e{trial}")
        media = () if trial_media is None else trial_media.get(prefix, ())
        player = "" if not media else _render_player(media, fps)
        transcript = (
            scene.policy_transcripts[trial] if trial < len(scene.policy_transcripts) else None
        )
        transcript_html = (
            ""
            if transcript is None
            else (
                f'<details class="transcript"{" open" if open_transcript else ""}>'
                f"<summary>Trial {trial} transcript</summary>"
                f"{_render_trial_transcript(transcript, frame_ctx, scene.scene_id, trial)}"
                "</details>"
            )
        )
        wire = _render_trial_wire(scene, trial, log_path, budget)
        player_class = " has-player" if media else ""
        trials.append(
            f'<div class="trial{player_class}" id="trial-{_escape(prefix)}">'
            f"{player}{transcript_html}{wire}</div>"
        )
    return (
        '<section class="scene">'
        f'<div class="scene-head"><h2>{_escape(scene.scene_id)}</h2>'
        f"{_status_badge(scene.status)}</div>{instruction}{error}{reduced_block}{epoch_block}"
        f"{reasons_block}{judgements_block}{notes_block}{''.join(trials)}</section>"
    )


def render_html(
    log: EvalLog,
    *,
    title: str,
    log_path: Path | None = None,
    frames_dir: Path | None = None,
    frames_budget_bytes: int = 50_000_000,
    trial_media: Mapping[str, Sequence[tuple[str, str]]] | None = None,
) -> str:
    """Return one HTML document describing the complete evaluation log."""
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
    budget = _FrameBudget(limit=frames_budget_bytes)
    frame_ctx = None if frames_dir is None else _FrameContext(frames_dir, "", budget)
    fps = default_fps(log.eval.embodiment_info)[0]
    scenes = "".join(
        _scene_section(
            scene,
            open_transcript=transcript_count == 1,
            budget=budget,
            log_path=log_path,
            fps=fps,
            trial_media=trial_media,
            frame_ctx=frame_ctx,
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
        f"{frames_budget_bytes / 1_000_000:g} MB ({budget.embedded} embedded)</span>"
    )
    meta_tail = (
        f"<span>inspect-robots {_escape(log.eval.inspect_robots_version)}</span>"
        f"<span>{git}</span>{frames_chip}"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_STYLES}</style>
<script>{_VIEW_SCRIPT}</script>
</head>
<body>
<header><div class="header-inner">
  <div class="header-top"><h1>{_escape(log.eval.task)}</h1>{_status_badge(log.status)}</div>
  <div class="meta"><span>{_escape(log.eval.created)}</span>
  {meta_tail}</div>
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
