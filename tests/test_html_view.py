"""Pure rendering tests for the self-contained HTML eval-log viewer."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import html
import json
import re
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots._html import (
    _JSON_STRING_LIMIT,
    _frame_markup,
    _FrameBudget,
    _FrameContext,
    _render_chat_transcript,
    _render_tool_call,
    _render_transcript,
    _trial_camera_streams,
    _VideoBudget,
    _VideoContext,
    _warn_video_degrade,
    render_html,
)
from inspect_robots._pngenc import png_data_url
from inspect_robots.frames import FrameStore, _safe
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult


def _chat(*messages: object) -> list[object]:
    return list(messages)


def _log(
    *,
    status: str = "success",
    transcripts: tuple[Any, ...] = (),
) -> EvalLog:
    return EvalLog(
        version=1,
        status=status,
        eval=EvalSpec(
            task="pick-cube",
            policy="agent",
            embodiment="arm",
            created="2026-07-16T12:00:00Z",
            inspect_robots_version="0.7.0",
            git_commit="abc123",
            policy_config={"effort": "low", "temperature": 0.2},
            seed=7,
            max_steps=120,
        ),
        results=EvalResults(
            total_scenes=1,
            total_trials=max(1, len(transcripts)),
            metrics={"success_at_end": 0.75},
            errored_trials=1,
        ),
        stats=EvalStats(
            started_at="start",
            completed_at="end",
            duration_s=12.5,
            total_steps=42,
            mean_inference_latency_s=0.125,
        ),
        samples=(
            SceneResult(
                scene_id="scene-0",
                status=status,
                reduced={"success_at_end": 0.75},
                epochs=({"success_at_end": 1.0}, {}),
                error="one trial failed",
                instruction="pick up the cube",
                operator_judgements=("y", None),
                # Deliberately on trial 1, not trial 0: a note at index 0 cannot
                # tell correct indexing apart from enumerating the filtered list.
                operator_notes=(None, "gripper closed early"),
                termination_reasons=("success", None),
                policy_transcripts=transcripts,
            ),
        ),
        error="run warning",
    )


def _frame_path(root: Path, name: str, step: int, *, epoch: int = 0) -> Path:
    return root / f"{_safe(f'scene-0-e{epoch}')}_{_safe(name)}_{step:06d}.npy"


def _save_frame(
    root: Path,
    name: str,
    step: int,
    image: npt.NDArray[Any],
    *,
    epoch: int = 0,
) -> Path:
    path = _frame_path(root, name, step, epoch=epoch)
    np.save(path, image)
    return path


def _parts(name: str = "top_cam", step: int = 4) -> list[object]:
    return [
        {"type": "text", "text": f"camera {name!r} (step {step}):"},
        {"type": "text", "text": "[image omitted: streamed camera frame]"},
    ]


def _frame_log(parts: Sequence[object], *, role: str = "user") -> EvalLog:
    return _log(transcripts=(_chat({"role": role, "content": list(parts)}),))


def _wire_log(
    tmp_path: Path,
    rows: Sequence[object],
    *,
    pointer: str = "wire/run/scene-0-e0/calls.jsonl",
    scene_id: str = "scene-0",
) -> tuple[EvalLog, Path, Path]:
    log = _log()
    scene = dataclasses.replace(
        log.samples[0],
        scene_id=scene_id,
        trial_metadata=({"wire_capture": pointer}, {}),
    )
    log_path = tmp_path / "run.json"
    calls_path = tmp_path / pointer
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    calls_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    return dataclasses.replace(log, samples=(scene,)), log_path, calls_path


def _png_dimensions_from_document(document: str) -> tuple[int, int]:
    (source,) = re.findall(r'src="(data:image/png;base64,[^"]+)"', document)
    encoded = base64.b64decode(source.partition(",")[2])
    return struct.unpack(">II", encoded[16:24])


@pytest.mark.parametrize(
    ("status", "label", "badge_class"),
    [
        ("success", "completed", "status-completed"),
        ("error", "error", "status-error"),
        ("cancelled", "cancelled", "status-cancelled"),
        ("started", "running", "status-running"),
        ("unexpected", "unexpected", "status-neutral"),
    ],
)
def test_header_status_metrics_and_scene_content(status: str, label: str, badge_class: str) -> None:
    document = render_html(_log(status=status), title="pick-cube - run.json")

    assert "<!doctype html>" in document
    assert "pick-cube - run.json" in document
    assert f'class="badge {badge_class}">{label}</span>' in document
    assert "2026-07-16T12:00:00Z" in document
    assert "inspect-robots 0.7.0" in document
    assert "git abc123" in document
    assert "agent" in document and "arm" in document
    assert "effort" in document and "temperature" in document
    assert "seed" in document and "max steps" in document
    assert "mean inference latency" in document
    assert "duration" in document and "total steps" in document
    assert "success_at_end" in document and "0.75" in document
    assert "scenes" in document and "trials" in document and "errored" in document
    assert "scene-0" in document and "pick up the cube" in document
    assert "Reduced scores" in document and "Trial scores" in document
    assert "Termination reasons" in document and "Operator judgements" in document
    assert "Grader notes" in document
    assert document.count('class="grader-note"') == 1
    assert ">trial 1</span>gripper closed early</div>" in document
    assert "one trial failed" in document
    assert document.count(">n/a</span>") == 2
    assert "prefers-color-scheme: light" in document
    assert "prefers-color-scheme: dark" in document
    assert document.count("<script") == 2
    # The literal is duplicated here on purpose: an independent copy is what
    # makes this an injection guard rather than a tautology.
    assert (
        "<script>document.addEventListener('click', (event) => {\n"
        "  const cell = event.target.closest('.frame-cell');\n"
        "  if (cell) cell.classList.toggle('wide');\n"
        "});</script>" in document
    )


def test_non_finite_metric_written_as_null_renders_as_not_available() -> None:
    """Regression for #253: json_log.py sanitizes inf/nan metrics to ``None``
    so the log stays valid JSON. The HTML viewer must tolerate the ``None``
    it reads back rather than crashing on ``.4g`` formatting."""
    log = _log()
    log = dataclasses.replace(
        log,
        results=dataclasses.replace(log.results, metrics={"min_distance_to_goal": None}),  # type: ignore[dict-item]
    )

    document = render_html(log, title="pick-cube - run.json")

    assert "min_distance_to_goal" in document
    assert '<div class="stat-value">n/a</div>' in document


def test_running_refresh_banner_pending_scores_and_escaping() -> None:
    log = _log(status="started")
    scene = dataclasses.replace(
        log.samples[0],
        trial_metadata=(
            {},
            {"live": {"step": 4, "updated_at": "2026-08-06T12:34:56<script>"}},
        ),
    )

    document = render_html(
        dataclasses.replace(log, samples=(scene,)),
        title="running",
        refresh_seconds=2,
    )

    assert '<meta http-equiv="refresh" content="2">' in document
    assert "RUNNING — refreshes every 2s · last update 12:34:56" in document
    assert '<span class="badge status-running">running</span>' in document
    assert '<span class="score-chip">trial 1 pending</span>' in document

    hostile_scene = dataclasses.replace(
        scene,
        trial_metadata=({}, {"live": {"step": 4, "updated_at": "<script>"}}),
    )
    hostile = render_html(
        dataclasses.replace(log, samples=(hostile_scene,)),
        title="running",
        refresh_seconds=2,
    )
    assert "last update &lt;script&gt;" in hostile
    assert "&amp;lt;script&amp;gt;" not in hostile


def test_refresh_and_pending_output_are_opt_in() -> None:
    completed = render_html(_log(), title="complete")
    static_started = render_html(_log(status="started"), title="stale")

    assert 'http-equiv="refresh"' not in completed
    assert "RUNNING —" not in completed
    assert "pending" not in completed
    assert 'http-equiv="refresh"' not in static_started
    assert "RUNNING —" not in static_started
    assert "trial 1 pending" in static_started


def test_absent_optional_fields_and_empty_scene_sequences_are_omitted() -> None:
    log = _log()
    spec = dataclasses.replace(
        log.eval,
        git_commit=None,
        policy_config={},
        seed=None,
        max_steps=None,
    )
    stats = dataclasses.replace(log.stats, mean_inference_latency_s=None)
    scene = dataclasses.replace(
        log.samples[0],
        instruction=None,
        error=None,
        reduced={},
        epochs=(),
        operator_judgements=(),
        operator_notes=(),
        termination_reasons=(),
    )

    document = render_html(
        dataclasses.replace(log, eval=spec, stats=stats, samples=(scene,)), title="minimal"
    )

    assert '<span class="chip">unknown</span>' in document
    assert "<dt>seed</dt>" not in document
    assert "<dt>max steps</dt>" not in document
    assert "mean inference latency" not in document
    assert "pick up the cube" not in document
    assert "one trial failed" not in document
    assert "Reduced scores" not in document
    assert "Trial scores" not in document
    assert "Termination reasons" not in document
    assert "Operator judgements" not in document
    assert "Grader notes" not in document
    assert 'class="grader-note"' not in document


def test_multiline_rubric_renders_as_an_escaped_dropdown() -> None:
    """Preserve a rubric's line structure while escaping its foreign text."""
    log = _log()
    scene = dataclasses.replace(
        log.samples[0],
        scene_metadata={"rubric": "Cube is <lifted> cleanly.\nGripper stays closed."},
    )

    document = render_html(dataclasses.replace(log, samples=(scene,)), title="rubric")

    assert (
        '<details class="rubric"><summary>Rubric</summary>'
        '<p class="rubric-text">Cube is &lt;lifted&gt; cleanly.\n'
        "Gripper stays closed.</p></details>" in document
    )
    assert '<p class="instruction">pick up the cube</p><details class="rubric">' in document


@pytest.mark.parametrize(
    "scene_metadata",
    [
        {},
        {"rubric": " \n\t "},
        {"rubric": ["not", "a", "string"]},
    ],
)
def test_ineligible_rubric_values_render_nothing(scene_metadata: dict[str, Any]) -> None:
    """Omit the rubric dropdown for absent, blank, and non-string values."""
    log = _log()
    scene = dataclasses.replace(log.samples[0], scene_metadata=scene_metadata)

    document = render_html(dataclasses.replace(log, samples=(scene,)), title="no rubric")

    assert 'class="rubric"' not in document


def test_started_log_with_synthetic_scene_metadata_renders_rubric() -> None:
    """Render rubric metadata independently of the log's lifecycle state."""
    log = _log(status="started")
    scene = dataclasses.replace(
        log.samples[0],
        scene_metadata={"rubric": "The cube clears the table."},
    )

    document = render_html(dataclasses.replace(log, samples=(scene,)), title="started rubric")

    assert '<details class="rubric"><summary>Rubric</summary>' in document
    assert '<p class="rubric-text">The cube clears the table.</p></details>' in document


def test_scene_where_no_trial_carried_a_note_renders_no_grader_notes_block() -> None:
    # The ordinary case: operator_notes is parallel to epochs, so an unannotated
    # run is a tuple of Nones, not the empty tuple the omission test above uses.
    log = _log()
    scene = dataclasses.replace(log.samples[0], operator_notes=(None, None))

    document = render_html(dataclasses.replace(log, samples=(scene,)), title="no notes")

    assert "Operator judgements" in document
    assert "Grader notes" not in document
    assert 'class="grader-note"' not in document


def test_seconds_horizon_shows_declared_and_resolved_limits() -> None:
    log = _log()
    spec = dataclasses.replace(log.eval, max_seconds=12.5, max_steps=188)
    document = render_html(dataclasses.replace(log, eval=spec), title="timed")

    assert "<dt>max seconds</dt>" in document
    assert "<dd>12.5</dd>" in document
    assert "<dt>resolved max steps</dt>" in document
    assert "<dd>188</dd>" in document
    assert "<dt>max steps</dt>" not in document


def test_seconds_horizon_without_resolved_steps_omits_step_limit() -> None:
    log = _log()
    spec = dataclasses.replace(log.eval, max_seconds=12.5, max_steps=None)
    document = render_html(dataclasses.replace(log, eval=spec), title="incomplete")

    assert "<dt>max seconds</dt>" in document
    assert "<dt>resolved max steps</dt>" not in document


def test_every_foreign_text_surface_is_escaped_exactly_once() -> None:
    attack = "<script>alert(1)</script>"
    transcript = _chat(
        {"role": "user", "content": attack},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "move_by", "arguments": attack}},
            ],
        },
    )
    log = _log(status=attack, transcripts=(transcript,))
    scene = dataclasses.replace(
        log.samples[0],
        scene_id=attack,
        status=attack,
        instruction=attack,
        scene_metadata={"rubric": attack},
        error=attack,
        operator_notes=(attack, None),
    )
    results = dataclasses.replace(log.results, metrics={attack: 1.0})

    document = render_html(
        dataclasses.replace(log, samples=(scene,), results=results), title=attack
    )

    assert attack not in document
    assert document.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 9
    assert "&amp;lt;script" not in document


def test_chat_roles_system_details_tool_call_and_result_are_rendered() -> None:
    transcript = _chat(
        {"role": "system", "content": "follow the rules"},
        {"role": "user", "content": "where is the cube?"},
        {
            "role": "assistant",
            "content": "moving now",
            "tool_calls": [
                {"function": {"name": "move_by", "arguments": '{"dx": 0.1}'}},
            ],
        },
        {"role": "tool", "content": "moved 2 steps"},
    )

    document = render_html(_log(transcripts=(transcript,)), title="conversation")

    assert '<details class="system-message"><summary>system</summary>' in document
    assert '<details class="system-message" open>' not in document
    assert '<div class="message user">' in document
    assert '<div class="message assistant">' in document
    assert '<div class="message tool">' in document
    assert "where is the cube?" in document and "moving now" in document
    assert "move_by({&quot;dx&quot;: 0.1})" in document
    assert "moved 2 steps" in document


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("move_by", '{"note": "edge closer"}', "edge closer"),
        ("move_by", {"note": "dict note"}, "dict note"),
        ("done", {"summary": "cube secured"}, "cube secured"),
        ("give_up", '{"reason": "blocked"}', "blocked"),
    ],
)
def test_agent_note_callouts_support_all_argument_shapes(
    name: str, arguments: object, expected: str
) -> None:
    transcript = _chat(
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        }
    )

    document = render_html(_log(transcripts=(transcript,)), title="notes")

    assert document.count('class="agent-note"') == 1
    assert document.count(">agent note</span>") == 1
    assert expected in document
    assert document.index(expected) < document.index(f'class="call">{name}')


def test_malformed_json_arguments_render_verbatim_without_a_callout() -> None:
    arguments = '{"note": <broken>}'
    transcript = _chat(
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "move_by", "arguments": arguments}},
            ],
        }
    )

    document = render_html(_log(transcripts=(transcript,)), title="malformed")

    assert 'class="agent-note"' not in document
    assert "{&quot;note&quot;: &lt;broken&gt;}" in document


def test_non_string_empty_and_whitespace_notes_do_not_render_callouts() -> None:
    transcript = _chat(
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "move", "arguments": {"note": 3}}},
                {"function": {"name": "move", "arguments": {"note": ""}}},
                {"function": {"name": "move", "arguments": {"note": "  \n "}}},
                {"function": {"name": "move", "arguments": "[1, 2]"}},
            ],
        }
    )

    document = render_html(_log(transcripts=(transcript,)), title="empty notes")

    assert 'class="agent-note"' not in document
    assert document.count('class="pretty-call"') == 3
    assert document.count('class="call"') == 5


def test_chat_defensive_guards_skip_malformed_calls_and_role_only_content() -> None:
    transcript = _chat(
        {"role": "assistant", "content": 42, "tool_calls": "not a list"},
        {
            "role": "assistant",
            "tool_calls": [
                "not a call",
                {"function": "not a function"},
                {"function": {"name": "move", "arguments": {"dx": 1}}},
            ],
        },
    )

    document = render_html(_log(transcripts=(transcript,)), title="defensive")

    assert document.count('<div class="message assistant">') == 4
    assert "not a list" not in document
    assert "not a call" not in document
    assert "not a function" not in document
    assert "move({&quot;dx&quot;: 1})" in document
    assert 'class="content"' not in document

    assert _render_chat_transcript(["not a message"]).count('class="raw-transcript"') == 1


def test_role_less_message_renders_as_unknown_instead_of_crashing() -> None:
    """Transcripts are foreign data: a dict without a role must degrade, not raise."""
    document = _render_chat_transcript([{"content": "orphan"}, {"role": "assistant"}])
    assert 'class="message unknown"' in document
    assert "orphan" in document


def test_non_dict_message_falls_back_to_preformatted_json() -> None:
    transcript = [{"role": "user", "content": "hello"}, "not a message"]

    document = render_html(_log(transcripts=(transcript,)), title="fallback")

    assert "<pre>" in document
    assert "not a message" in document
    assert '<div class="conversation">' not in document


def test_media_parts_leave_image_markers_in_pov_only() -> None:
    """Foreign media payloads never embed, but the raw transcript records their presence."""
    transcript = _chat(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look here"},
                {"type": "image_url", "image_url": {"url": "large-data"}},
                "unknown media",
            ],
        }
    )

    document = render_html(_log(transcripts=(transcript,)), title="media")

    assert "look here\n[image]\n[image]" in document
    assert "large-data" not in document
    assert "unknown media" not in document


def test_json_fallback_keeps_a_string_at_exactly_the_limit_untruncated() -> None:
    """A value of exactly the elision limit must not gain a truncation marker."""
    transcript = ({"payload": "x" * _JSON_STRING_LIMIT},)

    document = render_html(_log(transcripts=(transcript,)), title="boundary")

    assert "chars truncated" not in document


def test_non_chat_json_fallback_elides_long_raw_string_values_before_escaping() -> None:
    long_value = "x" * 2047 + "<" + "y" * 10
    transcript = ({"payload": [long_value, 3], "short": "kept"},)

    document = render_html(_log(transcripts=(transcript,)), title="large fallback")

    assert "<pre>" in document
    assert "[... 10 chars truncated]" in document
    assert "&lt;[... 10 chars truncated]" in document
    assert "short" in document and "kept" in document
    assert long_value not in document


def test_none_transcripts_are_skipped_and_exactly_one_panel_opens() -> None:
    transcript = _chat({"role": "user", "content": "hello"})

    document = render_html(_log(transcripts=(None, transcript, None)), title="one")

    assert document.count('<details class="transcript" open>') == 1
    assert "Trial 1 transcript" in document
    assert "Trial 0 transcript" not in document
    assert "Trial 2 transcript" not in document


def test_two_transcripts_leave_every_panel_collapsed() -> None:
    first = _chat({"role": "user", "content": "first"})
    second = {"custom": "second"}

    document = render_html(_log(transcripts=(first, second)), title="two")

    assert document.count('<details class="transcript">') == 2
    assert '<details class="transcript" open>' not in document


def test_transcript_free_log_reports_none_recorded() -> None:
    document = render_html(_log(transcripts=(None,)), title="none")

    assert "no policy transcripts recorded" in document
    assert '<details class="transcript"' not in document


def test_stored_frame_embeds_between_escaped_text_runs(tmp_path: Path) -> None:
    name = "cam <wide>"
    parts = [
        {"type": "text", "text": "caption <before>"},
        *_parts(name, 7),
        {"type": "text", "text": "after frame"},
    ]
    _save_frame(tmp_path, name, 7, np.arange(36, dtype=np.uint8).reshape(3, 4, 3))

    document = render_html(_frame_log(parts), title="frames", frames_dir=tmp_path)

    assert document.count('<img class="frame"') == 1
    assert document.count('<div class="frame-row">') == 1
    assert document.count('<figure class="frame-cell">') == 1
    assert 'loading="lazy"' in document
    assert 'src="data:image/png;base64,' in document
    assert 'alt="camera cam &lt;wide&gt; step 7"' in document
    assert "<figcaption>cam &lt;wide&gt;</figcaption>" in document
    assert '<div class="turn-step">step 7</div>' in document
    assert document.index('<img class="frame"') < document.index("after frame")
    assert document.count("[image omitted: streamed camera frame]") == 1
    assert document.count('<div class="content"></div>') == 0
    assert "img.frame" in document


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 1), (3, 4, 3), (3, 4, 4)])
def test_viewer_embeds_every_allowed_frame_shape(tmp_path: Path, shape: tuple[int, ...]) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros(shape, dtype=np.uint8))

    document = render_html(_frame_log(_parts()), title="shape", frames_dir=tmp_path)

    assert document.count('<img class="frame"') == 1


def test_all_frame_misses_are_byte_identical_to_frames_off(tmp_path: Path) -> None:
    transcript = _chat(
        {"role": "user", "content": []},
        {"role": "user", "content": "plain text"},
        {
            "role": "user",
            "content": [
                {"type": "text"},
                {"type": "text", "text": 7},
                {"type": "image_url"},
                "not a dictionary",
                *_parts("missing", 9),
            ],
        },
    )
    log = _log(transcripts=(transcript,))

    assert render_html(log, title="same", frames_dir=tmp_path) == render_html(log, title="same")


def test_errored_trial_final_step_with_no_file_degrades(tmp_path: Path) -> None:
    log = _frame_log(_parts("top_cam", 11))
    scene = dataclasses.replace(log.samples[0], status="error", error="step failed")

    document = render_html(
        dataclasses.replace(log, samples=(scene,)), title="missing", frames_dir=tmp_path
    )

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_hostile_huge_step_label_degrades_instead_of_crashing(tmp_path: Path) -> None:
    """A multi-thousand-digit step must miss the regex, not trip int()'s digit limit."""
    parts = [
        {"type": "text", "text": f"camera 'top_cam' (step {'1' * 5000}):"},
        {"type": "text", "text": "[image omitted: streamed camera frame]"},
    ]

    document = render_html(_frame_log(parts), title="hostile", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_label_with_trailing_text_degrades(tmp_path: Path) -> None:
    """fullmatch, not match: trailing text after the colon must not arm a lookup."""
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))
    parts = [
        {"type": "text", "text": "camera 'top_cam' (step 4): extra"},
        {"type": "text", "text": "[image omitted: streamed camera frame]"},
    ]

    document = render_html(_frame_log(parts), title="trailing", frames_dir=tmp_path)

    assert '<img class="frame"' not in document


def test_truncation_is_sticky_even_for_a_smaller_later_frame(tmp_path: Path) -> None:
    """The first overflow degrades every later lookup, including ones that would fit."""
    big = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    small = np.zeros((1, 1, 1), dtype=np.uint8)
    first_payload = len(png_data_url(big).partition(",")[2])
    parts = [*_parts("first", 1), *_parts("second", 2), *_parts("third", 3)]
    _save_frame(tmp_path, "first", 1, big)
    _save_frame(tmp_path, "second", 2, big)
    _save_frame(tmp_path, "third", 3, small)

    document = render_html(
        _frame_log(parts),
        title="sticky",
        frames_dir=tmp_path,
        frames_budget_bytes=first_payload,
    )

    assert document.count('<img class="frame"') == 1
    assert document.count("[image omitted: streamed camera frame]") == 3


def test_label_without_step_suffix_degrades(tmp_path: Path) -> None:
    parts = [
        {"type": "text", "text": "camera 'top_cam':"},
        {"type": "text", "text": "[image omitted: streamed camera frame]"},
    ]

    document = render_html(_frame_log(parts), title="legacy", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_single_quote_camera_name_regex_miss_degrades(tmp_path: Path) -> None:
    name = "'"
    _save_frame(tmp_path, name, 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts(name)), title="quote", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_backslash_camera_name_regex_hit_but_file_lookup_misses(tmp_path: Path) -> None:
    name = "rear\\cam"
    stored = _save_frame(tmp_path, name, 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts(name)), title="slash", frames_dir=tmp_path)

    assert stored.exists()
    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_space_in_camera_name_reconstructs_and_embeds(tmp_path: Path) -> None:
    name = "top camera"
    _save_frame(tmp_path, name, 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts(name)), title="space", frames_dir=tmp_path)

    assert document.count('<img class="frame"') == 1


def test_placeholder_without_pending_label_degrades(tmp_path: Path) -> None:
    parts = [{"type": "text", "text": "[image omitted: streamed camera frame]"}]

    document = render_html(_frame_log(parts), title="orphan", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_label_without_following_placeholder_stays_text(tmp_path: Path) -> None:
    parts = [{"type": "text", "text": "camera 'top_cam' (step 4):"}]

    document = render_html(_frame_log(parts), title="label", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "camera &#x27;top_cam&#x27; (step 4):" in document


def test_corrupt_frame_file_degrades(tmp_path: Path) -> None:
    _frame_path(tmp_path, "top_cam", 4).write_bytes(b"not an npy file")

    document = render_html(_frame_log(_parts()), title="corrupt", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_pickled_frame_file_degrades_without_unpickling(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.array([{"unsafe": True}], dtype=object))

    document = render_html(_frame_log(_parts()), title="pickle", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_wrong_frame_channel_count_degrades(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 3, 2), dtype=np.uint8))

    document = render_html(_frame_log(_parts()), title="channels", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_wrong_frame_rank_degrades(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((3,), dtype=np.uint8))

    document = render_html(_frame_log(_parts()), title="rank", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_non_uint8_frame_degrades(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 3, 3), dtype=np.float32))

    document = render_html(_frame_log(_parts()), title="dtype", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_zero_size_frame_degrades(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((0, 3, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts()), title="empty", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def test_three_camera_pairs_embed_in_part_order(tmp_path: Path) -> None:
    parts: list[object] = []
    for step, name in enumerate(("left", "top", "right"), start=1):
        parts.extend(_parts(name, step))
        _save_frame(tmp_path, name, step, np.full((2, 2, 3), step, dtype=np.uint8))

    document = render_html(_frame_log(parts), title="three", frames_dir=tmp_path)

    assert document.count('<div class="frame-row">') == 1
    assert document.count('<figure class="frame-cell">') == 3
    assert document.count('<img class="frame"') == 3
    assert document.count("<figcaption>") == 3
    assert "<figcaption>left</figcaption>" in document
    assert "<figcaption>top</figcaption>" in document
    assert "<figcaption>right</figcaption>" in document
    assert document.index("camera left step 1") < document.index("camera top step 2")
    assert document.index("camera top step 2") < document.index("camera right step 3")
    assert "camera &#x27;left&#x27; (step 1):" in document
    assert "camera &#x27;top&#x27; (step 2):" in document
    assert "camera &#x27;right&#x27; (step 3):" in document
    assert document.count('<div class="content"></div>') == 0


def test_separate_user_observations_render_separate_frame_rows(tmp_path: Path) -> None:
    _save_frame(tmp_path, "first", 1, np.zeros((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "second", 2, np.ones((2, 2, 3), dtype=np.uint8))
    transcript = _chat(
        {"role": "user", "content": _parts("first", 1)},
        {"role": "user", "content": _parts("second", 2)},
    )

    document = render_html(_log(transcripts=(transcript,)), title="two rows", frames_dir=tmp_path)

    assert document.count('<div class="frame-row">') == 2
    assert document.count('<figure class="frame-cell">') == 2


def test_non_empty_text_between_frames_splits_rows(tmp_path: Path) -> None:
    parts = [*_parts("first", 1), {"type": "text", "text": "between"}, *_parts("second", 2)]
    _save_frame(tmp_path, "first", 1, np.zeros((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "second", 2, np.ones((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(parts), title="split rows", frames_dir=tmp_path)

    assert document.count('<div class="frame-row">') == 1
    assert document.count('<figure class="frame-cell">') == 2
    assert "between" in document


def test_empty_text_between_frames_does_not_split_row(tmp_path: Path) -> None:
    parts = [*_parts("first", 1), {"type": "text", "text": ""}, *_parts("second", 2)]
    _save_frame(tmp_path, "first", 1, np.zeros((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "second", 2, np.ones((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(parts), title="one row", frames_dir=tmp_path)

    assert document.count('<div class="frame-row">') == 1
    assert document.count('<figure class="frame-cell">') == 2
    assert document.count('<div class="content"></div>') == 0


def test_trailing_empty_text_after_embed_emits_no_div(tmp_path: Path) -> None:
    parts = [*_parts("solo", 3), {"type": "text", "text": ""}]
    _save_frame(tmp_path, "solo", 3, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(parts), title="trailing", frames_dir=tmp_path)

    assert document.count('<figure class="frame-cell">') == 1
    assert document.count('<div class="content"></div>') == 0


def test_text_between_label_and_placeholder_precedes_captioned_frame(tmp_path: Path) -> None:
    parts = [
        {"type": "text", "text": "camera 'top_cam' (step 4):"},
        {"type": "text", "text": "intervening text"},
        {"type": "text", "text": "[image omitted: streamed camera frame]"},
    ]
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(parts), title="intervening", frames_dir=tmp_path)

    caption = "<figcaption>top_cam</figcaption>"
    assert caption in document
    assert "intervening text" in document
    assert document.count('<figure class="frame-cell">') == 1
    assert "camera &#x27;top_cam&#x27; (step 4):" in document


def test_later_label_captions_frame_and_earlier_label_stays_in_text(tmp_path: Path) -> None:
    parts = [
        {"type": "text", "text": "camera 'first' (step 1):"},
        {"type": "text", "text": "camera 'second' (step 2):"},
        {"type": "text", "text": "[image omitted: streamed camera frame]"},
    ]
    _save_frame(tmp_path, "second", 2, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(parts), title="later label", frames_dir=tmp_path)

    assert "camera &#x27;first&#x27; (step 1):" in document
    assert "<figcaption>second</figcaption>" in document
    assert "camera &#x27;second&#x27; (step 2):" in document
    assert document.count('<figure class="frame-cell">') == 1


def test_single_frame_renders_one_row_and_one_cell(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts()), title="single", frames_dir=tmp_path)

    assert document.count('<div class="frame-row">') == 1
    assert document.count('<figure class="frame-cell">') == 1
    assert document.count('<div class="content"></div>') == 0


def test_budget_truncation_keeps_exact_frame_text_fallback(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(
        _frame_log(_parts()),
        title="truncated",
        frames_dir=tmp_path,
        frames_budget_bytes=1,
    )

    assert (
        '<div class="content">camera &#x27;top_cam&#x27; (step 4):\n'
        "[image omitted: streamed camera frame]</div>"
    ) in document
    assert '<figure class="frame-cell">' not in document
    assert '<div class="frame-row">' not in document


def test_camera_name_is_escaped_in_frame_caption_and_alt(tmp_path: Path) -> None:
    name = "<b>"
    _save_frame(tmp_path, name, 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts(name)), title="escaped", frames_dir=tmp_path)

    assert "<figcaption>&lt;b&gt;</figcaption>" in document
    assert 'alt="camera &lt;b&gt; step 4"' in document
    assert "<b>" not in document


def test_frame_expand_script_and_wide_cell_rule_ship_without_frames() -> None:
    document = render_html(_log(), title="no frames")

    assert "event.target.closest('.frame-cell')" in document
    assert ".frame-cell.wide { grid-column: 1 / -1; }" in document


def test_transcript_index_is_used_as_epoch_for_frame_lookup(tmp_path: Path) -> None:
    chat = _chat({"role": "user", "content": _parts()})
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8), epoch=1)

    document = render_html(_log(transcripts=(None, chat)), title="epoch", frames_dir=tmp_path)

    assert document.count('<img class="frame"') == 1
    assert "Trial 1 transcript" in document


def test_oversize_frame_is_stride_subsampled_below_limit(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((1000, 701, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts()), title="large", frames_dir=tmp_path)

    width, height = _png_dimensions_from_document(document)
    assert max(width, height) <= 448


def test_frame_budget_embeds_first_then_truncates_remaining(tmp_path: Path) -> None:
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    payload_size = len(png_data_url(frame).partition(",")[2])
    parts = [*_parts("first", 1), *_parts("second", 2), *_parts("third", 3)]
    for step, name in enumerate(("first", "second", "third"), start=1):
        _save_frame(tmp_path, name, step, frame)

    document = render_html(
        _frame_log(parts),
        title="budget",
        frames_dir=tmp_path,
        frames_budget_bytes=payload_size,
    )

    assert document.count('<img class="frame"') == 1
    assert document.count("[image omitted: streamed camera frame]") == 3
    budget_mb = payload_size / 1_000_000
    assert f"embedded media truncated at {budget_mb:g} MB (1 embedded)" in document


def test_live_frame_budget_allocates_newest_first_and_reuses_encoded_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    payload_size = len(png_data_url(frame).partition(",")[2])
    store = FrameStore(str(tmp_path))
    store.put("scene-0-e0", 1, "older", frame)
    store.put("scene-0-e1", 2, "newest", frame)
    log = _log(
        transcripts=(
            _chat({"role": "user", "content": _parts("older", 1)}),
            _chat(
                {
                    "role": "user",
                    "content": [
                        *_parts("newest", 2),
                        *_parts("newest", 2),
                        *_parts("missing", 3),
                    ],
                }
            ),
        )
    )
    encode_calls: list[npt.NDArray[np.uint8]] = []
    encode = png_data_url

    def counting_encode(image: npt.NDArray[np.uint8]) -> str:
        encode_calls.append(image)
        return encode(image)

    monkeypatch.setattr("inspect_robots._html.png_data_url", counting_encode)

    document = render_html(
        log,
        title="live budget",
        frames_dir=tmp_path,
        frames_budget_bytes=8 * payload_size,
        live_frames_budget_bytes=payload_size,
    )

    assert document.count('<img class="frame"') == 2
    assert "camera newest step 2" in document
    assert "camera older step 1" not in document
    assert document.count("[image omitted: streamed camera frame]") == 4
    assert len(encode_calls) == 2
    assert f"embedded media truncated at {payload_size / 1_000_000:g} MB (1 embedded)" in document


def test_live_frame_budget_zero_semantics_match_the_full_budget(tmp_path: Path) -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    payload_size = len(png_data_url(frame).partition(",")[2])
    parts = [*_parts("first", 1), *_parts("second", 2)]
    store = FrameStore(str(tmp_path))
    store.put("scene-0-e0", 1, "first", frame)
    store.put("scene-0-e0", 2, "second", frame)
    log = _frame_log(parts)

    full_unlimited = render_html(
        log,
        title="both unlimited",
        frames_dir=tmp_path,
        frames_budget_bytes=0,
        live_frames_budget_bytes=0,
    )
    full_caps_live = render_html(
        log,
        title="full cap",
        frames_dir=tmp_path,
        frames_budget_bytes=payload_size,
        live_frames_budget_bytes=0,
    )
    live_caps_full = render_html(
        log,
        title="live cap",
        frames_dir=tmp_path,
        frames_budget_bytes=0,
        live_frames_budget_bytes=payload_size,
    )

    assert full_unlimited.count('<img class="frame"') == 2
    assert full_caps_live.count('<img class="frame"') == 1
    assert live_caps_full.count('<img class="frame"') == 1


def test_live_effective_budget_uses_tighter_full_limit_and_reports_it(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    parts: list[object] = []
    store = FrameStore(str(tmp_path))
    for step, name in enumerate(("oldest", "middle", "newest"), start=1):
        parts.extend(_parts(name, step))
        store.put(
            "scene-0-e0",
            step,
            name,
            rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8),
        )

    document = render_html(
        _frame_log(parts),
        title="effective min",
        frames_dir=tmp_path,
        frames_budget_bytes=2_000_000,
        live_frames_budget_bytes=8_000_000,
    )

    assert document.count('<img class="frame"') == 2
    assert "camera newest step 3" in document
    assert "camera middle step 2" in document
    assert "camera oldest step 1" not in document
    assert "embedded media truncated at 2 MB (2 embedded)" in document


def test_zero_frame_budget_is_unlimited(tmp_path: Path) -> None:
    parts = [*_parts("first", 1), *_parts("second", 2)]
    for step, name in enumerate(("first", "second"), start=1):
        _save_frame(tmp_path, name, step, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(
        _frame_log(parts), title="unlimited", frames_dir=tmp_path, frames_budget_bytes=0
    )

    assert document.count('<img class="frame"') == 2
    assert document.count("embedded media truncated at") == 0


def test_non_truncated_frame_render_has_no_truncation_chip(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts()), title="fits", frames_dir=tmp_path)

    assert document.count("embedded media truncated at") == 0


def test_non_chat_transcript_never_embeds_frames(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))
    transcript = {"parts": _parts()}

    document = render_html(
        _log(transcripts=(transcript,)),
        title="json",
        frames_dir=tmp_path,
        live_frames_budget_bytes=8_000_000,
    )

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


@pytest.mark.parametrize("role", ["assistant", "tool"])
def test_non_user_chat_messages_never_embed_frames(tmp_path: Path, role: str) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts(), role=role), title="role", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


def _without_pov(document: str) -> str:
    return re.sub(r'<details class="raw-transcript">.*?</details>', "", document, flags=re.DOTALL)


def test_turns_keep_string_users_visible_and_raw_observations_only_in_pov(
    tmp_path: Path,
) -> None:
    state = "state[joint_pos]: <script>bad()</script>"
    first_parts = [*_parts("top", 3), {"type": "text", "text": state}]
    capture_parts = [*_parts("wrist", 3), {"type": "text", "text": "feedback: slower"}]
    transcript = _chat(
        {"role": "assistant", "content": "leading assistant"},
        {"role": "user", "content": "Goal: lift the cube"},
        {"role": "user", "content": first_parts},
        {
            "role": "assistant",
            "content": "I will move carefully.",
            "tool_calls": [
                {
                    "function": {
                        "name": "move_joints",
                        "arguments": json.dumps(
                            {
                                "targets": {"j1": 0.123456, "j2": -0.00004},
                                "speed": 0.5,
                                "note": "edge <close>",
                            }
                        ),
                    }
                },
                {
                    "function": {
                        "name": "take_pic",
                        "arguments": {"cameras": ["top", "wrist"]},
                    }
                },
            ],
        },
        {"role": "tool", "content": "moved exactly"},
        {"role": "user", "content": "Respond with exactly one tool call."},
        {"role": "user", "content": capture_parts},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "move_by",
                        "arguments": {
                            "deltas": {"x": 0.00055},
                            "hindsight": "next time go slower",
                        },
                    }
                }
            ],
        },
    )
    _save_frame(tmp_path, "top", 3, np.zeros((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "wrist", 3, np.ones((2, 2, 3), dtype=np.uint8))

    document = render_html(_log(transcripts=(transcript,)), title="turns", frames_dir=tmp_path)
    human = _without_pov(document)

    assert document.count('class="turn"') == 3
    assert document.count('class="raw-transcript"') == 3
    assert document.count("<summary>Raw transcript</summary>") == 3
    assert document.count('<div class="turn-step">step 3</div>') == 2
    assert "leading assistant" in human and "Goal: lift the cube" in human
    assert "Respond with exactly one tool call." in human
    assert "I will move carefully." in human
    assert "<figcaption>top</figcaption>" in human
    assert "<figcaption>wrist</figcaption>" in human
    assert state not in human and "feedback: slower" not in human
    assert "moved exactly" not in human  # tool results live in the raw transcript only
    assert "state[joint_pos]: &lt;script&gt;bad()&lt;/script&gt;" in document
    assert "feedback: slower" in document and "moved exactly" in document
    assert "j1</span>0.1235" in human and "j2</span>-0.0000" in human
    assert "x</span>0.0006" in human and "speed</span>0.5" in human
    assert "cameras</span>top, wrist" in human
    assert "edge &lt;close&gt;" in human and "next time go slower" in human
    assert '<span class="call-key">note</span>' not in human
    assert '<span class="call-key">hindsight</span>' not in human
    assert "move_joints({&quot;" in document and "moved exactly" in document
    assert "&amp;lt;script&amp;gt;" not in document


def test_pretty_call_malformed_arguments_use_raw_rendering() -> None:
    arguments = '{"targets": <broken>}'
    transcript = _chat(
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "move_joints", "arguments": arguments}},
            ],
        }
    )

    document = render_html(_log(transcripts=(transcript,)), title="raw")

    assert 'class="pretty-call"' not in document
    assert document.count("move_joints({&quot;targets&quot;: &lt;broken&gt;})") == 2


def test_feedback_matches_latest_same_step_turn_and_residual_is_exact(
    tmp_path: Path,
) -> None:
    transcript = _chat(
        {"role": "user", "content": _parts("first", 5)},
        {"role": "assistant", "content": "first response"},
        {"role": "user", "content": _parts("capture", 5)},
        {"role": "assistant", "content": "second response"},
    )
    _save_frame(tmp_path, "first", 5, np.zeros((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "capture", 5, np.ones((2, 2, 3), dtype=np.uint8))
    log = _log(transcripts=(transcript,))
    messages = (
        {"t": 4, "text": "too early", "source": "console"},
        {"t": 5, "text": "same step tail", "source": "voice"},
    )
    scene = dataclasses.replace(log.samples[0], operator_messages=(messages,))

    document = render_html(
        dataclasses.replace(log, samples=(scene,)), title="feedback", frames_dir=tmp_path
    )
    turns = document.split('<section class="turn')[1:]

    assert "same step tail" not in turns[0]
    assert "same step tail" in turns[1]
    assert '<span class="feedback-source">voice</span>' in turns[1]
    residual = document[document.index("<h3>Operator feedback</h3>") :]
    assert "too early" in residual and "same step tail" not in residual
    assert document.index("Operator feedback") > document.index("Trial 0 transcript")


def test_feedback_fully_placed_hides_residual_and_zero_step_chat_keeps_it(
    tmp_path: Path,
) -> None:
    message = {"t": 2, "text": "hold position", "source": "console"}
    transcript = _chat({"role": "user", "content": _parts("top", 2)})
    _save_frame(tmp_path, "top", 2, np.zeros((2, 2, 3), dtype=np.uint8))
    log = _log(transcripts=(transcript,))
    scene = dataclasses.replace(log.samples[0], operator_messages=((message,),))
    placed = render_html(
        dataclasses.replace(log, samples=(scene,)), title="placed", frames_dir=tmp_path
    )

    assert "hold position" in placed
    assert "Operator feedback" not in placed

    zero_step = _log(transcripts=(_chat({"role": "user", "content": "Goal: move"}),))
    zero_scene = dataclasses.replace(zero_step.samples[0], operator_messages=((message,),))
    residual = render_html(dataclasses.replace(zero_step, samples=(zero_scene,)), title="zero")
    assert "Operator feedback" in residual and "hold position" in residual
    assert 'class="feedback-chip"' not in residual


def test_non_chat_feedback_is_residual_and_json_fallback_is_unchanged() -> None:
    transcript = {"policy": ["opaque", {"value": 3}]}
    message = {"t": 1, "text": "script note", "source": "voice"}
    log = _log(transcripts=(transcript,))
    scene = dataclasses.replace(log.samples[0], operator_messages=((message,),))

    document = render_html(dataclasses.replace(log, samples=(scene,)), title="non-chat")

    expected = json.dumps(transcript, indent=2, sort_keys=True, ensure_ascii=False)
    assert f"<pre>{html.escape(expected, quote=True)}</pre>" in document
    assert "Operator feedback" in document and "script note" in document


def test_started_scene_badge_uses_live_marker_without_changing_other_scenes() -> None:
    log = _log(status="started")
    live_scene = dataclasses.replace(
        log.samples[0], scene_id="live", status="success", trial_metadata=({}, {"live": {}})
    )
    done_scene = dataclasses.replace(
        log.samples[0], scene_id="done", status="success", trial_metadata=({}, {})
    )
    started = render_html(dataclasses.replace(log, samples=(live_scene, done_scene)), title="mixed")

    assert 'scene-head"><h2>live</h2><span class="badge status-running">running</span>' in started
    assert (
        'scene-head"><h2>done</h2><span class="badge status-completed">completed</span>' in started
    )

    completed = render_html(
        dataclasses.replace(log, status="success", samples=(live_scene,)), title="complete"
    )
    assert (
        'scene-head"><h2>live</h2><span class="badge status-completed">completed</span>'
        in completed
    )


def test_flipbook_reuses_frame_urls_and_requires_two_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("inspect_robots._html.shutil.which", lambda _name: None)
    transcript = _chat(
        {"role": "user", "content": _parts("top", 1)},
        {"role": "user", "content": _parts("top", 2)},
    )
    _save_frame(tmp_path, "top", 1, np.zeros((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "top", 2, np.ones((2, 2, 3), dtype=np.uint8))

    document = render_html(_log(transcripts=(transcript,)), title="flip", frames_dir=tmp_path)
    sources = re.findall(r'src="(data:image/png;base64,[^"]+)"', document)

    assert len(sources) == 2 and all(document.count(source) == 1 for source in sources)
    assert document.count("document.addEventListener('DOMContentLoaded'") == 1
    assert 'class="run-media" data-trial="scene-0-e0"' in document
    assert 'class="flipbook-player"' in document
    assert (
        'class="flipbook-player"'
        not in re.findall(r'<figure class="frame-cell">.*?</figure>', document, flags=re.DOTALL)[0]
    )
    assert 'data-camera="top" data-step="1" data-trial="scene-0-e0"' in document
    assert "no ffmpeg" in document
    assert 'data-steps="' not in document
    assert '<button type="button" class="camera-tab" data-follow>Follow</button>' not in document
    assert '<section class="turn" data-step="1"><div class="turn-step">step 1</div>' in document
    assert '<section class="turn" data-step="2"><div class="turn-step">step 2</div>' in document

    single = _chat({"role": "user", "content": _parts("top", 1)})
    one = render_html(_log(transcripts=(single,)), title="one", frames_dir=tmp_path)
    assert 'class="run-media"' not in one


def test_mp4_tier_renders_one_composite_in_turn_order_without_autoplay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("inspect_robots._html.shutil.which", lambda _name: "/fake/ffmpeg")
    encoded: list[list[tuple[str, list[int]]]] = []

    def fake_encode(
        streams: Sequence[tuple[str, Sequence[tuple[int, Path]]]], fps: float, ffmpeg: str
    ) -> tuple[bytes, tuple[str, ...], tuple[int, ...]]:
        assert fps == 10.0 and ffmpeg == "/fake/ffmpeg"
        encoded.append([(key, [step for step, _path in frames]) for key, frames in streams])
        return b"composite", tuple(key for key, _frames in streams), (0, 1)

    monkeypatch.setattr("inspect_robots._video._encode_composite_mp4", fake_encode)
    for camera in ("alpha", "top camera", "unseen_camera"):
        _save_frame(tmp_path, camera, 0, np.zeros((2, 2, 3), dtype=np.uint8))
        _save_frame(tmp_path, camera, 1, np.ones((2, 2, 3), dtype=np.uint8))
    transcript = _chat(
        {"role": "assistant", "content": "Preamble"},
        {"role": "user", "content": [*_parts("top camera", 0), *_parts("alpha", 0)]},
        {"role": "user", "content": [*_parts("top camera", 1), *_parts("alpha", 1)]},
    )

    document = render_html(_log(transcripts=(transcript,)), title="mp4", frames_dir=tmp_path)

    assert encoded == [
        [
            (_safe("top camera"), [0, 1]),
            ("alpha", [0, 1]),
            ("unseen_camera", [0, 1]),
        ]
    ]
    assert document.count("data:video/mp4;base64,") == 1
    assert document.count("<video") == 1
    assert 'data-camera-tab="' not in document
    assert '<button type="button" class="camera-tab" data-follow>Follow</button>' in document
    assert '<div class="camera-order">top camera · alpha · unseen_camera</div>' in document
    assert "autoplay" not in document
    assert (
        '<video controls muted loop preload="metadata" data-steps="0,1" data-fps="10"' in document
    )
    assert '<section class="turn"><div class="message assistant">' in document
    assert '<section class="turn" data-step="0"><div class="turn-step">step 0</div>' in document
    assert '<section class="turn" data-step="1"><div class="turn-step">step 1</div>' in document
    run_media = re.search(
        r'<div class="run-media".*?<div class="conversation">', document, flags=re.DOTALL
    )
    assert run_media is not None
    assert "data-camera-panel" not in run_media.group()
    assert " hidden" not in run_media.group()
    summary = "<summary>Trial 0 transcript</summary>"
    assert (
        document.index(summary)
        < document.index('class="run-media"')
        < document.index('class="conversation"')
    )


def test_composite_caption_omits_an_all_empty_stored_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("inspect_robots._html.shutil.which", lambda _name: "/fake/ffmpeg")

    def fake_encode(
        streams: Sequence[tuple[str, Sequence[tuple[int, Path]]]], *_args: object
    ) -> tuple[bytes, tuple[str, ...], tuple[int, ...]]:
        survivors = tuple(
            key for key, frames in streams if any(np.load(path).size for _step, path in frames)
        )
        return b"composite", survivors, (0, 1)

    monkeypatch.setattr("inspect_robots._video._encode_composite_mp4", fake_encode)
    for step in (0, 1):
        _save_frame(tmp_path, "top", step, np.ones((2, 2, 3), dtype=np.uint8))
        _save_frame(tmp_path, "empty", step, np.empty((0,), dtype=np.uint8))
    transcript = _chat(
        {"role": "user", "content": _parts("top", 0)},
        {"role": "user", "content": _parts("top", 1)},
    )

    document = render_html(_log(transcripts=(transcript,)), title="empty", frames_dir=tmp_path)

    assert '<div class="camera-order">top</div>' in document
    assert '<div class="camera-order">top · empty</div>' not in document


def test_mp4_budget_and_failures_degrade_to_flipbook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("inspect_robots._html.shutil.which", lambda _name: "/fake/ffmpeg")
    transcript = _chat(
        {"role": "user", "content": _parts("top", 0)},
        {"role": "user", "content": _parts("top", 1)},
    )
    _save_frame(tmp_path, "top", 0, np.zeros((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "top", 1, np.ones((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "inspect_robots._video._encode_composite_mp4",
        lambda streams, *_args: (
            b"large",
            tuple(key for key, _frames in streams),
            tuple(sorted({step for _key, frames in streams for step, _path in frames})),
        ),
    )

    budget = render_html(
        _log(transcripts=(transcript,)),
        title="budget",
        frames_dir=tmp_path,
        video_budget_bytes=1,
    )
    assert "video budget" in budget and 'class="flipbook-player"' in budget
    assert "data:video/mp4" not in budget

    monkeypatch.setattr("inspect_robots._video._encode_composite_mp4", lambda *_args: None)
    failed = render_html(_log(transcripts=(transcript,)), title="failed", frames_dir=tmp_path)
    assert 'class="flipbook-player"' in failed and "data:video/mp4" not in failed
    warning = capsys.readouterr().err
    assert warning.count("warning: report video unavailable") == 1
    assert "encode failed for scene-0-e0 composite" in warning


def test_mp4_budget_is_sticky_first_wins_and_skips_later_encodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder_calls: list[object] = []

    def fake_encode(
        streams: Sequence[tuple[str, Sequence[tuple[int, Path]]]], *_args: object
    ) -> tuple[bytes, tuple[str, ...], tuple[int, ...]]:
        encoder_calls.append(streams)
        return (
            b"overflow",
            tuple(key for key, _frames in streams),
            tuple(sorted({step for _key, frames in streams for step, _path in frames})),
        )

    monkeypatch.setattr("inspect_robots._html.shutil.which", lambda _name: "/fake/ffmpeg")
    monkeypatch.setattr("inspect_robots._video._encode_composite_mp4", fake_encode)
    transcripts = []
    for epoch, camera in enumerate(("alpha", "beta")):
        transcripts.append(
            _chat(
                {"role": "user", "content": _parts(camera, 0)},
                {"role": "user", "content": _parts(camera, 1)},
            )
        )
        _save_frame(tmp_path, camera, 0, np.zeros((2, 2, 3), dtype=np.uint8), epoch=epoch)
        _save_frame(tmp_path, camera, 1, np.ones((2, 2, 3), dtype=np.uint8), epoch=epoch)

    document = render_html(
        _log(transcripts=tuple(transcripts)),
        title="first wins",
        frames_dir=tmp_path,
        video_budget_bytes=1,
    )

    assert "data:video/mp4;base64," not in document
    assert document.count('class="flipbook-player"') == 2
    assert document.count("video budget") == 2
    # Sticky truncation skips the encode entirely for trials after the overflow.
    assert len(encoder_calls) == 1


def test_single_frame_camera_degrades_to_no_panel_on_denial_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A camera below the two-frame flipbook floor vanishes quietly when its video is denied."""
    monkeypatch.setattr("inspect_robots._html.shutil.which", lambda _name: "/fake/ffmpeg")
    encoder_calls: list[object] = []

    def overflow(
        streams: Sequence[tuple[str, Sequence[tuple[int, Path]]]], *_args: object
    ) -> tuple[bytes, tuple[str, ...], tuple[int, ...]]:
        encoder_calls.append(streams)
        return (
            b"overflow",
            tuple(key for key, _frames in streams),
            tuple(sorted({step for _key, frames in streams for step, _path in frames})),
        )

    monkeypatch.setattr("inspect_robots._video._encode_composite_mp4", overflow)
    first = _chat(
        {"role": "user", "content": _parts("aa", 0)},
        {"role": "user", "content": _parts("aa", 1)},
    )
    second = _chat({"role": "user", "content": _parts("zz", 0)})
    for step in (0, 1):
        _save_frame(tmp_path, "aa", step, np.ones((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "zz", 0, np.ones((2, 2, 3), dtype=np.uint8), epoch=1)

    # Trial 0 overflows; trial 1 skips its encode and has no two-frame fallback.
    document = render_html(
        _log(transcripts=(first, second)),
        title="sticky no panel",
        frames_dir=tmp_path,
        video_budget_bytes=1,
    )
    assert 'data-camera-panel="zz"' not in document
    assert len(encoder_calls) == 1

    # Encode failure: the same sub-two-frame camera cannot fall back to a
    # flipbook either, exercising the panel-less failure branch.
    monkeypatch.setattr("inspect_robots._video._encode_composite_mp4", lambda *_args: None)
    for camera, steps in (("aa", (0, 1)), ("zz", (0,))):
        for step in steps:
            _save_frame(tmp_path, camera, step, np.ones((2, 2, 3), dtype=np.uint8))
    transcript = _chat(
        {"role": "user", "content": [*_parts("aa", 0), *_parts("zz", 0)]},
        {"role": "user", "content": _parts("aa", 1)},
    )
    document = render_html(
        _log(transcripts=(transcript,)),
        title="failure no panel",
        frames_dir=tmp_path,
    )
    assert 'data-camera-panel="zz"' not in document
    assert 'data-camera-panel="aa"' in document


@pytest.mark.parametrize(
    ("status", "serve_pass", "no_video"),
    [("started", False, False), ("success", True, False), ("success", False, True)],
)
def test_suppressed_video_tiers_never_call_encoder(
    status: str,
    serve_pass: bool,
    no_video: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("inspect_robots._html.shutil.which", lambda _name: "/fake/ffmpeg")
    monkeypatch.setattr(
        "inspect_robots._video._encode_composite_mp4",
        lambda *_args: (_ for _ in ()).throw(AssertionError("encoder called")),
    )
    transcript = _chat(
        {"role": "user", "content": _parts("top", 0)},
        {"role": "user", "content": _parts("top", 1)},
    )
    _save_frame(tmp_path, "top", 0, np.zeros((2, 2, 3), dtype=np.uint8))
    _save_frame(tmp_path, "top", 1, np.ones((2, 2, 3), dtype=np.uint8))

    document = render_html(
        _log(status=status, transcripts=(transcript,)),
        title="suppressed",
        frames_dir=tmp_path,
        serve_pass=serve_pass,
        no_video=no_video,
    )

    assert 'class="flipbook-player"' in document
    assert "data:video/mp4" not in document


def test_report_private_degrade_edges_remain_tolerant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unsupported = {"function": {"name": "move", "arguments": {"nested": {"label": "not numeric"}}}}
    assert 'class="call"' in _render_tool_call(unsupported)
    assert 'class="pretty-call"' not in _render_tool_call(unsupported)

    context = _FrameContext(tmp_path, "trial", _FrameBudget(limit=0))
    markup = _frame_markup("data:image/png;base64,AA==", context, "top", 1)
    assert 'data-trial="trial"' in markup

    chat = [{"role": "user", "content": "hello"}]
    assert 'class="raw-transcript"' in _render_transcript(chat, trial_prefix="trial")

    np.save(tmp_path / "trial_bad.npy", np.zeros((1,), dtype=np.uint8))
    assert _trial_camera_streams(tmp_path, "trial") == {}

    video_context = _VideoContext(False, None, 0.0, _VideoBudget())
    _warn_video_degrade(video_context, "first")
    _warn_video_degrade(video_context, "second")
    assert capsys.readouterr().err.count("warning: report video unavailable") == 1


def test_malformed_feedback_and_extra_trial_messages_stay_residual(tmp_path: Path) -> None:
    transcript = _chat({"role": "user", "content": _parts("top", 2)})
    _save_frame(tmp_path, "top", 2, np.zeros((2, 2, 3), dtype=np.uint8))
    log = _log(transcripts=(transcript,))
    scene = dataclasses.replace(
        log.samples[0],
        operator_messages=(
            ({"t": "bad", "text": "malformed time"},),
            ({"t": 9, "text": "missing transcript"},),
        ),
    )

    document = render_html(
        dataclasses.replace(log, samples=(scene,)), title="residual", frames_dir=tmp_path
    )

    assert "Operator feedback" in document
    assert "malformed time" in document and "missing transcript" in document


def test_video_budget_without_flipbook_source_omits_camera_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("inspect_robots._html.shutil.which", lambda _name: "/fake/ffmpeg")
    monkeypatch.setattr(
        "inspect_robots._video._encode_composite_mp4",
        lambda streams, *_args: (
            b"large",
            tuple(key for key, _frames in streams),
            tuple(sorted({step for _key, frames in streams for step, _path in frames})),
        ),
    )
    _save_frame(tmp_path, "unseen", 0, np.zeros((2, 2, 3), dtype=np.uint8))
    transcript = _chat({"role": "user", "content": "no observation frames"})

    document = render_html(
        _log(transcripts=(transcript,)),
        title="no source",
        frames_dir=tmp_path,
        video_budget_bytes=1,
    )

    assert 'class="run-media"' not in document


def test_wire_section_renders_blob_stubs_breakpoints_and_request_summary(
    tmp_path: Path,
) -> None:
    png = b"\x89PNG\r\nwire"
    sha = hashlib.sha256(png).hexdigest()
    row = {
        "call": 0,
        "attempt": 0,
        "endpoint": "/messages",
        "duration_s": 0.125,
        "status": 200,
        "request": {
            "model": "claude-test",
            "reasoning_effort": "high",
            "temperature": 0.2,
            "system": [{"type": "text", "text": "system prompt"}],
            "tools": [{"name": "move", "description": "move safely"}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "[image omitted: context window]",
                            "count": 1,
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": f"$blob:{sha}",
                            },
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
        },
    }
    log, log_path, calls_path = _wire_log(tmp_path, [row], scene_id="scene / 0")
    blob_dir = calls_path.parent.parent / "blobs"
    blob_dir.mkdir()
    (blob_dir / f"{sha}.png").write_bytes(png)

    document = render_html(log, title="wire", log_path=log_path)

    anchor = f"wire-{_safe('scene / 0-e0')}-{sha[:12]}"
    assert "Trial 0 Wire" in document
    assert "/messages" in document
    assert "claude-test" in document and "high" in document and "0.2" in document
    assert "[image omitted: context window]" in document
    assert "cache_control" in document and "ephemeral" in document
    assert f'id="{anchor}"' in document
    assert f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}" in document


def test_wire_media_can_be_elided_without_loading_started_page_blobs(tmp_path: Path) -> None:
    sha = "a" * 64
    row = {"call": 0, "request": {"messages": [{"content": f"$blob:{sha}"}]}}
    log, log_path, _calls_path = _wire_log(tmp_path, [row])

    document = render_html(
        dataclasses.replace(log, status="started"),
        title="live wire",
        log_path=log_path,
        wire_media_elided=True,
    )

    assert "[blob elided: live page]" in document
    assert "[broken blob reference" not in document
    assert "data:image/png;base64" not in document


@pytest.mark.parametrize(
    "mode",
    ["missing-metadata", "deleted", "traversal", "shallow", "empty", "non-object", "malformed"],
)
def test_wire_section_is_absent_for_unavailable_sidecars(tmp_path: Path, mode: str) -> None:
    if mode == "missing-metadata":
        document = render_html(_log(), title="absent", log_path=tmp_path / "run.json")
    elif mode == "traversal":
        log = _log()
        scene = dataclasses.replace(
            log.samples[0],
            trial_metadata=({"wire_capture": "../outside/calls.jsonl"}, {}),
        )
        document = render_html(
            dataclasses.replace(log, samples=(scene,)),
            title="absent",
            log_path=tmp_path / "run.json",
        )
    elif mode == "shallow":
        # A depth-0 pointer resolves inside the log dir, but its derived
        # blobs dir (parent.parent / "blobs") would land one level above it.
        (tmp_path / "calls.jsonl").write_text('{"call": 0}\n', encoding="utf-8")
        log = _log()
        scene = dataclasses.replace(
            log.samples[0],
            trial_metadata=({"wire_capture": "calls.jsonl"}, {}),
        )
        document = render_html(
            dataclasses.replace(log, samples=(scene,)),
            title="absent",
            log_path=tmp_path / "run.json",
        )
    else:
        log, log_path, calls_path = _wire_log(tmp_path, [{"request": {}}])
        if mode == "deleted":
            calls_path.unlink()
        elif mode == "empty":
            calls_path.write_text("", encoding="utf-8")
        elif mode == "non-object":
            calls_path.write_text("[]\n", encoding="utf-8")
        else:
            calls_path.write_text("{\n", encoding="utf-8")
        document = render_html(log, title="absent", log_path=log_path)

    assert "Trial 0 Wire" not in document


def test_wire_section_survives_torn_tail_line(tmp_path: Path) -> None:
    log, log_path, calls_path = _wire_log(tmp_path, [{"call": 0, "request": {}}])
    with calls_path.open("a", encoding="utf-8") as handle:
        handle.write('{"call": 1, "request": {"model": "trunc')

    document = render_html(log, title="torn", log_path=log_path)

    assert "Trial 0 Wire" in document


def test_wire_hostile_and_missing_blob_references_render_broken(tmp_path: Path) -> None:
    uppercase = "A" * 64
    missing = "b" * 64
    row = {
        "call": 0,
        "attempt": 0,
        "endpoint": "/responses",
        "duration_s": "unknown",
        "status": 200,
        "request": {
            "input": [
                {"content": "$blob:../../x"},
                {"content": f"$blob:{uppercase}"},
                {"content": f"$blob:{missing}"},
            ],
            "tools": "malformed",
        },
    }
    log, log_path, _ = _wire_log(tmp_path, [row])

    document = render_html(log, title="hostile", log_path=log_path)

    assert document.count("broken blob reference") == 3
    assert "../../x" in document
    assert "data:image/png;base64," not in document


def test_wire_budget_denial_never_emits_a_dangling_repeat_link(tmp_path: Path) -> None:
    png = b"budgeted"
    sha = hashlib.sha256(png).hexdigest()
    message = {"role": "user", "content": f"data:image/png;base64,$blob:{sha}"}
    rows = [
        {
            "call": call,
            "attempt": 0,
            "endpoint": "/chat/completions",
            "duration_s": 0.1,
            "status": 200,
            "request": {"messages": [message] if call == 0 else [message, message]},
        }
        for call in range(2)
    ]
    log, log_path, calls_path = _wire_log(tmp_path, rows)
    blob_dir = calls_path.parent.parent / "blobs"
    blob_dir.mkdir()
    (blob_dir / f"{sha}.png").write_bytes(png)

    document = render_html(log, title="budget", log_path=log_path, frames_budget_bytes=1)

    assert document.count("[blob elided: media budget]") == 2
    assert f'href="#wire-scene-0-e0-{sha[:12]}"' not in document
    assert "embedded media truncated at" in document


def test_wire_repeat_blob_links_to_the_single_first_embed(tmp_path: Path) -> None:
    png = b"repeat"
    sha = hashlib.sha256(png).hexdigest()
    reference = f"data:image/png;base64,$blob:{sha}"
    rows = [
        {
            "call": 0,
            "attempt": 0,
            "endpoint": "/responses",
            "duration_s": 0.1,
            "status": 200,
            "request": {"input": [{"content": reference}]},
        },
        {
            "call": 1,
            "attempt": 0,
            "endpoint": "/responses",
            "duration_s": 0.2,
            "status": 200,
            "request": {"input": [{"content": reference}, {"content": reference}]},
        },
    ]
    log, log_path, calls_path = _wire_log(tmp_path, rows)
    blob_dir = calls_path.parent.parent / "blobs"
    blob_dir.mkdir()
    (blob_dir / f"{sha}.png").write_bytes(png)

    document = render_html(log, title="repeat", log_path=log_path)

    anchor = f"wire-scene-0-e0-{sha[:12]}"
    assert document.count(f'id="{anchor}"') == 1
    assert document.count(f'href="#{anchor}"') == 1


def test_wire_messages_are_delta_rendered_with_changed_as_sent_form(
    tmp_path: Path,
) -> None:
    old_message = {"role": "user", "content": "original full text"}
    rewritten = {
        "role": "user",
        "content": [{"type": "text", "text": "[image omitted: evicted view]"}],
    }
    new_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "brand-new message"}],
        "cache_control": {"type": "ephemeral"},
    }
    rows = [
        {
            "call": 0,
            "attempt": 0,
            "endpoint": "/messages",
            "duration_s": 0.1,
            "status": 200,
            "request": {"messages": [old_message]},
        },
        {
            "call": 1,
            "attempt": 0,
            "endpoint": "/messages",
            "duration_s": 0.1,
            "status": 200,
            "request": {
                "messages": [rewritten, new_message],
                "reasoning": {"effort": "medium"},
            },
        },
    ]
    log, log_path, _ = _wire_log(tmp_path, rows)

    document = render_html(log, title="delta", log_path=log_path)
    second_call = document[document.index("call 1 attempt 0") :]

    assert "message 0 changed as sent" in second_call
    assert "[image omitted: evicted view]" in second_call
    assert "brand-new message" in second_call
    assert "cache_control" in second_call
    assert "original full text" not in second_call


def test_wire_tools_and_system_render_once_then_only_report_changes(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "call": 0,
            "attempt": 0,
            "endpoint": "/messages",
            "duration_s": 0.1,
            "status": 200,
            "request": {
                "system": "initial-system",
                "tools": [{"name": "initial-tool"}],
                "messages": [],
                "output_config": {"effort": "low"},
            },
        },
        {
            "call": 0,
            "attempt": 1,
            "endpoint": "/messages",
            "duration_s": 0.2,
            "status": 429,
            "request": {
                "system": "initial-system",
                "tools": [{"name": "initial-tool"}],
                "messages": [],
            },
        },
        {
            "call": 1,
            "attempt": 0,
            "endpoint": "/messages",
            "duration_s": 0.3,
            "status": 200,
            "request": {
                "system": "changed-system",
                "tools": [{"name": "changed-tool"}],
                "messages": [],
            },
        },
    ]
    log, log_path, _ = _wire_log(tmp_path, rows)

    document = render_html(log, title="static", log_path=log_path)

    assert document.count("initial-system") == 1
    assert document.count("initial-tool") == 1
    assert "changed-system" not in document
    assert "changed-tool" not in document
    assert document.count("tools changed from the trial") == 1
    assert document.count("system changed from the trial") == 1
    assert "no new messages" in document


def test_wire_malformed_request_degrades_to_an_empty_call(tmp_path: Path) -> None:
    row = {
        "call": 0,
        "attempt": 0,
        "endpoint": "/chat/completions",
        "duration_s": None,
        "status": None,
        "request": [],
    }
    log, log_path, _ = _wire_log(tmp_path, [row])

    document = render_html(log, title="malformed", log_path=log_path)

    assert "Trial 0 Wire" in document
    assert "no new messages" in document
    assert "<dd>n/a</dd>" in document
