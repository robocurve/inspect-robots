"""Pure rendering tests for the self-contained HTML eval-log viewer."""

from __future__ import annotations

import base64
import dataclasses
import re
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots._html import _JSON_STRING_LIMIT, _render_chat_transcript, render_html
from inspect_robots._pngenc import png_data_url
from inspect_robots.frames import _safe
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
        ("started", "started", "status-neutral"),
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
    assert "one trial failed" in document
    assert document.count(">n/a</span>") == 2
    assert "prefers-color-scheme: light" in document
    assert "prefers-color-scheme: dark" in document
    assert "<script" not in document


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
        error=attack,
    )
    results = dataclasses.replace(log.results, metrics={attack: 1.0})

    document = render_html(
        dataclasses.replace(log, samples=(scene,), results=results), title=attack
    )

    assert attack not in document
    assert document.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 8
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
    assert document.count('class="call"') == 4


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

    assert document.count('<div class="message assistant">') == 2
    assert "not a list" not in document
    assert "not a call" not in document
    assert "not a function" not in document
    assert "move({&quot;dx&quot;: 1})" in document
    assert 'class="content"' not in document

    assert _render_chat_transcript(["not a message"]) == '<div class="conversation"></div>'


def test_non_dict_message_falls_back_to_preformatted_json() -> None:
    transcript = [{"role": "user", "content": "hello"}, "not a message"]

    document = render_html(_log(transcripts=(transcript,)), title="fallback")

    assert "<pre>" in document
    assert "not a message" in document
    assert '<div class="conversation">' not in document


def test_media_parts_collapse_to_image_chip() -> None:
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
    assert 'loading="lazy"' in document
    assert 'src="data:image/png;base64,' in document
    assert 'alt="camera cam &lt;wide&gt; step 7"' in document
    assert document.index("caption &lt;before&gt;") < document.index('<img class="frame"')
    assert document.index('<img class="frame"') < document.index("after frame")
    assert "[image omitted: streamed camera frame]" not in document
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
    assert document.count("[image omitted: streamed camera frame]") == 2


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

    assert document.count('<img class="frame"') == 3
    assert document.index("camera left step 1") < document.index("camera top step 2")
    assert document.index("camera top step 2") < document.index("camera right step 3")


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
    assert document.count("[image omitted: streamed camera frame]") == 2
    budget_mb = payload_size / 1_000_000
    assert f"frames truncated at {budget_mb:g} MB (1 embedded)" in document


def test_zero_frame_budget_is_unlimited(tmp_path: Path) -> None:
    parts = [*_parts("first", 1), *_parts("second", 2)]
    for step, name in enumerate(("first", "second"), start=1):
        _save_frame(tmp_path, name, step, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(
        _frame_log(parts), title="unlimited", frames_dir=tmp_path, frames_budget_bytes=0
    )

    assert document.count('<img class="frame"') == 2
    assert document.count("frames truncated at") == 0


def test_non_truncated_frame_render_has_no_truncation_chip(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts()), title="fits", frames_dir=tmp_path)

    assert document.count("frames truncated at") == 0


def test_non_chat_transcript_never_embeds_frames(tmp_path: Path) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))
    transcript = {"parts": _parts()}

    document = render_html(_log(transcripts=(transcript,)), title="json", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document


@pytest.mark.parametrize("role", ["assistant", "tool"])
def test_non_user_chat_messages_never_embed_frames(tmp_path: Path, role: str) -> None:
    _save_frame(tmp_path, "top_cam", 4, np.zeros((2, 2, 3), dtype=np.uint8))

    document = render_html(_frame_log(_parts(), role=role), title="role", frames_dir=tmp_path)

    assert '<img class="frame"' not in document
    assert "[image omitted: streamed camera frame]" in document
