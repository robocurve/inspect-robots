"""Pure rendering tests for the self-contained HTML eval-log viewer."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from inspect_robots._html import _render_chat_transcript, render_html
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
    assert "Reduced scores" in document and "Epoch scores" in document
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
    assert "Epoch scores" not in document
    assert "Termination reasons" not in document
    assert "Operator judgements" not in document


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
