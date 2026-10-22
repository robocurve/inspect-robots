"""Operator End via the supervisor's end file: stop waiting on the model, never use its reply."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent.policy import END_FILE_ENV


def _tool_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ]
    }


_MOVE = _tool_response("move_by", {"deltas": {"dx": 0.05, "dy": 0.05}, "note": "approach the cube"})
_DONE = _tool_response("done", {"summary": "close enough"})


class _BlockingModel:
    """Mock model server whose ``block_on``-th request waits until ``release`` is set."""

    def __init__(self, responses: list[dict[str, Any]], *, block_on: int = 1) -> None:
        self.responses = responses
        self.block_on = block_on
        self.requests = 0
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        index = self.requests
        if index == self.block_on:
            self.waiting.set()
            assert self.release.wait(10)
        payload = self.responses[min(index, len(self.responses)) - 1]
        if index == self.block_on:
            self.finished.set()
        return httpx.Response(200, json=payload)


class _Records(NullSink):
    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def on_trial_end(self, record: TrialRecord) -> None:
        self.records.append(record)


def _policy(model: _BlockingModel, end_file: Path | None) -> LLMAgentPolicy:
    env = {} if end_file is None else {END_FILE_ENV: str(end_file)}
    return LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(model),
        env=env,
        transcript_echo=False,
    )


def _task(epochs: int = 1) -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s0", instruction="reach the cube", init_seed=0)],
        scorer=success_at_end(),
        max_steps=20,
        epochs=epochs,
    )


def _end_when(event: threading.Event, end_file: Path) -> threading.Thread:
    """Create the end file once ``event`` fires, as the dashboard's End would."""

    def run() -> None:
        assert event.wait(10)
        end_file.touch()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_end_during_model_wait_holds_and_never_uses_the_reply(tmp_path: Path) -> None:
    end_file = tmp_path / "end"
    model = _BlockingModel([_MOVE])
    sink = _Records()
    embodiment = CubePickEmbodiment()
    ender = _end_when(model.waiting, end_file)

    started = time.monotonic()
    logs = ir_eval(_task(), _policy(model, end_file), embodiment, sinks=[sink])
    elapsed = time.monotonic() - started
    ender.join(5)

    (record,) = sink.records
    assert record.termination_reason == "operator_end"
    assert elapsed < 5, "End must not wait for the blocked model reply"
    (step,) = record.steps
    # The single step is the hold-still stop (a zero delta here): nothing moved.
    np.testing.assert_allclose(
        step.result.observation.state["eef_pos"], step.observation.state["eef_pos"]
    )
    assert step.action.meta["request_stop"] is True
    # The late move reply is never appended to the transcript.
    (transcript,) = logs[0].samples[0].policy_transcripts
    assert transcript is not None
    assert not any(entry.get("role") == "assistant" for entry in transcript)
    model.release.set()


def test_end_before_request_skips_the_model(tmp_path: Path) -> None:
    end_file = tmp_path / "end"
    model = _BlockingModel([_DONE], block_on=0)
    policy = _policy(model, end_file)
    embodiment = CubePickEmbodiment()
    policy.bind(embodiment.info)
    policy.reset(_task().scenes[0])
    end_file.touch()

    chunk = policy.act(embodiment.reset(_task().scenes[0], seed=0))

    assert model.requests == 0
    assert chunk.actions[0].meta["stop_reason"] == "operator_end"


def test_reset_discards_an_end_from_an_earlier_trial(tmp_path: Path) -> None:
    end_file = tmp_path / "end"
    end_file.touch()
    model = _BlockingModel([_DONE], block_on=0)
    sink = _Records()

    ir_eval(_task(), _policy(model, end_file), CubePickEmbodiment(), sinks=[sink])

    assert not end_file.exists()
    assert sink.records[0].termination_reason == "done"
    assert model.requests == 1


def test_next_trial_waits_for_the_abandoned_request_then_runs_normally(tmp_path: Path) -> None:
    end_file = tmp_path / "end"
    model = _BlockingModel([_MOVE, _DONE])
    sink = _Records()
    ender = _end_when(model.waiting, end_file)
    # Let the abandoned request finish only after trial 1 has ended.
    releaser = threading.Thread(target=lambda: (time.sleep(0.3), model.release.set()), daemon=True)
    releaser.start()

    ir_eval(_task(epochs=2), _policy(model, end_file), CubePickEmbodiment(), sinks=[sink])
    ender.join(5)
    releaser.join(5)

    first, second = sink.records
    assert first.termination_reason == "operator_end"
    # The abandoned reply finished before trial 2 reused the client; trial 2 is unaffected.
    assert model.finished.is_set()
    assert second.termination_reason == "done"


def test_configured_end_file_without_end_behaves_normally(tmp_path: Path) -> None:
    model = _BlockingModel([_DONE], block_on=0)
    sink = _Records()

    ir_eval(_task(), _policy(model, tmp_path / "end"), CubePickEmbodiment(), sinks=[sink])

    assert sink.records[0].termination_reason == "done"


def test_model_errors_surface_unchanged_through_the_background_request(tmp_path: Path) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(failing),
        env={END_FILE_ENV: str(tmp_path / "end")},
        transcript_echo=False,
    )
    embodiment = CubePickEmbodiment()
    policy.bind(embodiment.info)
    scene = _task().scenes[0]
    policy.reset(scene)
    observation = embodiment.reset(scene, seed=0)
    direct = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(failing),
        env={},
        transcript_echo=False,
    )
    direct.bind(embodiment.info)
    direct.reset(scene)

    with pytest.raises(Exception) as threaded:
        policy.act(observation)
    with pytest.raises(Exception) as unthreaded:
        direct.act(observation)

    assert type(threaded.value) is type(unthreaded.value)
    assert str(threaded.value) == str(unthreaded.value)
