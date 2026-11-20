"""IR+ (``toolset="plus"``) end to end through ``eval()``.

Conversations are scripted with httpx.MockTransport: no network, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.task import Task
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent._plus import DEFAULT_SECONDS, GRIP_SECONDS, PARK_SECONDS, smoothstep


def _calls(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for i, (name, args) in enumerate(calls)
                    ],
                }
            }
        ]
    }


_DONE = ("done", {"summary": "ok", "hindsight": "none"})


class _Script:
    """Serves queued responses (the last one repeats) and keeps every request body."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.queue = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        payload = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        return httpx.Response(200, json=payload)


class _Arms:
    """Two one-dimensional arms with grippers; the state follows each command at once."""

    def __init__(self, mode: str = "eef_abs_pose") -> None:
        axis = "x" if mode == "eef_abs_pose" else "j"
        self.labels = (f"left_{axis}", "left_gripper", f"right_{axis}", "right_gripper")
        self.key = "eef_state" if mode == "eef_abs_pose" else "joint_pos"
        self.q = np.array([0.0, 1.0, 0.0, 1.0])
        self.actions: list[np.ndarray] = []
        self.info = EmbodimentInfo(
            name="arms",
            action_space=Box(
                shape=(4,),
                low=np.array([-1.0, 0.0, -1.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0]),
                semantics=ActionSemantics(mode, rotation_repr="none", dim_labels=self.labels),
            ),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(name="top", height=2, width=2, channels=3),),
                state=StateSpec(fields=(StateField(key=self.key, shape=(4,)),)),
            ),
            control_hz=10.0,
            is_simulated=True,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.q = np.array([0.0, 1.0, 0.0, 1.0])
        self.actions = []
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        self.q = np.asarray(action.data, dtype=np.float64).copy()
        self.actions.append(self.q.copy())
        return StepResult(observation=self._observe(None))

    def close(self) -> None:
        return None

    def _observe(self, instruction: str | None) -> Observation:
        frame = np.full((2, 2, 3), len(self.actions) % 255, dtype=np.uint8)
        return Observation(
            images={"top": frame}, state={self.key: self.q.copy()}, instruction=instruction
        )


def _policy(script: _Script, **kwargs: Any) -> LLMAgentPolicy:
    # Half of each range per second: a full gripper close is 2 s at the speed limit.
    kwargs.setdefault("max_speed_frac", 0.5)
    return LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(script),
        env={},
        toolset="plus",
        clock=lambda: 100.0,
        **kwargs,
    )


def _run(script: _Script, arms: _Arms, tmp_path: Path, max_steps: int = 200) -> None:
    task = Task(
        name="t",
        scenes=[Scene(id="s0", instruction="tidy up", init_seed=0)],
        scorer=success_at_end(),
        max_steps=max_steps,
    )
    logs = ir_eval(task, _policy(script), arms, log_dir=str(tmp_path))
    assert logs[0].status == "success", logs[0].error


def _tool_messages(request: dict[str, Any]) -> list[str]:
    return [m["content"] for m in request["messages"] if m.get("role") == "tool"]


def test_tool_list_for_cartesian_and_joint_spaces() -> None:
    for mode, move in (("eef_abs_pose", "go"), ("joint_pos", "move_joints")):
        policy = _policy(_Script(_calls(_DONE)))
        policy.bind(_Arms(mode).info)
        assert policy._adapter is not None and policy._plus is not None
        library = policy._plus_library(policy._adapter, policy._plus)
        assert library.names == (move, "grip", "park", "state", "look", "wait", "done", "give_up")
        schemas = {s["function"]["name"]: s["function"] for s in library.schemas()}
        assert schemas["grip"]["parameters"]["properties"]["gripper"]["enum"] == [
            "left_gripper",
            "right_gripper",
        ]
        assert schemas["park"]["parameters"]["properties"]["arm"]["enum"] == ["left", "right"]
        assert schemas[move]["parameters"]["required"] == ["targets"]


def test_a_turn_runs_its_calls_in_order_and_the_model_sees_one_observation(
    tmp_path: Path,
) -> None:
    script = _Script(
        _calls(
            ("go", {"targets": {"left_x": 0.5}, "note": "reach"}),
            ("look", {"cameras": ["top"]}),
            ("grip", {"gripper": "left_gripper", "opening": 0.0}),
            ("state", {}),
            ("go", {"targets": {"right_x": -0.5}, "seconds": 2.0}),
            ("wait", {"seconds": 0.5}),
        ),
        _calls(_DONE),
    )
    arms = _Arms()
    _run(script, arms, tmp_path)
    assert len(script.requests) == 2  # the whole sequence took one model call
    # 0.5 is 6 steps at the limit (with the core's headroom), 9 eased: the default 1.3 s wins.
    first_move = round(DEFAULT_SECONDS * 10)
    # A full close is 21 steps at the limit, 32 eased: longer than grip's own 0.5 s.
    grip = 32
    assert grip > GRIP_SECONDS * 10
    assert len(arms.actions) == first_move + grip + 20 + 5 + 1  # moves, wait, done
    left = [a[0] for a in arms.actions[:first_move]]
    expected = 0.5 * smoothstep(np.arange(1, first_move + 1) / first_move)
    assert np.allclose(left, expected)  # eased: slow, fast, slow
    assert arms.actions[first_move + grip - 1][1] == 0.0  # gripped
    assert arms.actions[-1][0] == 0.5 and arms.actions[-1][2] == -0.5  # done holds the setpoint
    results = _tool_messages(script.requests[1])
    assert (
        results[0].startswith("executing go over 13 steps (1.3s)")
        and "Target reached" in results[0]
    )
    assert results[1] == "captured 'top'; the frames follow"
    assert results[2].startswith("executing grip over 32 steps (3.2s)")
    assert "eef_state" in results[3] and "setpoint" in results[3]
    assert results[4].startswith("executing go over 20 steps (2.0s)")
    assert results[5].startswith("waiting 0.5 s (5 steps)") and "Slept 0.5 s" in results[5]
    observation = script.requests[1]["messages"][-1]["content"]
    texts = [p["text"] for p in observation if p["type"] == "text"]
    assert texts[0].startswith("Current observation.")
    assert any(t.startswith("From look at step 13") for t in texts)


def test_a_failed_call_skips_the_rest_of_the_turn(tmp_path: Path) -> None:
    script = _Script(
        _calls(("go", {"targets": {"nope": 1.0}}), ("go", {"targets": {"left_x": 0.3}})),
        _calls(_DONE),
    )
    arms = _Arms()
    _run(script, arms, tmp_path)
    results = _tool_messages(script.requests[1])
    assert results[0].startswith("unknown dimension 'nope'")
    assert results[1] == "skipped: an earlier call in this turn failed"
    assert script.requests[1]["messages"][-1]["content"][0]["text"].startswith(
        "The turn moved nothing"
    )
    assert len(arms.actions) == 1  # only done's hold


def test_done_mid_turn_ends_the_trial_after_the_calls_before_it(tmp_path: Path) -> None:
    script = _Script(
        _calls(("go", {"targets": {"left_x": 0.2}}), _DONE, ("go", {"targets": {"left_x": 0.9}}))
    )
    arms = _Arms()
    _run(script, arms, tmp_path)
    assert len(script.requests) == 1
    assert arms.actions[-1][0] == pytest.approx(0.2)


def test_park_returns_one_arm_to_where_the_trial_started(tmp_path: Path) -> None:
    script = _Script(
        _calls(("move_joints", {"targets": {"left_j": 0.5, "right_j": 0.4}})),
        _calls(("park", {"arm": "left"})),
        _calls(_DONE),
    )
    arms = _Arms("joint_pos")
    _run(script, arms, tmp_path)
    parked = arms.actions[-2]
    assert parked[0] == 0.0 and parked[2] == pytest.approx(0.4)
    assert _tool_messages(script.requests[2])[-1].startswith(
        f"executing park over {round(PARK_SECONDS * 10)} steps"
    )


def test_seconds_are_validated(tmp_path: Path) -> None:
    state = _calls(("state", {}))  # a turn that works, so failures never pile up to an abort
    script = _Script(
        _calls(("go", {"targets": {"left_x": 0.2}, "seconds": -1})),
        state,
        _calls(("go", {"targets": {"left_x": 0.2}, "seconds": 11})),
        state,
        _calls(("look", {"cameras": ["side"]})),
        state,
        _calls(("grip", {"gripper": "middle", "opening": 0})),
        _calls(_DONE),
    )
    _run(script, _Arms(), tmp_path)
    errors = [_tool_messages(script.requests[i])[-1] for i in (1, 3, 5, 7)]
    assert errors[0] == "seconds must be a positive number of seconds when given"
    assert errors[1] == "seconds 11 is over the 10 s limit for one move"
    assert errors[2].startswith("unknown camera 'side'")
    assert errors[3] == "gripper must be one of: left_gripper, right_gripper"


def test_system_prompt_allows_several_calls_per_turn() -> None:
    policy = _policy(_Script(_calls(_DONE)))
    policy.bind(_Arms().info)
    policy.reset(Scene(id="s", instruction="x", init_seed=0))
    system = policy._messages[0]["content"]
    assert "several tool calls in one turn" in system and "optional `note`" in system
    assert "exactly one tool call per turn" not in system
    assert "Every move tool call must include" not in system


def test_configuration_errors() -> None:
    with pytest.raises(ConfigError, match="toolset must be one of"):
        LLMAgentPolicy(model="test/model", base_url="http://x/v1", env={}, toolset="minus")
    with pytest.raises(ConfigError, match="does not combine with images"):
        _policy(_Script(_calls(_DONE)), images="on_demand")
    with pytest.raises(ConfigError, match="supports wire"):
        LLMAgentPolicy(
            model="google/gemini-3.7-flash",
            env={"GEMINI_API_KEY": "k"},
            wire="interactions",
            toolset="plus",
        )
    policy = _policy(_Script(_calls(_DONE)))
    with pytest.raises(ConfigError, match="needs an absolute action space"):
        policy.bind(CubePickEmbodiment().info)
    assert policy.config.toolset == "plus"
