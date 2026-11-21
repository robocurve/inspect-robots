"""IR+ (``toolset="plus"``) end to end through ``eval()``.

Conversations are scripted with httpx.MockTransport: no network, no LLM. The
arms are a test model whose joints are the grasp point's position and the
tool's Euler angles, so IK has an exact answer to check against.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

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
from inspect_robots_agent._plus import (
    DEFAULT_SECONDS,
    PARK_SECONDS,
    SETTLE_SECONDS,
    ArmModel,
    smoothstep,
)

Vec = npt.NDArray[np.float64]
DOWN = np.diag([1.0, -1.0, -1.0])
LABELS = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*(f"j{i}" for i in range(6)), "gripper")
)
SETTLE = round(SETTLE_SECONDS * 10)


class _TestArm:
    """Joints 0-2 are the grasp point (m), joints 3-5 the tool's ZYX Euler angles (rad) on DOWN."""

    joint_low = np.array([-0.5, -0.5, -0.5, -3.0, -3.0, -3.0])
    joint_high = np.array([0.5, 0.5, 0.5, 3.0, 3.0, 3.0])

    def fk(self, q: Vec) -> tuple[Vec, Vec]:
        q = np.asarray(q, dtype=np.float64)
        return q[:3].copy(), Rotation.from_euler("ZYX", q[3:6]).as_matrix() @ DOWN


def _kinematics(info: EmbodimentInfo) -> Mapping[str, ArmModel]:
    return {"left": _TestArm(), "right": _TestArm()}


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


def _start() -> Vec:
    arm = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    return np.asarray(np.r_[arm, arm * np.array([1, -1, 1, 1, 1, 1, 1])], dtype=np.float64)


class _Arms:
    """Two six-joint arms with grippers; the joints follow each command at once."""

    def __init__(self, mode: str = "joint_pos") -> None:
        self.q = _start()
        self.actions: list[Vec] = []
        low = np.tile(np.r_[_TestArm.joint_low, 0.0], 2)
        high = np.tile(np.r_[_TestArm.joint_high, 1.0], 2)
        max_step = tuple(0.1 if label.endswith("gripper") else None for label in LABELS)
        self.info = EmbodimentInfo(
            name="arms",
            action_space=Box(
                shape=(14,),
                low=low,
                high=high,
                semantics=ActionSemantics(
                    mode,  # type: ignore[arg-type]
                    rotation_repr="none",
                    dim_labels=LABELS,
                    max_step=max_step,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(name="top", height=2, width=2, channels=3),),
                state=StateSpec(fields=(StateField(key="joint_pos", shape=(14,)),)),
            ),
            control_hz=10.0,
            is_simulated=True,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self.q = _start()
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
            images={"top": frame},
            state={"joint_pos": self.q.copy(), "joint_eff": np.full(14, 0.5)},
            instruction=instruction,
        )


def _policy(script: _Script, **kwargs: Any) -> LLMAgentPolicy:
    kwargs.setdefault("kinematics", _kinematics)
    return LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(script),
        env={},
        toolset="plus",
        clock=lambda: 100.0,
        **kwargs,
    )


def _run(script: _Script, arms: _Arms, tmp_path: Path, max_steps: int = 400) -> None:
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


def test_tool_list_and_schemas() -> None:
    policy = _policy(_Script(_calls(_DONE)))
    policy.bind(_Arms().info)
    assert policy._adapter is not None and policy._plus is not None
    library = policy._plus_library(policy._adapter, policy._plus)
    assert library.names == (
        "go",
        "move_joints",
        "grip",
        "park",
        "state",
        "look",
        "wait",
        "sequence",
        "done",
        "give_up",
    )
    schemas = {s["function"]["name"]: s["function"] for s in library.schemas()}
    go = schemas["go"]["parameters"]
    assert go["required"] == ["arm", "x", "y", "z"]
    assert set(go["properties"]) == {
        "arm",
        "x",
        "y",
        "z",
        "yaw",
        "tilt",
        "roll",
        "rotation",
        "gripper",
        "seconds",
        "note",
    }
    assert go["properties"]["arm"]["enum"] == ["left", "right"]
    assert schemas["move_joints"]["parameters"]["properties"]["joints"]["minItems"] == 6
    step = schemas["sequence"]["parameters"]["properties"]["steps"]["items"]
    assert step["properties"]["tool"]["enum"] == [
        "go",
        "move_joints",
        "grip",
        "park",
        "state",
        "look",
        "wait",
    ]


def test_go_solves_ik_once_and_eases_the_joints(tmp_path: Path) -> None:
    script = _Script(
        _calls(("go", {"arm": "left", "x": 0.4, "y": 0.1, "z": 0.05, "yaw": 90, "tilt": 30})),
        _calls(_DONE),
    )
    arms = _Arms()
    _run(script, arms, tmp_path)
    # The largest joint change is yaw's pi/2 rad: 1.65 s at 0.95 rad/s, over the default 1.3 s.
    steps = round(np.pi / 2 / 0.95 * 10)
    assert len(arms.actions) == steps + SETTLE + 1  # the move, its settle, done's hold
    final = arms.actions[steps - 1][:6]
    rotation = Rotation.from_euler("ZYX", final[3:]).as_matrix() @ DOWN
    expected = Rotation.from_euler("ZYX", [90, -30, 0], degrees=True).as_matrix() @ DOWN
    assert np.allclose(final[:3], [0.4, 0.1, 0.05], atol=1e-4)
    assert np.allclose(rotation, expected, atol=1e-4)
    start = _start()[:6]
    path = np.array([a[:6] for a in arms.actions[:steps]])
    eased = start + np.outer(smoothstep(np.arange(1, steps + 1) / steps), final - start)
    assert np.allclose(path, eased)  # every joint together, slow, fast, slow
    assert all(np.allclose(a[7:], _start()[7:]) for a in arms.actions)  # right arm held
    result = _tool_messages(script.requests[1])[0]
    assert result.startswith(
        f"left arm moved over {steps / 10:.1f}s; now grasp point (+0.400, +0.100, +0.050) m"
    )
    assert "yaw +90.0, tilt +30.0, roll +0.0 deg; gripper 1.000 (0 mm from the target)" in result


def test_rotation_matrix_and_default_duration(tmp_path: Path) -> None:
    script = _Script(
        _calls(("go", {"arm": "right", "x": 0.3, "y": -0.05, "z": 0.2, "rotation": DOWN.tolist()})),
        _calls(_DONE),
    )
    arms = _Arms()
    _run(script, arms, tmp_path)
    steps = round(DEFAULT_SECONDS * 10)  # a 5 cm move: the default 1.3 s
    assert len(arms.actions) == steps + SETTLE + 1
    assert np.allclose(arms.actions[-1][7:13], [0.3, -0.05, 0.2, 0, 0, 0], atol=1e-4)


def test_an_unreachable_target_moves_nothing(tmp_path: Path) -> None:
    script = _Script(_calls(("go", {"arm": "left", "x": 0.9, "y": 0.0, "z": 0.2})), _calls(_DONE))
    arms = _Arms()
    _run(script, arms, tmp_path)
    error = _tool_messages(script.requests[1])[0]
    assert re.match(
        r"unreachable; the closest solution puts the grasp point 400 mm and 0 deg", error
    )
    assert error.endswith("at (+0.500, +0.000, +0.200); nothing moved")
    assert len(arms.actions) == 1  # only done's hold


def test_grip_ramps_at_the_gripper_limit_and_park_returns(tmp_path: Path) -> None:
    script = _Script(
        _calls(
            ("grip", {"arm": "left", "opening": 0.0}),
            ("move_joints", {"arm": "left", "joints": [0.1, 0.2, 0.1, 0.0, 0.5, 0.0]}),
            ("state", {"arm": "left"}),
            ("park", {"arm": "left"}),
        ),
        _calls(_DONE),
    )
    arms = _Arms()
    _run(script, arms, tmp_path)
    results = _tool_messages(script.requests[1])
    # 1.0 to 0.0 at 0.1 per step: 10 steps, longer than grip's 0.5 s.
    assert [a[6] for a in arms.actions[:10]] == pytest.approx([0.9 - 0.1 * i for i in range(10)])
    assert results[0] == "left gripper commanded 0.00; reads 0.000 after 1.0s"
    assert results[1].startswith(
        "left arm moved over 1.3s; now joints [+0.100, +0.200, +0.100, +0.000, +0.500, +0.000]"
    )
    assert results[2].startswith("left: joints [+0.100") and "joint effort [+0.50" in results[2]
    assert "tilt -28.6" in results[2]  # pitch +0.5 rad leans the tip back (-x)
    parked = round(PARK_SECONDS * 10)
    assert results[3].startswith(f"left arm parked over {parked / 10:.1f}s at its start pose")
    assert np.allclose(arms.actions[-2][:7], _start()[:7])  # gripper open again too


def test_a_sequence_runs_like_separate_calls_and_answers_once(tmp_path: Path) -> None:
    steps = [
        {"tool": "go", "args": {"arm": "left", "x": 0.35, "y": 0.0, "z": 0.1}},
        {"tool": "look", "args": {"cameras": ["top"]}},
        {"tool": "grip", "arm": "left", "opening": 0.5},  # args inline
        {"tool": "state"},
        {"tool": "wait", "args": {"seconds": 0.5}},
    ]
    script = _Script(
        _calls(
            ("sequence", {"steps": steps, "note": "pick"}),
            ("go", {"arm": "right", "x": 0.3, "y": 0.0, "z": 0.2}),
        ),
        _calls(_DONE),
    )
    arms = _Arms()
    _run(script, arms, tmp_path)
    assert len(script.requests) == 2
    move = round(DEFAULT_SECONDS * 10) + SETTLE
    assert len(arms.actions) == move + 5 + 5 + move + 1
    results = _tool_messages(script.requests[1])
    assert len(results) == 2  # one answer for the sequence, one for the go after it
    lines = results[0].split("\n")
    assert [line.split(":")[0] for line in lines if line[:1].isdigit()] == [
        "1. go",
        "2. look",
        "3. grip",
        "4. state",
        "5. wait",
    ]
    assert lines[0].startswith(
        "1. go: left arm moved over 1.3s; now grasp point (+0.350, +0.000, +0.100)"
    )
    assert "3. grip: left gripper commanded 0.50; reads 0.500 after 0.5s" in lines
    assert "5. wait: waited 0.5s" in lines
    assert results[1].startswith("right arm moved over 1.3s")
    observation = script.requests[1]["messages"][-1]["content"]
    texts = [p["text"] for p in observation if p["type"] == "text"]
    assert any(t.startswith(f"From look at step {move}") for t in texts)


def test_a_failing_step_skips_the_rest_of_its_sequence_and_turn(tmp_path: Path) -> None:
    steps = [
        {"tool": "go", "args": {"arm": "left", "x": 0.35, "y": 0.0, "z": 0.1}},
        {"tool": "grip", "args": {"arm": "middle", "opening": 0}},
        {"tool": "park", "args": {"arm": "left"}},
    ]
    script = _Script(_calls(("sequence", {"steps": steps}), ("state", {})), _calls(_DONE))
    arms = _Arms()
    _run(script, arms, tmp_path)
    results = _tool_messages(script.requests[1])
    lines = results[0].split("\n")
    assert lines[1] == "2. grip: arm must be one of: left, right"
    assert lines[2] == "3. park: skipped: an earlier call in this turn failed"
    assert results[1] == "skipped: an earlier call in this turn failed"
    assert arms.actions[-1][:3] == pytest.approx([0.35, 0.0, 0.1], abs=1e-4)


def test_a_malformed_sequence_fails_as_one_call(tmp_path: Path) -> None:
    state = _calls(("state", {}))
    bad: list[dict[str, Any]] = [
        {"steps": []},
        {"steps": [{"tool": "done"}]},
        {"steps": [{"tool": "go", "args": [1]}]},
        {"steps": ["go"]},
        {"steps": [{"tool": "state"}] * 21},
    ]
    responses = []
    for arguments in bad:
        responses += [_calls(("sequence", arguments)), state]
    script = _Script(*responses, _calls(_DONE))
    _run(script, _Arms(), tmp_path)
    errors = [_tool_messages(script.requests[2 * i + 1])[-1] for i in range(len(bad))]
    assert errors[0] == "steps must be a non-empty list of {tool, args} objects"
    assert errors[1].startswith("step 1: tool must be one of: go, move_joints, grip, park")
    assert errors[2] == "step 1: args must be an object"
    assert errors[3] == "step 1 must be an object with `tool` and `args`"
    assert errors[4] == "a sequence holds at most 20 steps; got 21"


def test_arguments_are_validated(tmp_path: Path) -> None:
    state = _calls(("state", {}))  # a turn that works, so failures never pile up to an abort
    bad: list[tuple[str, dict[str, Any]]] = [
        ("go", {"arm": "left", "x": 0.3, "y": 0.0}),
        ("go", {"arm": "left", "x": 0.3, "y": 0.0, "z": 0.1, "rotation": [[1, 0], [0, 1]]}),
        ("go", {"arm": "left", "x": 0.3, "y": 0.0, "z": 0.1, "seconds": 25}),
        ("move_joints", {"arm": "left", "joints": [0.1, 0.2]}),
        ("move_joints", {"arm": "left", "joints": [0.9, 0, 0, 0, 0, 0]}),
        ("grip", {"arm": "left", "opening": "wide"}),
        ("look", {"cameras": ["side"]}),
        ("state", {"arm": "middle"}),
    ]
    responses = []
    for call in bad:
        responses += [_calls(call), state]
    script = _Script(*responses, _calls(_DONE))
    _run(script, _Arms(), tmp_path)
    errors = [_tool_messages(script.requests[2 * i + 1])[-1] for i in range(len(bad))]
    assert errors == [
        "z is required",
        "rotation must be a 3x3 matrix (rows)",
        "that move would take 25.0s, over the 20s limit for one move; nothing moved",
        "joints must be 6 angles (rad), in the arm's joint order",
        "joint 0 target +0.900 is outside its limits [-0.500, +0.500]; nothing moved",
        "opening must be a number, 0 closed .. 1 open",
        "unknown camera 'side'; available now: 'top'",
        "arm must be one of: left, right",
    ]


def test_system_prompt_allows_several_calls_per_turn() -> None:
    policy = _policy(_Script(_calls(_DONE)))
    policy.bind(_Arms().info)
    policy.reset(Scene(id="s", instruction="x", init_seed=0))
    system = policy._messages[0]["content"]
    assert "several tool calls in one turn" in system and "optional `note`" in system
    assert "`sequence` holds several steps in one call" in system
    assert "exactly one tool call per turn" not in system
    assert "Every move tool call must include" not in system
    assert "re-check the observation after every motion" not in system


def test_configuration_errors() -> None:
    with pytest.raises(ConfigError, match="toolset must be one of"):
        LLMAgentPolicy(model="test/model", base_url="http://x/v1", env={}, toolset="minus")
    with pytest.raises(ConfigError, match="does not combine with images"):
        _policy(_Script(_calls(_DONE)), images="on_demand")
    with pytest.raises(ConfigError, match="needs it"):
        _policy(_Script(_calls(_DONE)), kinematics=None)
    with pytest.raises(ConfigError, match="supports wire"):
        LLMAgentPolicy(
            model="google/gemini-3.7-flash",
            env={"GEMINI_API_KEY": "k"},
            wire="interactions",
            toolset="plus",
            kinematics=_kinematics,
        )
    with pytest.raises(ConfigError, match="commands joint positions"):
        _policy(_Script(_calls(_DONE))).bind(_Arms("eef_abs_pose").info)
    with pytest.raises(ConfigError, match="commands joint positions"):
        _policy(_Script(_calls(_DONE))).bind(CubePickEmbodiment().info)
    with pytest.raises(ConfigError, match="its model 6 joints"):
        _policy(_Script(_calls(_DONE)), kinematics=lambda info: {"up": _TestArm()}).bind(
            _Arms().info
        )
    with pytest.raises(ConfigError, match="must be 'module:factory'"):
        _policy(_Script(_calls(_DONE)), kinematics="nope").bind(_Arms().info)
    policy = _policy(_Script(_calls(_DONE)), kinematics="test_plus:_kinematics")
    assert policy.config.toolset == "plus"  # type: ignore[attr-defined]
    assert policy.config.kinematics == "test_plus:_kinematics"  # type: ignore[attr-defined]
