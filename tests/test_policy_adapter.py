"""The reference policy adapter: tools, instructions and observation text from the embodiment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from inspect_robots import eval as ir_eval
from inspect_robots.errors import ConfigError
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.policy_adapter import (
    PolicyAdapter,
    ToolAdapter,
    approvals_line,
    embodiment_notes,
    extra_lines,
    observation_lines,
    operator_lines,
    state_lines,
)
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace, StateField, StateSpec
from inspect_robots.task import Task
from inspect_robots.tools import Tool, ToolError, ToolLibrary, ToolResult
from inspect_robots.types import Action, ActionChunk, Observation


def _info(docs: str | None = "Joint 0 is the shoulder.", pinned: bool = False) -> Any:
    high = np.array([1.0, 0.0 if pinned else 1.0])
    return replace(
        CubePickEmbodiment().info,
        name="arm",
        docs=docs,
        action_space=Box(
            shape=(2,),
            low=np.array([-1.0, 0.0 if pinned else -1.0]),
            high=high,
            semantics=ActionSemantics("joint_pos", dim_labels=("shoulder", "elbow")),
        ),
        observation_space=ObservationSpace(
            state=StateSpec(fields=(StateField(key="joint_pos", shape=(2,)),))
        ),
        control_hz=10.0,
    )


def _wave(arguments: dict[str, Any], observation: Observation) -> ToolResult:
    return ToolResult(chunk=ActionChunk(actions=[Action(data=np.zeros(2))]), note="waved")


_WAVE = Tool(
    "wave",
    {"type": "function", "function": {"name": "wave", "description": "Wave.", "parameters": {}}},
    _wave,
)


def test_the_adapter_satisfies_the_protocol_and_lists_tools_in_order() -> None:
    adapter = ToolAdapter()
    adapter.bind(_info())
    assert isinstance(adapter, PolicyAdapter)
    assert adapter.tools.names == ("move_joints", "done", "give_up", "sleep")
    assert [tool.kind for tool in adapter.tools] == [
        "control",
        "termination",
        "termination",
        "control",
    ]
    assert adapter.tools is adapter.tools  # cached until the tools change

    adapter.add_tools(ToolLibrary([_WAVE]))
    assert "wave" in adapter.tools
    assert len(adapter.tools) == 5
    result = adapter.tools.execute("wave", "{}", Observation())
    assert result.note == "waved"


def test_contributed_tools_cannot_shadow_built_in_ones() -> None:
    adapter = ToolAdapter()
    adapter.bind(_info())
    clash = replace(_WAVE, name="done")
    with pytest.raises(ToolError, match="duplicate tool name"):
        adapter.add_tools(ToolLibrary([clash]))
    assert "wave" not in adapter.tools


def test_rebinding_drops_contributed_tools() -> None:
    adapter = ToolAdapter()
    adapter.bind(_info())
    adapter.add_tools(ToolLibrary([_WAVE]))
    adapter.bind(_info())
    assert "wave" not in adapter.tools


def test_instructions_carry_notes_bounds_and_pinned_dimensions() -> None:
    adapter = ToolAdapter()
    adapter.bind(_info(pinned=True))
    assert adapter.instructions() == (
        "Embodiment notes:\nJoint 0 is the shoulder.\n\n"
        "Embodiment bounds:\nPer-dimension bounds: shoulder: [-1, 1], elbow: [0, 0].\n"
        "Pinned dimensions: elbow (fixed; do not attempt to move them)."
    )
    adapter.bind(_info(docs="  "))
    assert adapter.instructions().startswith("Embodiment bounds:")
    assert embodiment_notes(_info(docs=None)) is None


def test_an_unbound_adapter_says_so() -> None:
    with pytest.raises(RuntimeError, match="before bind"):
        ToolAdapter().tools  # noqa: B018


def test_observation_lines_put_the_setpoint_after_the_readback() -> None:
    adapter = ToolAdapter()
    adapter.bind(_info())
    adapter.start_trial()
    observation = Observation(
        state={"joint_pos": np.array([0.1, 0.2]), "gripper": np.array([0.5])},
        extra={
            "approvals": [{"detail": "clamped"}, {"detail": "clamped"}, {"detail": None}],
            "motion_report": {"lines": ["shoulder: on target.", " ", 3]},
            "motor_temps": {"lines": ["motors: 31 C of 60 C."]},
            "operator_messages": [{"t": 4, "text": "slower please"}],
        },
    )
    assert adapter.observation_lines(observation) == [
        "state[joint_pos]: shoulder=0.1 elbow=0.2",
        "state[gripper]: [0.5]",
        "setpoint: none yet; the first move starts from the readback.",
        "approver: 3 step(s) modified (clamped \u00d72).",
        "shoulder: on target.",
        "motors: 31 C of 60 C.",
        "operator feedback (step 4): slower please",
    ]


def test_observation_helpers_tolerate_missing_and_malformed_data() -> None:
    assert approvals_line(Observation()) is None
    assert approvals_line(Observation(extra={"approvals": [{}, "x"]})) == (
        "approver: 2 step(s) modified."
    )
    assert approvals_line(Observation(extra={"approvals": [{"detail": "delta_clamped"}]})) == (
        "approver: 1 step(s) modified (delta_clamped)."
    )
    assert extra_lines(Observation(extra={"motion_report": "x"}), "motion_report") == []
    assert extra_lines(Observation(extra={"motion_report": {"lines": "x"}}), "motion_report") == []
    labels = ("joint_pos", ("a", "b", "c"))
    assert state_lines(Observation(state={"joint_pos": np.array([1.0, 2.0])}), labels) == [
        "state[joint_pos]: [1.0, 2.0]"
    ]
    assert observation_lines(Observation()) == []


def test_operator_lines_skip_malformed_values_and_entries() -> None:
    malformed: list[object] = [
        None,
        "feedback",
        {"t": "2", "text": "wrong step type"},
        {"t": 2, "text": 3},
        {"t": 2},
        {"text": "missing step"},
        {"t": 5, "text": "valid feedback"},
    ]

    assert operator_lines(Observation(extra={"operator_messages": malformed})) == [
        "operator feedback (step 5): valid feedback"
    ]
    assert operator_lines(Observation(extra={"operator_messages": {"t": 1}})) == []
    assert operator_lines(Observation()) == []


# -- embodiment-contributed tools through eval() -------------------------------


class _ToolPolicy:
    """Records the tools eval() hands it."""

    def __init__(self) -> None:
        from inspect_robots.mock import ScriptedPolicy

        self._inner = ScriptedPolicy()
        self.info = self._inner.info
        self.config = self._inner.config
        self.tools: ToolLibrary | None = None

    def bind_tools(self, tools: ToolLibrary) -> None:
        self.tools = tools

    def reset(self, scene: Scene) -> None:
        self._inner.reset(scene)

    def act(self, observation: Observation) -> ActionChunk:
        return self._inner.act(observation)


def _task() -> Task:
    return Task(
        name="t",
        scenes=[Scene(id="s", instruction="reach", init_seed=0)],
        scorer=success_at_end(),
        max_steps=5,
    )


def test_eval_hands_embodiment_tools_to_the_policy(tmp_path: Any) -> None:
    class _Contributing(CubePickEmbodiment):
        def contribute_tools(self, action_space: Box) -> ToolLibrary:
            return ToolLibrary([_WAVE])

    policy = _ToolPolicy()
    ir_eval(_task(), policy, _Contributing(), log_dir=str(tmp_path))
    assert policy.tools is not None and policy.tools.names == ("wave",)

    plain = _ToolPolicy()
    ir_eval(_task(), plain, CubePickEmbodiment(), log_dir=str(tmp_path))
    assert plain.tools is None


def test_a_contribution_that_is_not_a_library_is_a_config_error(tmp_path: Any) -> None:
    class _Bad(CubePickEmbodiment):
        def contribute_tools(self, action_space: Box) -> list[Tool]:
            return [_WAVE]

    with pytest.raises(ConfigError, match="contribute_tools returned list"):
        ir_eval(_task(), _ToolPolicy(), _Bad(), log_dir=str(tmp_path))


def test_a_move_without_a_declared_rate_reports_steps_only() -> None:
    adapter = ToolAdapter()
    adapter.bind(replace(_info(), control_hz=None))
    result = adapter.tools.execute(
        "move_joints",
        '{"targets": {"shoulder": 0.01}, "note": "nudge"}',
        Observation(state={"joint_pos": np.zeros(2)}),
    )
    assert result.note.startswith("executing move_joints over ")
    assert result.note.endswith(" steps")
