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


def _open_elbow(arguments: dict[str, Any], observation: Observation) -> ToolResult:
    """The docs' open_gripper example: set the last dimension from the readback."""
    target = np.asarray(observation.state["joint_pos"], dtype=np.float64).copy()
    target[-1] = 1.0
    return ToolResult(chunk=ActionChunk(actions=[Action(data=target)]), note="opening")


_OPEN = Tool(
    "open_elbow",
    {
        "type": "function",
        "function": {"name": "open_elbow", "description": "Open.", "parameters": {}},
    },
    _open_elbow,
)


def _arm_obs(shoulder: float, elbow: float, step: int | None = None) -> Observation:
    extra = {} if step is None else {"env_step": step}
    return Observation(state={"joint_pos": np.array([shoulder, elbow])}, extra=extra)


@pytest.mark.parametrize("follow", ["move_joints", "sleep", "done", "give_up"])
def test_a_contributed_motion_becomes_the_held_setpoint(follow: str) -> None:
    adapter = ToolAdapter()
    adapter.bind(_info())
    adapter.add_tools(ToolLibrary([_OPEN]))
    tools = adapter.tools
    close = '{"targets": {"elbow": -1.0}, "note": "close"}'
    assert tools.execute("move_joints", close, _arm_obs(0.0, 0.0)).error is None
    assert tools.execute("open_elbow", "{}", _arm_obs(0.0, -1.0)).chunk is not None

    arguments = {
        "move_joints": '{"targets": {"shoulder": 0.5}, "note": "reach"}',
        "sleep": '{"seconds": 1, "note": "wait"}',
        "done": '{"summary": "ok"}',
        "give_up": '{"reason": "stuck"}',
    }[follow]
    result = tools.execute(follow, arguments, _arm_obs(0.0, 1.0))
    assert result.error is None and result.chunk is not None
    # The elbow stays where the contributed tool opened it; it never snaps shut.
    assert all(action.data[1] == 1.0 for action in result.chunk.actions)
    assert adapter.control.setpoint_lines(_arm_obs(0.0, 1.0))[0].endswith("elbow=1.0")


def test_a_contributed_chunk_settles_like_a_built_in_move() -> None:
    adapter = ToolAdapter()
    adapter.bind(_info())

    def slide(arguments: dict[str, Any], observation: Observation) -> ToolResult:
        waypoints = [Action(data=np.array([0.1 * i, 0.0])) for i in range(1, 6)]
        return ToolResult(chunk=ActionChunk(actions=waypoints))

    adapter.add_tools(ToolLibrary([replace(_WAVE, name="slide", run=slide)]))
    adapter.tools.execute("slide", "{}", _arm_obs(0.0, 0.0, step=10))
    # Cut short after two steps: the setpoint is the second waypoint.
    lines = adapter.control.setpoint_lines(_arm_obs(0.2, 0.0, step=12))
    assert lines[0] == "setpoint: shoulder=0.2 elbow=0.0"


@pytest.mark.parametrize(
    "data", [np.array([0.1, np.nan]), np.zeros(3), np.array(["a", "b"], dtype=object)]
)
def test_an_unreadable_contributed_chunk_clears_the_setpoint(data: Any) -> None:
    adapter = ToolAdapter()
    adapter.bind(_info())

    def odd(arguments: dict[str, Any], observation: Observation) -> ToolResult:
        return ToolResult(chunk=ActionChunk(actions=[Action(data=data)]))

    adapter.add_tools(ToolLibrary([replace(_WAVE, name="odd", run=odd)]))
    adapter.tools.execute(
        "move_joints", '{"targets": {"elbow": -1.0}, "note": "x"}', _arm_obs(0, 0)
    )
    adapter.tools.execute("odd", "{}", _arm_obs(0.0, -1.0))
    assert adapter.control.setpoint_lines(_arm_obs(0.0, -1.0)) == [
        "setpoint: none yet; the first move starts from the readback."
    ]


def test_contributed_results_without_motion_leave_the_setpoint() -> None:
    adapter = ToolAdapter()
    adapter.bind(_info())

    def look(arguments: dict[str, Any], observation: Observation) -> ToolResult:
        return ToolResult(note="looked")

    def halt(arguments: dict[str, Any], observation: Observation) -> ToolResult:
        from inspect_robots.types import GIVE_UP, Termination

        return ToolResult(chunk=ActionChunk(actions=[], terminal=Termination(GIVE_UP, "e-stop")))

    adapter.add_tools(
        ToolLibrary([replace(_WAVE, name="look", run=look), replace(_WAVE, name="halt", run=halt)])
    )
    adapter.tools.execute(
        "move_joints", '{"targets": {"elbow": -1.0}, "note": "x"}', _arm_obs(0, 0)
    )
    adapter.tools.execute("look", "{}", _arm_obs(0.0, -1.0))
    adapter.tools.execute("halt", "{}", _arm_obs(0.0, -1.0))
    assert adapter.control.setpoint_lines(_arm_obs(0.0, -1.0))[0].endswith("elbow=-1.0")


def test_displacement_surfaces_ignore_contributed_chunks() -> None:
    from inspect_robots.tools import control_tools

    space = Box(
        shape=(2,),
        low=np.array([-0.1, -0.1]),
        high=np.array([0.1, 0.1]),
        semantics=ActionSemantics("eef_delta_pos", frame="world"),
    )
    control = control_tools(space, ObservationSpace(), control_hz=10.0)
    control.track(ActionChunk(actions=[Action(data=np.array([0.1, 0.1]))]), Observation())
    assert control.setpoint_lines(Observation()) == []
