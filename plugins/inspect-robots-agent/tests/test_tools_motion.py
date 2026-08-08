"""Tool surface + speed-limited motion synthesis (plan 0014)."""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation
from inspect_robots_agent._llm import ToolCall
from inspect_robots_agent._tools import ToolResult, ToolsetError, build_toolset

_ARM_LABELS = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*[f"j{i}" for i in range(6)], "gripper")
)
_DEFAULT_MOVE_NOTE = "The robot state is visible, so I chose this motion to make progress."
_NOTE_ERROR = "note is required: describe what you observe and why you chose this motion"


def _bimanual_space() -> Box:
    return Box(
        shape=(14,),
        low=np.array([-np.pi] * 6 + [0.0] + [-np.pi] * 6 + [0.0]),
        high=np.array([np.pi] * 6 + [1.0] + [np.pi] * 6 + [1.0]),
        semantics=ActionSemantics("joint_pos", dim_labels=_ARM_LABELS),
    )


def _bimanual_obs_space() -> ObservationSpace:
    return ObservationSpace(state=StateSpec(fields=(StateField(key="joint_pos", shape=(14,)),)))


def _absolute_space(
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    labels: tuple[str, ...] | None = ("joint",),
) -> Box:
    return Box(
        shape=(1,),
        low=np.array([-1.0]) if low is None else low,
        high=np.array([1.0]) if high is None else high,
        semantics=ActionSemantics("joint_pos", dim_labels=labels),
    )


def _absolute_obs_space(dim: int = 1, key: str = "q") -> ObservationSpace:
    return ObservationSpace(state=StateSpec(fields=(StateField(key=key, shape=(dim,)),)))


def _declaring_space(
    max_step: tuple[float | None, float | None] = (None, 0.1),
) -> Box:
    return Box(
        shape=(2,),
        low=np.array([-1.0, 0.0]),
        high=np.array([1.0, 1.0]),
        semantics=ActionSemantics(
            "joint_pos",
            dim_labels=("joint", "gripper"),
            max_step=max_step,
        ),
    )


def _camera_obs_space() -> ObservationSpace:
    return ObservationSpace(
        cameras=(
            CameraSpec(name="top", height=2, width=2),
            CameraSpec(name="wrist", height=2, width=2),
        ),
        state=StateSpec(fields=(StateField(key="q", shape=(1,)),)),
    )


def _delta_space(
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> Box:
    return Box(
        shape=(2,),
        low=np.array([-0.1, -0.1]) if low is None else low,
        high=np.array([0.1, 0.1]) if high is None else high,
        semantics=ActionSemantics("eef_delta_pos", frame="world"),
    )


def _call(name: str, **arguments: object) -> ToolCall:
    if name in ("move_joints", "move_to", "move_by"):
        arguments.setdefault("note", _DEFAULT_MOVE_NOTE)
    return ToolCall(id="call_1", name=name, arguments=json.dumps(arguments))


def _obs(state: dict[str, np.ndarray] | None = None) -> Observation:
    return Observation(state={"q": np.array([0.0])} if state is None else state)


def _execute_absolute(
    target: float,
    current: float = 0.0,
    *,
    control_hz: float | None = 10.0,
    max_speed_frac: float = 0.1,
    note: object = _DEFAULT_MOVE_NOTE,
) -> ToolResult:
    toolset = build_toolset(
        _absolute_space(),
        _absolute_obs_space(),
        control_hz=control_hz,
        max_speed_frac=max_speed_frac,
    )
    return toolset.execute(
        _call("move_joints", note=note, targets={"joint": target}),
        _obs({"q": np.array([current])}),
    )


# --- bind-time validation ------------------------------------------------------


def test_build_refuses_unsupported_configurations() -> None:
    no_sem = Box(shape=(2,), low=np.zeros(2), high=np.ones(2))
    with pytest.raises(ToolsetError, match="semantics"):
        build_toolset(no_sem, _bimanual_obs_space(), control_hz=10.0)

    vel = Box(
        shape=(2,),
        low=-np.ones(2),
        high=np.ones(2),
        semantics=ActionSemantics("joint_vel"),
    )
    with pytest.raises(ToolsetError, match="joint_vel"):
        build_toolset(vel, _bimanual_obs_space(), control_hz=10.0)

    quat = Box(
        shape=(7,),
        low=-np.ones(7),
        high=np.ones(7),
        semantics=ActionSemantics("eef_abs_pose", rotation_repr="quat_wxyz"),
    )
    with pytest.raises(ToolsetError, match="rotation_repr"):
        build_toolset(quat, _bimanual_obs_space(), control_hz=10.0)


def test_pre_check_refuses_displacement_modes_only_when_configured() -> None:
    def allow(waypoints: npt.NDArray[np.float64]) -> str | None:
        return None

    with pytest.raises(ToolsetError, match=r"displacement.*plan 0031"):
        build_toolset(
            _delta_space(),
            ObservationSpace(),
            control_hz=10.0,
            pre_check=allow,
        )

    build_toolset(
        _absolute_space(),
        _absolute_obs_space(),
        control_hz=10.0,
        pre_check=allow,
    )
    build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)


def test_absolute_mode_requires_exactly_one_aligned_state_field() -> None:
    space = _bimanual_space()
    with pytest.raises(ToolsetError, match="StateSpec"):
        build_toolset(space, ObservationSpace(), control_hz=10.0)

    none_match = ObservationSpace(state=StateSpec(fields=(StateField(key="eef", shape=(3,)),)))
    with pytest.raises(ToolsetError, match="exactly one"):
        build_toolset(space, none_match, control_hz=10.0)

    two_match = ObservationSpace(
        state=StateSpec(fields=(StateField(key="a", shape=(14,)), StateField(key="b", shape=(14,))))
    )
    with pytest.raises(ToolsetError, match="exactly one"):
        build_toolset(space, two_match, control_hz=10.0)


def test_absolute_mode_labels_its_reference_even_when_the_key_is_noncanonical() -> None:
    canonical = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    assert canonical.state_labels() == ("joint_pos", _ARM_LABELS)

    labels = ("x", "y", "z", "yaw")
    space = Box(
        shape=(4,),
        low=-np.ones(4),
        high=np.ones(4),
        semantics=ActionSemantics("eef_abs_pose", dim_labels=labels),
    )
    observation_space = ObservationSpace(
        state=StateSpec(fields=(StateField(key="eef_state", shape=(4,)),))
    )
    noncanonical = build_toolset(space, observation_space, control_hz=10.0)
    assert noncanonical.state_labels() == ("eef_state", labels)


def test_displacement_mode_labels_one_canonical_shape_match() -> None:
    labels = ("dx", "dy")
    space = Box(
        shape=(2,),
        low=-np.ones(2),
        high=np.ones(2),
        semantics=ActionSemantics("eef_delta_pos", dim_labels=labels),
    )
    observation_space = ObservationSpace(
        state=StateSpec(fields=(StateField(key="eef_pos", shape=(2,)),))
    )
    assert build_toolset(space, observation_space, 10.0).state_labels() == ("eef_pos", labels)


def test_displacement_mode_without_state_spec_has_no_state_labels() -> None:
    info = CubePickEmbodiment().info
    assert (
        build_toolset(info.action_space, info.observation_space, info.control_hz).state_labels()
        is None
    )


def test_displacement_mode_rejects_ambiguous_canonical_shape_matches() -> None:
    labels = ("a", "b")
    space = Box(
        shape=(2,),
        low=-np.ones(2),
        high=np.ones(2),
        semantics=ActionSemantics("joint_delta", dim_labels=labels),
    )
    observation_space = ObservationSpace(
        state=StateSpec(
            fields=(
                StateField(key="joint_pos", shape=(2,)),
                StateField(key="eef_pos", shape=(2,)),
            )
        )
    )
    assert build_toolset(space, observation_space, 10.0).state_labels() is None


def test_displacement_mode_excludes_noncanonical_shape_matches() -> None:
    space = Box(
        shape=(2,),
        low=-np.ones(2),
        high=np.ones(2),
        semantics=ActionSemantics("joint_delta", dim_labels=("a", "b")),
    )
    observation_space = ObservationSpace(
        state=StateSpec(fields=(StateField(key="object_pose", shape=(2,)),))
    )
    assert build_toolset(space, observation_space, 10.0).state_labels() is None


def test_synthesized_action_labels_do_not_label_state() -> None:
    observation_space = ObservationSpace(
        state=StateSpec(fields=(StateField(key="eef_pos", shape=(2,)),))
    )
    assert build_toolset(_delta_space(), observation_space, 10.0).state_labels() is None


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (None, np.array([1.0])),
        (np.array([-1.0]), None),
        (np.array([float("-inf")]), np.array([1.0])),
        (np.array([-1.0]), np.array([float("inf")])),
        (np.array([float("nan")]), np.array([1.0])),
    ],
)
def test_absolute_mode_requires_finite_bounds(
    low: np.ndarray | None, high: np.ndarray | None
) -> None:
    space = Box(
        shape=(1,),
        low=low,
        high=high,
        semantics=ActionSemantics("joint_pos", dim_labels=("joint",)),
    )
    with pytest.raises(ToolsetError, match="finite low and high bounds"):
        build_toolset(space, _absolute_obs_space(), control_hz=10.0)


@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        (None, np.array([0.1, 0.1]), "finite low and high bounds"),
        (np.array([-0.1, -0.1]), None, "finite low and high bounds"),
        (np.array([-np.inf, -0.1]), np.array([0.1, 0.1]), "finite low and high bounds"),
        (np.array([-0.1, -0.1]), np.array([0.1, np.inf]), "finite low and high bounds"),
        (np.array([0.01, -0.1]), np.array([0.1, 0.1]), "contain zero"),
        (np.array([-0.1, -0.1]), np.array([-0.01, 0.1]), "contain zero"),
    ],
)
def test_displacement_mode_requires_finite_zero_containing_bounds(
    low: np.ndarray | None, high: np.ndarray | None, message: str
) -> None:
    space = Box(
        shape=(2,),
        low=low,
        high=high,
        semantics=ActionSemantics("eef_delta_pos"),
    )
    with pytest.raises(ToolsetError, match=message):
        build_toolset(space, ObservationSpace(), control_hz=10.0)


def test_zero_width_and_zero_sided_bounds_bind() -> None:
    fixed = _absolute_space(low=np.array([0.3]), high=np.array([0.3]))
    build_toolset(fixed, _absolute_obs_space(), control_hz=10.0)

    build_toolset(
        _delta_space(low=np.array([0.0, -0.1]), high=np.array([0.1, 0.0])),
        ObservationSpace(),
        control_hz=10.0,
    )


@pytest.mark.parametrize("control_hz", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_declared_control_hz_must_be_finite_and_positive(control_hz: float) -> None:
    with pytest.raises(ToolsetError, match="control_hz must be finite and > 0"):
        build_toolset(_delta_space(), ObservationSpace(), control_hz=control_hz)


def test_control_hz_none_binds() -> None:
    build_toolset(_delta_space(), ObservationSpace(), control_hz=None)


def test_build_rejects_degenerate_derivations() -> None:
    with pytest.raises(ToolsetError, match="too large to derive a playout cap"):
        build_toolset(_delta_space(), ObservationSpace(), control_hz=1e308)

    with pytest.raises(ToolsetError, match="underflows the derived per-step limit"):
        build_toolset(
            _absolute_space(),
            _absolute_obs_space(),
            control_hz=10.0,
            max_speed_frac=5e-324,
        )

    with pytest.raises(ToolsetError, match="underflows to a zero per-step limit"):
        build_toolset(
            _delta_space(),
            ObservationSpace(),
            control_hz=10.0,
            max_speed_frac=5e-324,
        )

    huge = _absolute_space(low=np.array([-1e308]), high=np.array([1e308]))
    with pytest.raises(ToolsetError, match=r"range .* overflows"):
        build_toolset(huge, _absolute_obs_space(), control_hz=10.0)

    # float32 bounds whose difference overflows only in the native dtype:
    # DeltaLimitApprover subtracts without promoting, so this must reject too.
    huge32 = _absolute_space(
        low=np.array([-3e38], dtype=np.float32), high=np.array([3e38], dtype=np.float32)
    )
    with pytest.raises(ToolsetError, match=r"range .* overflows"):
        build_toolset(huge32, _absolute_obs_space(), control_hz=10.0)

    offset = _absolute_space(low=np.array([1e16]), high=np.array([1e16 + 2.0]))
    with pytest.raises(ToolsetError, match="too coarse at this magnitude"):
        build_toolset(offset, _absolute_obs_space(), control_hz=10.0)

    subnormal_range = _absolute_space(low=np.array([0.0]), high=np.array([5e-324]))
    with pytest.raises(ToolsetError, match="too coarse at this magnitude"):
        build_toolset(subnormal_range, _absolute_obs_space(), control_hz=10.0)

    # frac/hz is nonzero but multiplying by a small range underflows the
    # derived limit to zero; the dimension must not be misreported as fixed.
    with pytest.raises(ToolsetError, match="underflows the derived per-step limit"):
        build_toolset(
            _absolute_space(low=np.array([0.0]), high=np.array([0.1])),
            _absolute_obs_space(),
            control_hz=1.0,
            max_speed_frac=5e-324,
        )

    matrix = Box(
        shape=(2, 2),
        low=np.zeros((2, 2)),
        high=np.ones((2, 2)),
        semantics=ActionSemantics("joint_pos"),
    )
    with pytest.raises(ToolsetError, match="only 1-D"):
        build_toolset(matrix, _absolute_obs_space(dim=4), control_hz=10.0)


def test_low_precision_bounds_never_outrun_native_backstop() -> None:
    # DeltaLimitApprover derives 0.05 * (high - low) in the box's dtype;
    # float16 rounds that to 0.0999755859375, below the float64 0.1. Every
    # emitted step must respect the *native* value or the backstop clamps.
    space = Box(
        shape=(1,),
        low=np.array([-1.0], dtype=np.float16),
        high=np.array([1.0], dtype=np.float16),
        semantics=ActionSemantics("joint_pos", dim_labels=("joint",)),
    )
    # frac 0.5 puts the frac-derived limit (0.1) above the float16-native
    # backstop, so the elementwise min actually binds; the 0.1 default would
    # pass this test without the native-dtype fix at all.
    toolset = build_toolset(space, _absolute_obs_space(), control_hz=10.0, max_speed_frac=0.5)
    result = toolset.execute(
        _call("move_joints", targets={"joint": 0.9999}),
        _obs({"q": np.array([0.0])}),
    )
    assert result.chunk is not None
    chain = ChainApprover(ClampApprover(space), DeltaLimitApprover(space))
    store: dict[str, Any] = {}
    for action in result.chunk.actions:
        assert chain.review(action, store) is action


def test_declared_step_paces_gripper_while_undeclared_joint_is_unchanged() -> None:
    space = _declaring_space()
    toolset = build_toolset(space, _absolute_obs_space(dim=2), control_hz=10.0)
    observation = _obs({"q": np.array([0.0, 0.0])})

    gripper = toolset.execute(
        _call("move_joints", targets={"gripper": 1.0}),
        observation,
    )
    joint = toolset.execute(
        _call("move_joints", targets={"joint": 1.0}),
        observation,
    )

    assert gripper.error is None and gripper.chunk is not None
    assert len(gripper.chunk.actions) == 11
    assert joint.error is None and joint.chunk is not None
    assert len(joint.chunk.actions) == 51

    undeclared = Box(
        shape=(2,),
        low=space.low,
        high=space.high,
        semantics=ActionSemantics("joint_pos", dim_labels=("joint", "gripper")),
    )
    old_toolset = build_toolset(undeclared, _absolute_obs_space(dim=2), control_hz=10.0)
    old_full_stroke = old_toolset.execute(
        _call("move_joints", targets={"gripper": 1.0}),
        observation,
    )
    assert old_full_stroke.chunk is None
    assert old_full_stroke.error is not None
    assert "playout cap" in old_full_stroke.error


@pytest.mark.parametrize(
    ("max_speed_frac", "expected_steps"),
    [(0.05, 21), (0.2, 11)],
)
def test_declared_step_scales_down_but_not_above_its_ceiling(
    max_speed_frac: float,
    expected_steps: int,
) -> None:
    toolset = build_toolset(
        _declaring_space(),
        _absolute_obs_space(dim=2),
        control_hz=10.0,
        max_speed_frac=max_speed_frac,
    )
    result = toolset.execute(
        _call("move_joints", targets={"gripper": 1.0}),
        _obs({"q": np.array([0.0, 0.0])}),
    )

    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == expected_steps


def test_combined_declared_move_uses_largest_per_dimension_step_count() -> None:
    toolset = build_toolset(
        _declaring_space(),
        _absolute_obs_space(dim=2),
        control_hz=10.0,
    )
    result = toolset.execute(
        _call("move_joints", targets={"joint": 0.4, "gripper": 1.0}),
        _obs({"q": np.array([0.0, 0.0])}),
    )

    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == 21


def test_declared_step_is_the_float_spacing_guard_budget() -> None:
    space = Box(
        shape=(1,),
        low=np.array([1e9]),
        high=np.array([1e9 + 100.0]),
        semantics=ActionSemantics(
            "joint_pos",
            dim_labels=("joint",),
            max_step=(0.1,),
        ),
    )

    with pytest.raises(ToolsetError, match="too coarse at this magnitude"):
        build_toolset(space, _absolute_obs_space(), control_hz=10.0)


def test_declared_step_pacing_underflow_hits_movable_zero_limit_guard() -> None:
    space = Box(
        shape=(1,),
        low=np.array([0.0]),
        high=np.array([1e-300]),
        semantics=ActionSemantics(
            "joint_pos",
            dim_labels=("joint",),
            max_step=(1e-300,),
        ),
    )

    with pytest.raises(ToolsetError, match="underflows the derived per-step limit"):
        build_toolset(
            space,
            _absolute_obs_space(),
            control_hz=10.0,
            max_speed_frac=1e-300,
        )


def test_declared_step_chunks_pass_the_default_delta_approver_unmodified() -> None:
    space = _declaring_space()
    toolset = build_toolset(space, _absolute_obs_space(dim=2), control_hz=10.0)
    start = np.array([0.0, 0.0])
    result = toolset.execute(
        _call("move_joints", targets={"gripper": 1.0}),
        _obs({"q": start}),
    )
    assert result.chunk is not None

    approver = DeltaLimitApprover(space)
    store: dict[str, Any] = {}
    approver.review(Action(data=start), store)
    for action in result.chunk.actions:
        assert approver.review(action, store) is action


@pytest.mark.parametrize("max_speed_frac", [0.0, -0.1, float("inf"), float("nan")])
def test_build_rejects_invalid_max_speed_frac(max_speed_frac: float) -> None:
    with pytest.raises(ToolsetError, match="max_speed_frac must be finite and > 0"):
        build_toolset(
            _absolute_space(),
            _absolute_obs_space(),
            control_hz=10.0,
            max_speed_frac=max_speed_frac,
        )


# --- schemas -------------------------------------------------------------------


def test_schemas_match_control_mode_and_remove_duration() -> None:
    absolute = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    names = [t["function"]["name"] for t in absolute.schemas()]
    assert names == ["move_joints", "done", "give_up"]
    move_schema = json.dumps(absolute.schemas()[0])
    assert "left_j0" in move_schema and "right_gripper" in move_schema
    assert "duration_s" not in move_schema
    absolute_parameters = [schema["function"]["parameters"] for schema in absolute.schemas()]
    assert [parameters["required"] for parameters in absolute_parameters] == [
        ["targets", "note"],
        ["summary", "hindsight"],
        ["reason", "hindsight"],
    ]
    assert absolute_parameters[0]["properties"]["note"] == {
        "type": "string",
        "description": (
            "What you observe right now in the observation (images, if any, and state), and why "
            "you chose this motion. The user reads these notes live and in the saved transcript "
            "to follow what you see and what you decide. Write for them, in one or two plain "
            "sentences."
        ),
    }
    assert "note" not in absolute_parameters[1]["properties"]
    assert "note" not in absolute_parameters[2]["properties"]

    displacement = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    names = [t["function"]["name"] for t in displacement.schemas()]
    assert names == ["move_by", "done", "give_up"]
    displacement_parameters = [
        schema["function"]["parameters"] for schema in displacement.schemas()
    ]
    assert [parameters["required"] for parameters in displacement_parameters] == [
        ["deltas", "note"],
        ["summary", "hindsight"],
        ["reason", "hindsight"],
    ]
    assert displacement_parameters[0]["properties"]["note"]["type"] == "string"


def test_stop_schemas_describe_required_hindsight() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    expected_description = (
        "What do you know now that you wish you had known at the start of this episode? "
        "Concrete, transferable facts about this rig, task, or embodiment (camera mounting "
        "and extrinsics, table and base geometry, gripper axis and offsets, controller "
        "behavior, metric scale), written as advice to a future agent attempting the same "
        "task. Say 'none' if nothing qualifies."
    )

    for schema, detail_key in zip(toolset.schemas()[1:], ("summary", "reason"), strict=True):
        parameters = schema["function"]["parameters"]
        assert parameters["required"] == [detail_key, "hindsight"]
        assert parameters["properties"]["hindsight"] == {
            "type": "string",
            "description": expected_description,
        }


def test_take_pic_schema_is_exposed_only_on_demand() -> None:
    always = build_toolset(_absolute_space(), _camera_obs_space(), control_hz=10.0)
    on_demand = build_toolset(
        _absolute_space(),
        _camera_obs_space(),
        control_hz=10.0,
        images="on_demand",
    )

    assert [schema["function"]["name"] for schema in always.schemas()] == [
        "move_joints",
        "done",
        "give_up",
    ]
    assert [schema["function"]["name"] for schema in on_demand.schemas()] == [
        "move_joints",
        "done",
        "give_up",
        "take_pic",
    ]
    capture = on_demand.schemas()[-1]["function"]
    assert capture["parameters"]["required"] == ["note"]
    assert capture["parameters"]["properties"]["cameras"]["items"] == {"type": "string"}
    assert "top" in capture["description"] and "wrist" in capture["description"]

    stray = always.execute(
        _call("take_pic", note="I need to inspect the wrist camera."),
        Observation(
            state={"q": np.array([0.0])},
            images={"top": np.zeros((2, 2, 3), dtype=np.uint8)},
        ),
    )
    assert stray.error == "unknown tool 'take_pic'; available: move_joints, done, give_up"


def test_on_demand_requires_a_declared_camera_at_bind_time() -> None:
    with pytest.raises(ToolsetError, match=r"fix: drop -P images=on_demand"):
        build_toolset(
            _absolute_space(),
            _absolute_obs_space(),
            control_hz=10.0,
            images="on_demand",
        )


def test_take_pic_validation_uses_images_present_now() -> None:
    toolset = build_toolset(
        _absolute_space(),
        _camera_obs_space(),
        control_hz=10.0,
        images="on_demand",
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    observation = Observation(state={"q": np.array([0.0])}, images={"top": image})
    note = "I need a current image to verify the arm position."

    unknown = toolset.execute(_call("take_pic", cameras=["side"], note=note), observation)
    absent = toolset.execute(_call("take_pic", cameras=["wrist"], note=note), observation)
    wrong_type = toolset.execute(_call("take_pic", cameras="top", note=note), observation)
    empty = toolset.execute(_call("take_pic", cameras=[], note=note), observation)
    blank_note = toolset.execute(_call("take_pic", cameras=["top"], note=" "), observation)
    imageless = toolset.execute(
        _call("take_pic", note=note),
        Observation(state={"q": np.array([0.0])}),
    )

    assert unknown.error == "unknown or unavailable camera 'side'; available now: 'top'"
    assert absent.error == "unknown or unavailable camera 'wrist'; available now: 'top'"
    assert wrong_type.error == "cameras must be a non-empty list of strings when provided"
    assert empty.error == "cameras must be a non-empty list of strings when provided"
    assert blank_note.error is not None and blank_note.error.startswith("note is required:")
    assert imageless.error is None and imageless.capture == ()

    all_cameras = toolset.execute(_call("take_pic", note=note), observation)
    named = toolset.execute(_call("take_pic", cameras=["top"], note=note), observation)
    assert all_cameras.error is None and all_cameras.capture is None
    assert named.error is None and named.capture == ("top",)
    assert named.chunk is None


def test_explicit_take_pic_rejects_an_imageless_observation() -> None:
    toolset = build_toolset(
        _absolute_space(),
        _camera_obs_space(),
        control_hz=10.0,
        images="on_demand",
    )
    result = toolset.execute(
        _call(
            "take_pic",
            cameras=["top"],
            note="I need the top view to verify the arm position.",
        ),
        Observation(state={"q": np.array([0.0])}),
    )

    assert result.error is None
    assert result.capture == ()


# --- absolute-mode synthesis ---------------------------------------------------


def test_move_joints_derives_steps_and_snaps_bit_exact_target() -> None:
    # Factory default frac 0.1 at 10 Hz: 1% of range per step, distance 1.5
    # over range 2.0 with headroom -> 76 steps.
    result = _execute_absolute(1.0, current=-0.5)
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == 76
    assert result.chunk.actions[-1].data[0] == 1.0
    assert result.note == "executing move_joints over 76 steps (7.6s)"

    # Pins the snap itself: 0.3 is interior (clip cannot repair it) and plain
    # linspace arithmetic lands on 0.30000000000000004 instead.
    snapped = _execute_absolute(0.3, current=-0.1)
    assert snapped.chunk is not None
    assert snapped.chunk.actions[-1].data[0] == 0.3


def test_absolute_result_carries_target_and_residual_is_best_effort() -> None:
    toolset = build_toolset(
        _absolute_space(labels=("joint",)),
        _absolute_obs_space(),
        control_hz=10.0,
    )
    result = toolset.execute(
        _call("move_joints", targets={"joint": 0.5}),
        _obs({"q": np.array([0.0])}),
    )

    assert result.target is not None
    assert np.array_equal(result.target, np.array([0.5]))
    assert toolset.residual(result.target, _obs({"q": np.array([0.496])})) == (
        "joint",
        pytest.approx(0.004),
    )
    assert toolset.residual(result.target, Observation()) is None
    assert toolset.residual(result.target, _obs({"q": np.array([0.1, 0.2])})) is None
    assert toolset.residual(result.target, _obs({"q": np.array([np.nan])})) is None

    displacement = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    moved = displacement.execute(_call("move_by", deltas={"0": 0.05}), Observation())
    assert moved.target is None
    assert displacement.residual(np.zeros(2), Observation()) is None


def test_move_joints_clips_every_interpolant_into_box() -> None:
    space = _absolute_space(low=np.array([-1.0]), high=np.array([0.3]))
    toolset = build_toolset(space, _absolute_obs_space(), control_hz=10.0)
    result = toolset.execute(
        _call("move_joints", targets={"joint": 0.3}),
        _obs({"q": np.array([-0.1])}),
    )
    assert result.error is None and result.chunk is not None
    assert all(-1.0 <= float(action.data[0]) <= 0.3 for action in result.chunk.actions)


def test_pre_check_receives_exact_read_only_clipped_waypoints_once() -> None:
    low = np.array([-1.0, -0.5])
    high = np.array([0.3, 0.5])
    space = Box(
        shape=(2,),
        low=low,
        high=high,
        semantics=ActionSemantics("joint_pos", dim_labels=("a", "b")),
    )
    received: list[npt.NDArray[np.float64]] = []

    def allow(waypoints: npt.NDArray[np.float64]) -> str | None:
        assert waypoints.dtype == np.float64
        assert waypoints.ndim == 2
        assert waypoints.shape[1] == 2
        assert waypoints.flags.writeable is False
        assert bool(np.all(waypoints >= low))
        assert bool(np.all(waypoints <= high))
        assert np.array_equal(waypoints[-1], np.array([0.3, 0.4]))
        with pytest.raises(ValueError):
            waypoints[0, 0] = 0.0
        received.append(waypoints.copy())
        return None

    toolset = build_toolset(
        space,
        _absolute_obs_space(dim=2),
        control_hz=10.0,
        pre_check=allow,
    )
    result = toolset.execute(
        _call("move_joints", targets={"a": 0.3, "b": 0.4}),
        _obs({"q": np.array([-0.1, 0.0])}),
    )

    assert result.error is None and result.chunk is not None
    emitted = np.stack([action.data for action in result.chunk.actions])
    assert len(received) == 1
    assert np.array_equal(received[0], emitted)


def test_pre_check_rejection_returns_a_tool_error_without_a_chunk() -> None:
    def reject(waypoints: npt.NDArray[np.float64]) -> str | None:
        return "left_wrist:table at waypoint 4"

    toolset = build_toolset(
        _absolute_space(),
        _absolute_obs_space(),
        control_hz=10.0,
        pre_check=reject,
    )
    result = toolset.execute(
        _call("move_joints", targets={"joint": 0.3}),
        _obs(),
    )

    assert result.error == ("pre-check rejected this motion: left_wrist:table at waypoint 4")
    assert result.chunk is None


def test_move_joints_headroom_stays_within_default_backstop() -> None:
    current = -0.5
    result = _execute_absolute(1.0, current=current)
    assert result.chunk is not None
    emitted = [current, *(float(action.data[0]) for action in result.chunk.actions)]
    assert all(right - left <= 0.1 for left, right in pairwise(emitted))


@pytest.mark.parametrize(("target", "current"), [(0.01, 0.0), (0.4, 0.4)])
def test_move_joints_has_one_step_floor(target: float, current: float) -> None:
    result = _execute_absolute(target, current=current)
    assert result.chunk is not None
    assert len(result.chunk.actions) == 1


def test_move_joints_preserves_unnamed_dimensions_and_index_labels() -> None:
    space = Box(
        shape=(2,),
        low=np.zeros(2),
        high=np.ones(2),
        semantics=ActionSemantics("joint_pos"),
    )
    obs_space = _absolute_obs_space(dim=2)
    toolset = build_toolset(space, obs_space, control_hz=10.0)
    result = toolset.execute(
        _call("move_joints", targets={"1": 0.6}),
        _obs({"q": np.array([0.2, 0.2])}),
    )
    assert result.error is None and result.chunk is not None
    final = np.asarray(result.chunk.actions[-1].data)
    assert final[0] == 0.2 and final[1] == 0.6


def test_move_joints_rejects_out_of_bounds_but_accepts_bound() -> None:
    toolset = build_toolset(_absolute_space(), _absolute_obs_space(), control_hz=10.0)
    rejected = toolset.execute(
        _call("move_joints", targets={"joint": 1.01}),
        _obs(),
    )
    assert rejected.error == "target for joint is outside [-1.0, 1.0]"

    accepted = toolset.execute(
        _call("move_joints", targets={"joint": -1.0}),
        _obs(),
    )
    assert accepted.error is None and accepted.chunk is not None
    assert accepted.chunk.actions[-1].data[0] == -1.0
    # Also pins build_toolset's factory default (no frac passed): 0.1 at
    # 10 Hz over range 2.0, distance 1.0 -> 51 steps.
    assert len(accepted.chunk.actions) == 51


def test_zero_width_target_uses_bound_not_noisy_observation_and_no_steps() -> None:
    bound = 0.30000000000000004
    space = _absolute_space(low=np.array([bound]), high=np.array([bound]), labels=("fixed",))
    toolset = build_toolset(space, _absolute_obs_space(), control_hz=10.0)

    accepted = toolset.execute(
        _call("move_joints", targets={"fixed": bound}),
        _obs({"q": np.array([0.31])}),
    )
    assert accepted.error is None and accepted.chunk is not None
    assert len(accepted.chunk.actions) == 1
    assert accepted.chunk.actions[0].data[0] == bound

    rejected = toolset.execute(
        _call("move_joints", targets={"fixed": 0.3}),
        _obs({"q": np.array([bound])}),
    )
    assert rejected.error == "dimension fixed is fixed at 0.30000000000000004"


@pytest.mark.parametrize(("control_hz", "max_speed_frac"), [(5.0, 0.5), (10.0, 1.0)])
def test_move_joints_per_step_ceiling_matches_default_backstop(
    control_hz: float, max_speed_frac: float
) -> None:
    result = _execute_absolute(
        1.0,
        current=-1.0,
        control_hz=control_hz,
        max_speed_frac=max_speed_frac,
    )
    assert result.chunk is not None
    assert len(result.chunk.actions) == 21
    emitted = [-1.0, *(float(action.data[0]) for action in result.chunk.actions)]
    assert all(right - left <= 0.1 for left, right in pairwise(emitted))


def test_move_joints_honors_non_default_speed_fraction() -> None:
    result = _execute_absolute(0.5, max_speed_frac=0.25)
    assert result.chunk is not None
    assert len(result.chunk.actions) == 11


@pytest.mark.parametrize("bad_state", [float("nan"), float("inf"), float("-inf")])
def test_move_joints_rejects_non_finite_observed_state(bad_state: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _execute_absolute(0.5, current=bad_state)


def test_broken_sensor_raises_even_with_malformed_arguments() -> None:
    toolset = build_toolset(_absolute_space(), _absolute_obs_space(), control_hz=10.0)
    # A malformed tool call must not mask a broken sensor behind a
    # correctable structured error.
    with pytest.raises(ValueError, match="non-finite"):
        toolset.execute(
            ToolCall(
                id="call_1",
                name="move_joints",
                arguments=json.dumps({"targets": {"unknown_dim": 0.1}}),
            ),
            _obs({"q": np.array([float("nan")])}),
        )


def test_move_joints_absurd_finite_state_returns_cap_error() -> None:
    result = _execute_absolute(0.0, current=1e308)
    assert result.chunk is None
    assert result.error is not None and "split the move into smaller motions" in result.error


def test_move_joints_over_cap_returns_structured_error() -> None:
    result = _execute_absolute(1.0, current=-1.0, max_speed_frac=0.01)
    assert result.chunk is None
    assert result.error is not None and "split the move into smaller motions" in result.error

    boundary = _execute_absolute(0.98, current=-1.0, max_speed_frac=0.1)
    assert boundary.error is None and boundary.chunk is not None
    assert len(boundary.chunk.actions) == 100


def test_absolute_chunks_pass_default_approvers_across_calls() -> None:
    space = _absolute_space()
    # frac 0.5 pins the stress case where the per-step size sits exactly at
    # the backstop boundary (the 0.1 default runs far below it, and its
    # full-range second move would hit the per-call playout cap).
    toolset = build_toolset(space, _absolute_obs_space(), control_hz=10.0, max_speed_frac=0.5)
    chain = ChainApprover(ClampApprover(space), DeltaLimitApprover(space))
    store: dict[str, Any] = {}

    first = toolset.execute(
        _call("move_joints", targets={"joint": 1.0}),
        _obs({"q": np.array([-0.5])}),
    )
    second = toolset.execute(
        _call("move_joints", targets={"joint": -1.0}),
        _obs({"q": np.array([1.0])}),
    )
    assert first.chunk is not None and second.chunk is not None
    for action in (*first.chunk.actions, *second.chunk.actions):
        assert chain.review(action, store) is action


# --- displacement-mode synthesis ----------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "delta", "expected_steps", "expected_value"),
    [
        (-0.1, 0.05, 0.2, 5, 0.04),
        (-0.1, 0.05, -0.2, 3, -0.2 / 3),
    ],
)
def test_move_by_splits_by_directional_box_side(
    low: float, high: float, delta: float, expected_steps: int, expected_value: float
) -> None:
    space = _delta_space(low=np.array([low, -0.1]), high=np.array([high, 0.1]))
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": delta}), _obs({}))
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == expected_steps
    assert all(
        float(action.data[0]) == pytest.approx(expected_value) for action in result.chunk.actions
    )


def test_move_by_headroom_avoids_one_ulp_box_overrun() -> None:
    limit = 0.8137309994111953
    delta = 4.068654997055977
    space = _delta_space(high=np.array([limit, 0.1]))
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": delta}), _obs({}))
    assert result.chunk is not None
    assert len(result.chunk.actions) == 6
    assert all(abs(float(action.data[0])) <= limit for action in result.chunk.actions)


def test_move_by_all_zero_is_one_step_hold() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": 0.0, "1": 0.0}), _obs({}))
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == 1
    assert np.array_equal(result.chunk.actions[0].data, np.zeros(2))


@pytest.mark.parametrize(
    ("low", "high", "delta"),
    [(0.0, 0.1, -0.01), (-0.1, 0.0, 0.01)],
)
def test_move_by_rejects_zero_bound_direction(low: float, high: float, delta: float) -> None:
    space = _delta_space(low=np.array([low, -0.1]), high=np.array([high, 0.1]))
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": delta}), _obs({}))
    assert result.error == "dimension 0 cannot move in that direction"


def test_move_by_ignores_max_speed_frac() -> None:
    lengths = []
    for frac in (0.5, 0.1):
        toolset = build_toolset(
            _delta_space(), ObservationSpace(), control_hz=10.0, max_speed_frac=frac
        )
        result = toolset.execute(_call("move_by", deltas={"0": 0.2}), _obs({}))
        assert result.chunk is not None
        lengths.append(len(result.chunk.actions))
    assert lengths == [3, 3]


def test_move_by_over_cap_errors_and_boundary_succeeds() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    too_large = toolset.execute(_call("move_by", deltas={"0": 10.0}), _obs({}))
    assert too_large.chunk is None
    assert too_large.error is not None and "split the move into smaller motions" in too_large.error

    boundary = toolset.execute(_call("move_by", deltas={"0": 9.9}), _obs({}))
    assert boundary.error is None and boundary.chunk is not None
    assert len(boundary.chunk.actions) == 100


def test_huge_finite_move_by_returns_cap_error() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": 1e308}), _obs({}))
    assert result.chunk is None
    assert result.error is not None and "split the move into smaller motions" in result.error


def test_subnormal_delta_underflow_is_a_structured_error() -> None:
    space = _delta_space(low=np.array([-5e-324, -0.1]), high=np.array([5e-324, 0.1]))
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", deltas={"0": 5e-324}), _obs({}))
    assert result.chunk is None
    assert result.error is not None and "too small to split" in result.error


def test_arbitrary_precision_json_integer_is_a_structured_error() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    # 10**400 overflows float() and crashes np.isfinite; both int sizes must
    # come back as errors the LLM can correct, never exceptions.
    overflowing = toolset.execute(_call("move_by", deltas={"0": 10**400}), _obs({}))
    assert overflowing.chunk is None
    assert overflowing.error is not None and "must be a finite number" in overflowing.error

    representable = toolset.execute(_call("move_by", deltas={"0": 10**100}), _obs({}))
    assert representable.chunk is None
    assert representable.error is not None and "split the move" in representable.error


# --- done, notes, and structured errors ---------------------------------------


def test_missing_note_error_precedes_values_validation() -> None:
    """A call wrong in both ways reports the note error first (plan 0021 ordering)."""
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    call = ToolCall(
        id="call_1",
        name="move_by",
        arguments=json.dumps({"deltas": {"not_a_dimension": 0.05}}),
    )
    result = toolset.execute(call, _obs())
    assert result.chunk is None
    assert result.error == _NOTE_ERROR


def test_missing_move_note_is_a_structured_error_in_both_modes() -> None:
    absolute = build_toolset(_absolute_space(), _absolute_obs_space(), control_hz=10.0)
    absolute_call = ToolCall(
        id="call_1",
        name="move_joints",
        arguments=json.dumps({"targets": {"joint": 0.1}}),
    )
    absolute_result = absolute.execute(absolute_call, _obs())
    assert absolute_result.chunk is None
    assert absolute_result.error == _NOTE_ERROR

    displacement = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    displacement_call = ToolCall(
        id="call_1",
        name="move_by",
        arguments=json.dumps({"deltas": {"0": 0.05}}),
    )
    displacement_result = displacement.execute(displacement_call, _obs({}))
    assert displacement_result.chunk is None
    assert displacement_result.error == _NOTE_ERROR


@pytest.mark.parametrize("note", ["", "   ", 42])
def test_invalid_move_note_is_a_structured_error(note: object) -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(_call("move_by", note=note, deltas={"0": 0.05}), _obs({}))
    assert result.chunk is None
    assert result.error == _NOTE_ERROR


def test_valid_move_note_does_not_change_the_motion_chunk() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(
        _call(
            "move_by",
            note="The end effector is short of the target, so I will move it along x.",
            deltas={"0": 0.05},
        ),
        _obs({}),
    )
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == 1
    assert result.chunk.actions[0].data.tobytes() == np.array([0.05, 0.0]).tobytes()


def test_done_and_give_up_emit_control_mode_hold() -> None:
    calls = 0

    def reject(waypoints: npt.NDArray[np.float64]) -> str | None:
        nonlocal calls
        calls += 1
        return "all motion rejected"

    absolute = build_toolset(
        _bimanual_space(),
        _bimanual_obs_space(),
        control_hz=10.0,
        pre_check=reject,
    )
    state = np.full(14, 0.3)
    for name, detail_key in (("done", "summary"), ("give_up", "reason")):
        result = absolute.execute(
            _call(name, **{detail_key: "fork placed"}),
            Observation(state={"joint_pos": state}),
        )
        assert result.error is None and result.chunk is not None
        (action,) = result.chunk.actions
        assert np.array_equal(action.data, state)
        assert action.meta["request_stop"] is True
        assert action.meta["stop_reason"] == name
    assert calls == 0

    displacement = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = displacement.execute(_call("give_up", reason="cannot see"), _obs({}))
    assert result.chunk is not None
    assert np.array_equal(result.chunk.actions[0].data, np.zeros(2))
    assert result.chunk.actions[0].meta["stop_reason"] == "give_up"


def test_done_carries_hindsight_in_stop_meta() -> None:
    toolset = build_toolset(_absolute_space(), _absolute_obs_space(), control_hz=10.0)
    hindsight = "the jar lid needs two approach angles"

    result = toolset.execute(
        _call("done", summary="jar opened", hindsight=hindsight),
        _obs(),
    )

    assert result.error is None and result.chunk is not None
    (action,) = result.chunk.actions
    assert action.meta == {
        "request_stop": True,
        "stop_reason": "done",
        "stop_detail": "jar opened",
        "stop_hindsight": hindsight,
    }


@pytest.mark.parametrize(
    "arguments",
    [{"reason": "cannot see"}, {"reason": "cannot see", "hindsight": "  "}],
)
def test_give_up_accepts_absent_or_blank_hindsight(arguments: dict[str, object]) -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)

    result = toolset.execute(_call("give_up", **arguments), _obs({}))

    assert result.error is None and result.chunk is not None
    assert result.chunk.actions[0].meta == {
        "request_stop": True,
        "stop_reason": "give_up",
        "stop_detail": "cannot see",
    }
    assert result.note == "give_up: cannot see"


def test_tool_errors_are_messages_not_exceptions() -> None:
    toolset = build_toolset(_bimanual_space(), _bimanual_obs_space(), control_hz=10.0)
    cases = [
        _call(
            "move_joints",
            note="The arm is centered, so I will test an invalid named joint.",
            targets={"left_elbow": 0.1},
        ),
        _call("move_joints", targets={"left_j0": float("nan")}),
        _call("move_joints", targets={"left_j0": "fast"}),
        _call("move_joints", targets={}),
        _call("nonexistent_tool", x=1),
        ToolCall(id="c", name="move_joints", arguments="{not json"),
        ToolCall(id="c", name="move_joints", arguments="[]"),
    ]
    for call in cases:
        result = toolset.execute(call, Observation(state={"joint_pos": np.zeros(14)}))
        assert result.chunk is None and result.error, f"expected error for {call}"
    assert "left_elbow" in str(
        toolset.execute(cases[0], Observation(state={"joint_pos": np.zeros(14)})).error
    )


def test_stray_duration_key_is_ignored() -> None:
    toolset = build_toolset(_delta_space(), ObservationSpace(), control_hz=10.0)
    result = toolset.execute(
        _call(
            "move_by",
            note="The target is ahead, so I will move slightly along x.",
            deltas={"0": 0.05},
            duration_s=0.001,
        ),
        _obs({}),
    )
    assert result.error is None and result.chunk is not None


def test_control_hz_none_uses_fallback_without_seconds_note() -> None:
    result = _execute_absolute(1.0, control_hz=None)
    assert result.error is None and result.chunk is not None
    assert len(result.chunk.actions) == 51
    assert result.chunk.control_hz is None
    assert result.note == "executing move_joints over 51 steps"


def test_declared_rate_note_divides_steps_by_hz() -> None:
    result = _execute_absolute(0.5, control_hz=20.0)
    assert result.chunk is not None
    assert len(result.chunk.actions) == 51
    assert result.note == "executing move_joints over 51 steps (2.5s)"


def _pose_space() -> Box:
    return Box(
        shape=(4,),
        low=np.array([0.1, -0.3, 0.03, -np.pi]),
        high=np.array([0.5, 0.3, 0.4, np.pi]),
        semantics=ActionSemantics(
            "eef_abs_pose",
            rotation_repr="none",
            frame="base",
            dim_labels=("x", "y", "z", "yaw"),
        ),
    )


def test_pose_mode_tool_is_move_to_with_cartesian_contracts() -> None:
    obs_space = ObservationSpace(state=StateSpec(fields=(StateField(key="eef", shape=(4,)),)))
    toolset = build_toolset(_pose_space(), obs_space, control_hz=10.0)
    move = toolset.schemas()[0]["function"]
    assert move["name"] == "move_to"
    description = move["description"]
    assert "end-effector" in description
    assert "its own base frame" in description
    assert "relative to the trial's start orientation" in description
    assert "without wrapping" in description
    assert "Per-dimension bounds" in description
    # The executor accepts the renamed tool and still rejects the old name.
    result = toolset.execute(
        _call("move_to", targets={"x": 0.3}),
        Observation(state={"eef": np.array([0.2, 0.0, 0.1, 0.0])}),
    )
    assert result.error is None and result.chunk is not None
    stale = toolset.execute(
        _call("move_joints", targets={"x": 0.3}),
        Observation(state={"eef": np.array([0.2, 0.0, 0.1, 0.0])}),
    )
    assert stale.error is not None and "move_to" in stale.error


def test_joint_mode_tool_name_and_description_unchanged() -> None:
    toolset = build_toolset(_absolute_space(), _absolute_obs_space(), control_hz=10.0)
    move = toolset.schemas()[0]["function"]
    assert move["name"] == "move_joints"
    assert "end-effector" not in move["description"]
    assert "start orientation" not in move["description"]


def test_displacement_pose_mode_keeps_move_by() -> None:
    space = Box(
        shape=(3,),
        low=np.array([-0.05, -0.05, -0.1]),
        high=np.array([0.05, 0.05, 0.1]),
        semantics=ActionSemantics(
            "eef_delta_pose", rotation_repr="none", dim_labels=("dx", "dy", "dyaw")
        ),
    )
    toolset = build_toolset(space, ObservationSpace(), control_hz=10.0)
    move = toolset.schemas()[0]["function"]
    assert move["name"] == "move_by"
    assert move["description"].startswith("Move BY")
