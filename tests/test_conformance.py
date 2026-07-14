"""Adapter conformance checks: declared spaces must be guardrail- and agent-ready."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from inspect_robots.conformance import (
    assert_embodiment_conformant,
    check_embodiment,
    missing_runtime_requirements,
)
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    ObservationSpace,
    StateField,
    StateSpec,
)


def _info(
    *,
    space: Box,
    obs: ObservationSpace | None = None,
    control_hz: float | None = 10.0,
) -> EmbodimentInfo:
    return EmbodimentInfo(
        name="fixture",
        action_space=space,
        observation_space=obs or ObservationSpace(),
        control_hz=control_hz,
    )


def _good_absolute() -> EmbodimentInfo:
    return _info(
        space=Box(
            shape=(2,),
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            semantics=ActionSemantics("joint_pos", dim_labels=("j0", "gripper")),
        ),
        obs=ObservationSpace(state=StateSpec(fields=(StateField(key="joint_pos", shape=(2,)),))),
    )


def _codes(info: EmbodimentInfo) -> dict[str, str]:
    return {i.code: i.severity for i in check_embodiment(info).issues}


def test_runtime_requirements_absent_attribute_is_empty() -> None:
    class _Factory:
        pass

    assert missing_runtime_requirements(_Factory) == {}
    assert missing_runtime_requirements(None) == {}


def test_runtime_requirements_all_present_is_empty() -> None:
    class _Factory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {"os": "install operating system"}

    assert missing_runtime_requirements(_Factory) == {}


def test_runtime_requirements_return_missing_entries_in_declaration_order() -> None:
    class _Factory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
            "definitely_missing_xyz_one": "install one",
            "os": "already present",
            "definitely_missing_xyz_two": "install two",
        }

    assert list(missing_runtime_requirements(_Factory).items()) == [
        ("definitely_missing_xyz_one", "install one"),
        ("definitely_missing_xyz_two", "install two"),
    ]


def test_runtime_requirement_with_missing_parent_is_missing() -> None:
    class _Factory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
            "definitely_missing_xyz.sub": "install parent"
        }

    assert missing_runtime_requirements(_Factory) == {
        "definitely_missing_xyz.sub": "install parent"
    }


def test_runtime_requirement_probe_exception_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_oserror(_name: str, _package: str | None = None) -> None:
        raise OSError("broken parent package")

    monkeypatch.setattr("inspect_robots.conformance.importlib.util.find_spec", raise_oserror)

    class _Factory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {"broken": "repair it"}

    assert missing_runtime_requirements(_Factory) == {"broken": "repair it"}


def test_non_mapping_runtime_requirements_are_ignored() -> None:
    class _Factory:
        RUNTIME_REQUIREMENTS: ClassVar[list[str]] = ["definitely_missing_xyz"]

    assert missing_runtime_requirements(_Factory) == {}


def test_non_string_runtime_requirement_entries_are_ignored() -> None:
    class _Factory:
        RUNTIME_REQUIREMENTS: ClassVar[dict[object, object]] = {
            7: "ignored key",
            "os": 9,
            "definitely_missing_xyz": "install valid",
        }

    assert missing_runtime_requirements(_Factory) == {"definitely_missing_xyz": "install valid"}


def test_good_absolute_and_displacement_pass() -> None:
    report = check_embodiment(_good_absolute())
    assert report.ok and not report.issues

    displacement = _info(
        space=Box(
            shape=(2,),
            low=np.array([-0.1, -0.1]),
            high=np.array([0.1, 0.1]),
            semantics=ActionSemantics("eef_delta_pos", dim_labels=("dx", "dy")),
        )
    )
    assert check_embodiment(displacement).ok


def test_cubepick_is_conformant() -> None:
    # Dogfood: our own mock world must pass the kit it ships next to.
    assert_embodiment_conformant(CubePickEmbodiment().info)


def test_missing_semantics_is_an_error() -> None:
    info = _info(space=Box(shape=(2,), low=np.zeros(2), high=np.ones(2)))
    assert _codes(info)["semantics"] == "error"


def test_missing_or_nonfinite_bounds_is_an_error() -> None:
    unbounded = _info(space=Box(shape=(2,), semantics=ActionSemantics("joint_pos")))
    assert _codes(unbounded)["bounds"] == "error"
    half = _info(space=Box(shape=(2,), low=np.zeros(2), semantics=ActionSemantics("joint_pos")))
    assert _codes(half)["bounds"] == "error"
    inf = _info(
        space=Box(
            shape=(2,),
            low=np.zeros(2),
            high=np.array([1.0, np.inf]),
            semantics=ActionSemantics("joint_pos"),
        )
    )
    assert _codes(inf)["bounds"] == "error"


def test_missing_or_duplicate_labels_is_an_error() -> None:
    unlabeled = _info(
        space=Box(
            shape=(2,),
            low=np.zeros(2),
            high=np.ones(2),
            semantics=ActionSemantics("eef_delta_pos"),
        )
    )
    assert _codes(unlabeled)["dim_labels"] == "error"
    duplicated = _info(
        space=Box(
            shape=(2,),
            low=np.zeros(2),
            high=np.ones(2),
            semantics=ActionSemantics("eef_delta_pos", dim_labels=("a", "a")),
        )
    )
    assert _codes(duplicated)["dim_labels"] == "error"


def test_absolute_mode_needs_exactly_one_aligned_state_field() -> None:
    space = Box(
        shape=(2,),
        low=np.zeros(2),
        high=np.ones(2),
        semantics=ActionSemantics("joint_pos", dim_labels=("a", "b")),
    )
    no_spec = _info(space=space)
    assert _codes(no_spec)["state_alignment"] == "error"
    two = _info(
        space=space,
        obs=ObservationSpace(
            state=StateSpec(
                fields=(StateField(key="x", shape=(2,)), StateField(key="y", shape=(2,)))
            )
        ),
    )
    assert _codes(two)["state_alignment"] == "error"
    # Displacement modes have no alignment requirement.
    delta = _info(
        space=Box(
            shape=(2,),
            low=np.zeros(2),
            high=np.ones(2),
            semantics=ActionSemantics("joint_delta", dim_labels=("a", "b")),
        )
    )
    assert "state_alignment" not in _codes(delta)


def test_unlimitable_rotation_repr_is_an_error() -> None:
    info = _info(
        space=Box(
            shape=(7,),
            low=np.full(7, -1.0),
            high=np.full(7, 1.0),
            semantics=ActionSemantics(
                "eef_abs_pose",
                rotation_repr="quat_wxyz",
                dim_labels=tuple("abcdefg"),
            ),
        ),
        obs=ObservationSpace(state=StateSpec(fields=(StateField(key="pose", shape=(7,)),))),
    )
    codes = _codes(info)
    assert codes["guardrails"] == "error"


def test_missing_control_hz_is_a_warning() -> None:
    info = _good_absolute()
    silent = EmbodimentInfo(
        name="fixture",
        action_space=info.action_space,
        observation_space=info.observation_space,
        control_hz=None,
    )
    codes = _codes(silent)
    assert codes["control_hz"] == "warning"
    assert check_embodiment(silent).ok  # warnings never fail the check


def test_zero_width_dims_are_a_warning() -> None:
    info = _info(
        space=Box(
            shape=(2,),
            low=np.array([0.0, 0.5]),
            high=np.array([1.0, 0.5]),
            semantics=ActionSemantics("joint_pos", dim_labels=("a", "b")),
        ),
        obs=ObservationSpace(state=StateSpec(fields=(StateField(key="q", shape=(2,)),))),
    )
    assert _codes(info)["zero_width"] == "warning"


def test_assert_helper_raises_with_summary() -> None:
    bad = _info(space=Box(shape=(2,)))
    with pytest.raises(AssertionError, match="semantics"):
        assert_embodiment_conformant(bad)


def test_report_summary_is_readable() -> None:
    report = check_embodiment(_info(space=Box(shape=(2,))))
    text = report.summary()
    assert "error" in text and "semantics" in text
