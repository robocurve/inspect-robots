"""Cleanup failures and diagnostic failures must never replace a stop signal."""

from __future__ import annotations

import warnings

import pytest

from inspect_robots._parallel import HaltPreservingExitStack
from inspect_robots.errors import EmbodimentFault, SafetyAbort


@pytest.mark.parametrize("error", [SafetyAbort, EmbodimentFault, KeyboardInterrupt])
def test_warning_as_error_does_not_mask_halt_or_skip_cleanup(
    error: type[BaseException],
) -> None:
    halt = error("original halt")
    closed: list[str] = []

    def close(*, name: str) -> None:
        closed.append(name)
        raise RuntimeError(f"{name} cleanup failed")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(error) as raised, HaltPreservingExitStack() as resources:
            assert resources.callback(close, name="policy") is close
            resources.callback(close, name="embodiment")
            raise halt
    assert raised.value is halt
    assert closed == ["embodiment", "policy"]


@pytest.mark.parametrize("error", [SafetyAbort, EmbodimentFault, KeyboardInterrupt])
def test_cleanup_originated_halt_survives_later_cleanup_failure(
    error: type[BaseException],
) -> None:
    halt = error("cleanup requested halt")
    closed: list[str] = []

    def close_policy() -> None:
        closed.append("policy")
        raise RuntimeError("policy cleanup failed")

    def close_embodiment() -> None:
        closed.append("embodiment")
        raise halt

    with (
        pytest.warns(RuntimeWarning, match="RuntimeError: policy cleanup failed"),
        pytest.raises(error) as raised,
        HaltPreservingExitStack() as resources,
    ):
        resources.callback(close_policy)
        resources.callback(close_embodiment)
    assert raised.value is halt
    assert closed == ["embodiment", "policy"]
