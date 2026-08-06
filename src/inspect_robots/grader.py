"""Judgement capture as a component: the capture side of R6 (plan 0049).

R6 splits judging in two. A grader captures the success judgement once per
scored trial, after the rollout returns and before scorers run; it may be
interactive (a human prompt) or expensive (a VLM call), and it mutates the
[`TrialRecord`][inspect_robots.rollout.TrialRecord]. Scorers stay pure readers
of the record, so re-scoring a saved log remains deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Scene
    from inspect_robots.session import OperatorSession


@runtime_checkable
class Grader(Protocol):
    """Capture a judgement onto a trial record, once, before scorers run.

    ``grade`` is called only for trials that will be scored (never for errored
    or cancelled trials) and may mutate the record: set
    ``operator_judgement``/``operator_note`` and append an operator event.
    It must tolerate being unable to grade (e.g. no judge available) by
    leaving the record unchanged rather than raising.
    """

    name: str
    """Registry name identifying this grader in logs and CLI output."""

    def grade(self, record: TrialRecord, scene: Scene) -> None:
        """Capture (or decline to capture) a judgement for one trial."""
        ...


class _OperatorGrader:
    """Prompt the terminal operator for a verdict via an ``OperatorSession``."""

    name = "operator"

    def __init__(self, session: OperatorSession | None = None) -> None:
        self._session = session

    def connect_session(self, session: OperatorSession) -> None:
        """Adopt the run's session so prompts share the run's terminal seams."""
        self._session = session

    def grade(self, record: TrialRecord, scene: Scene) -> None:
        """Delegate to ``OperatorSession.prompt_verdict`` (R6 capture).

        Without a connected session (e.g. ``eval(grader="operator")`` from the
        Python API), a default ``OperatorSession`` is constructed lazily on
        first use; on dead stdin its prompt degrades to a no-op, so the
        judgement simply stays ``None``.
        """
        if self._session is None:
            from inspect_robots.session import OperatorSession

            self._session = OperatorSession()
        self._session.prompt_verdict(record, scene)


def operator_grader(session: OperatorSession | None = None) -> _OperatorGrader:
    """The builtin human grader: prompt for a verdict and optional notes.

    Adopts a console- or embodiment-definitive verdict without re-asking, and
    never blocks a run whose stdin cannot answer (EOF degrades to no verdict).
    """
    return _OperatorGrader(session)
