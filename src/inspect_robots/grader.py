"""Judgement capture as a component: the capture side of R6 (plan 0049).

R6 splits judging in two. A grader captures the success judgement once per
scored trial, after the rollout returns and before scorers run; it may be
interactive (a human prompt) or expensive (a VLM call), and it mutates the
[`TrialRecord`][inspect_robots.rollout.TrialRecord]. Scorers stay pure readers
of the record, so re-scoring a saved log remains deterministic.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from inspect_robots._chatwire import HttpPost, chat_completion
from inspect_robots.errors import ConfigError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from inspect_robots.rollout import StepRecord, TrialRecord
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


_DEFAULT_RUBRIC = (
    "Grade success if the system completed the task instruction, otherwise "
    "grade failure. If the outcome is ambiguous or not visible in the frames, "
    "grade failure."
)
_NOTE_CHAR_LIMIT = 2000
_VERDICT_RE = re.compile(r"GRADE:\s*\**\s*(success|failure)", re.IGNORECASE)
_BOTH_FRAMES_SENTENCE = (
    "You will see the initial frames (before the robot acted) and the final "
    "frames (after the trial ended)."
)
_FINAL_FRAMES_SENTENCE = "You will see the final frames (after the trial ended)."


def _phase_frames(
    step: StepRecord, *, final: bool, max_cameras: int
) -> list[tuple[str, npt.NDArray[np.uint8]]]:
    """Collect one step's named frames for a prompt phase, loading refs on demand.

    Inline observation images win; on-disk ``FrameRef`` handles are loaded
    otherwise (R5 strips inline images when a ``FrameStore`` is active).
    Cameras come back name-sorted and capped so the prompt is deterministic.
    """
    inline = step.result.observation.images if final else step.observation.images
    if inline:
        return [(camera, inline[camera]) for camera in sorted(inline)[:max_cameras]]
    refs = (step.result_image_refs if final else step.image_refs) or {}
    # Cap before loading: skipped cameras cost no disk reads, and a corrupt
    # frame beyond the cap cannot degrade the trial.
    return [(camera, refs[camera].load()) for camera in sorted(refs)[:max_cameras]]


def _parked_frames(
    record: TrialRecord, max_cameras: int
) -> list[tuple[str, npt.NDArray[np.uint8]]]:
    """Collect deterministic, capped frames from a fresh parked observation."""
    observation = record.parked_observation
    if observation is None or not observation.images:
        return []
    return [
        (camera, observation.images[camera]) for camera in sorted(observation.images)[:max_cameras]
    ]


class _VLMGrader:
    """Judge a trial from its first and last frames with a vision model."""

    name = "vlm"

    def __init__(
        self,
        model: str,
        rubric: str,
        *,
        base_url: str,
        api_key: str,
        max_cameras: int,
        http_post: HttpPost | None,
        effort: str | float | None,
    ) -> None:
        self._model = model
        self._rubric = rubric
        self._base_url = base_url
        self._api_key = api_key
        self._max_cameras = max_cameras
        self._http_post = http_post
        self._effort = effort

    def config(self) -> dict[str, Any]:
        """Report the configuration that actually governs each grading call.

        The optional hook ``eval()`` reads into ``EvalSpec.grader_config``.
        Every value is the resolved one this grader applies, not the
        constructor's input: ``rubric`` already has the default substituted
        and any ``rubric_file`` read, and ``effort`` is the normalized value
        that rides the request (``None`` omits ``reasoning_effort`` so the
        provider default applies, ``"none"`` asks for the minimum). ``rubric``
        is the run-level fallback only, since ``_rubric_for`` lets a scene's
        own ``metadata["rubric"]`` win for that scene. The API key is
        deliberately absent: a log is not a place for a credential.
        """
        return {
            "model": self._model,
            "base_url": self._base_url,
            "rubric": self._rubric,
            "max_cameras": self._max_cameras,
            "effort": self._effort,
        }

    def _rubric_for(self, scene: Scene) -> str:
        """Prefer a generated per-scene rubric over the run-level one.

        ``scene.metadata["rubric"]`` is the documented hand-off from task
        generation (plan 0070); it wins over the constructed
        ``rubric``/``rubric_file`` whenever it is a non-blank string.
        """
        scene_rubric = scene.metadata.get("rubric")
        if isinstance(scene_rubric, str) and scene_rubric.strip():
            return scene_rubric
        return self._rubric

    def _messages(
        self,
        instruction: str,
        rubric: str,
        initial: list[tuple[str, npt.NDArray[np.uint8]]],
        final: list[tuple[str, npt.NDArray[np.uint8]]],
    ) -> list[dict[str, Any]]:
        from inspect_robots._pngenc import png_data_url

        frames_sentence = _BOTH_FRAMES_SENTENCE if initial else _FINAL_FRAMES_SENTENCE
        text = (
            "You are grading one robot trial from its camera frames.\n\n"
            f"Task instruction: {instruction}\n\n"
            f"Rubric:\n{rubric}\n\n"
            f"{frames_sentence} Judge the trial against the rubric using only "
            "what is visible. Explain your judgement briefly, then end your "
            "reply with exactly one line:\nGRADE: success\nor\nGRADE: failure"
        )
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for phase, frames in (("initial", initial), ("final", final)):
            for camera, image in frames:
                parts.append({"type": "text", "text": f"{phase} frame, camera {camera!r}"})
                parts.append({"type": "image_url", "image_url": {"url": png_data_url(image)}})
        return [{"role": "user", "content": parts}]

    def grade(self, record: TrialRecord, scene: Scene) -> None:
        """Capture a model judgement, or degrade to an ungraded trial (R6).

        Adopts an existing console verdict or a definitive embodiment
        termination without spending a model call. Any failure after the
        rollout (frames, transport, parsing) prints one stderr note and
        leaves the record unchanged; grading never crashes a finished run.
        """
        from inspect_robots.session import _DEFINITIVE_REASONS
        from inspect_robots.transcript import operator_event

        if record.operator_judgement is not None:
            return
        if record.terminated and record.termination_reason in _DEFINITIVE_REASONS:
            verdict = record.termination_reason
            record.operator_judgement = verdict
            record.events.append(
                operator_event(t=len(record.steps), verdict=verdict, source="embodiment")
            )
            return
        try:
            parked = _parked_frames(record, self._max_cameras)
            final = parked or (
                _phase_frames(record.steps[-1], final=True, max_cameras=self._max_cameras)
                if record.steps
                else []
            )
            if not final:
                raise ConfigError("no final frames recorded")
            initial = (
                _phase_frames(record.steps[0], final=False, max_cameras=self._max_cameras)
                if record.steps
                else []
            )
            messages = self._messages(scene.instruction, self._rubric_for(scene), initial, final)
            reply = chat_completion(
                self._base_url,
                self._api_key,
                self._model,
                messages,
                what="grading",
                fix_hint="check -G model=... and -G base_url=..., then retry",
                http_post=self._http_post,
                effort=self._effort,
            )
            matches = _VERDICT_RE.findall(reply)
            if not matches:
                raise ConfigError(f"no GRADE line in the reply: {reply.strip()[:200]!r}")
        except Exception as exc:  # post-rollout, degrading beats crashing the run
            print(f"vlm grader: {exc}; trial left ungraded", file=sys.stderr)
            return
        judgement = str(matches[-1]).lower()
        note = reply.strip()[:_NOTE_CHAR_LIMIT]
        # Written only alongside a verdict: a degraded (ungraded) trial must
        # leave the record unchanged, marker included.
        if parked:
            record.metadata["graded_frames"] = "parked"
        record.operator_judgement = judgement
        record.operator_note = note
        record.events.append(
            operator_event(t=len(record.steps), verdict=judgement, source="vlm", note=note)
        )


class _Unset:
    """Sentinel distinguishing an omitted ``effort`` from an explicit ``None``."""


_UNSET = _Unset()


def vlm_grader(
    model: str,
    rubric: str | None = None,
    rubric_file: str | None = None,
    *,
    base_url: str = "https://api.anthropic.com/v1",
    api_key_env: str = "ANTHROPIC_API_KEY",
    max_cameras: int = 4,
    http_post: HttpPost | None = None,
    effort: str | float | None | _Unset = _UNSET,
) -> _VLMGrader:
    """The builtin automated grader: a vision model judges first and last frames.

    Configuration fails fast here (missing model, unreadable rubric file,
    unset API key, empty effort) so a misconfigured run stops before any
    rollout; after the rollout its ``grade`` only ever degrades to an
    ungraded trial, so an effort value the endpoint rejects surfaces as a
    per-trial stderr note, not a raise. ``effort`` rides each grading
    request as ``reasoning_effort``: leaving it unset omits the field for
    the provider default, ``None`` (``-G effort=none``) requests the
    minimum, and any other value passes through verbatim. The judgement
    lands on the same record fields the operator grader writes, so the
    ``operator`` scorer reads it unchanged.
    """
    resolved_effort: str | float | None = None
    if not isinstance(effort, _Unset):
        if effort == "":
            raise ConfigError(
                "the vlm grader effort must be a level name or number, got ''.\n"
                "fix: omit -G effort= for the provider default, or pass -G effort=none "
                "for minimum reasoning"
            )
        resolved_effort = "none" if effort is None else effort
    if not model:
        raise ConfigError("the vlm grader needs a model id.\nfix: pass -G model=...")
    if isinstance(max_cameras, bool) or not isinstance(max_cameras, int) or max_cameras < 1:
        raise ConfigError(
            f"max_cameras must be a positive integer, got {max_cameras!r}.\n"
            "fix: pass -G max_cameras=<positive int>"
        )
    if rubric is not None and rubric_file is not None:
        raise ConfigError(
            "rubric and rubric_file are mutually exclusive.\n"
            "fix: pass -G rubric=... or -G rubric_file=..., not both"
        )
    if rubric_file is not None:
        try:
            rubric = Path(rubric_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"cannot read rubric_file {rubric_file!r}: {exc}.\n"
                "fix: pass -G rubric_file=<readable path> or inline -G rubric=..."
            ) from exc
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ConfigError(
            f"no API key found in ${api_key_env}.\n"
            f"fix: set ${api_key_env}, or choose the variable with -G api_key_env=..."
        )
    return _VLMGrader(
        model,
        rubric if rubric is not None else _DEFAULT_RUBRIC,
        base_url=base_url,
        api_key=api_key,
        max_cameras=max_cameras,
        http_post=http_post,
        effort=resolved_effort,
    )
