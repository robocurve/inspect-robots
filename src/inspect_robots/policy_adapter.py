"""Policy adapters: how a multimodal language model is made into a policy.

The paper extends the policy abstraction to language models with a policy
adapter that supplies four things: preprocessed observations, instructions for
operating the current embodiment, a library of control tools and a library of
termination tools. A [`PolicyAdapter`][inspect_robots.policy_adapter.PolicyAdapter]
owns those four; a policy built on one owns only its model: prompting, the
provider's wire format and the conversation.

[`ToolAdapter`][inspect_robots.policy_adapter.ToolAdapter] is the reference
adapter. It builds [control and termination tools][inspect_robots.tools] from the
embodiment's spaces, adds any tools the embodiment contributes, renders the
embodiment's notes, bounds and pinned dimensions as instructions, and renders
an observation's state, setpoint and embodiment reports as text lines. The
helpers it uses (``state_lines``, ``observation_lines`` and
``embodiment_notes``) are public so other adapters render the same facts the
same way.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.tools import (
    ControlTools,
    PreCheck,
    ToolLibrary,
    control_tools,
    termination_tools,
)
from inspect_robots.types import Observation


@runtime_checkable
class PolicyAdapter(Protocol):
    """What a language-model policy needs from the embodiment it is bound to."""

    def bind(self, info: EmbodimentInfo) -> None:
        """Build tools and instructions for ``info``; raise ``ToolError`` if it cannot be driven."""
        ...

    def add_tools(self, tools: ToolLibrary) -> None:
        """Add tools the embodiment contributes; they are listed after the built-in ones."""
        ...

    @property
    def tools(self) -> ToolLibrary:
        """Every tool the model may call: control, termination and contributed."""
        ...

    def instructions(self) -> str:
        """How to operate this embodiment, as text for the model's system prompt."""
        ...

    def observation_lines(self, observation: Observation) -> list[str]:
        """The observation's state and embodiment reports as text lines for the model."""
        ...

    def start_trial(self) -> None:
        """Forget per-trial state (such as the held setpoint) at a trial reset."""
        ...


def embodiment_notes(info: EmbodimentInfo) -> str | None:
    """The embodiment's operating notes as an instructions section, or ``None`` without any."""
    docs = getattr(info, "docs", None)
    if not isinstance(docs, str) or not docs.strip():
        return None
    return "Embodiment notes:\n" + docs.strip()


def state_lines(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """One ``state[key]: ...`` line per state entry, values rounded to 4 decimals.

    The entry named by ``state_labels`` (a key and one label per element) is
    written as ``label=value`` pairs when its shape matches.
    """
    lines: list[str] = []
    for key, value in observation.state.items():
        array = np.asarray(value, dtype=np.float64)
        rounded = np.round(array, 4).tolist()
        if state_labels is not None and key == state_labels[0]:
            labels = state_labels[1]
            if array.shape == (len(labels),):
                labeled = " ".join(
                    f"{label}={item}" for label, item in zip(labels, rounded, strict=True)
                )
                lines.append(f"state[{key}]: {labeled}")
                continue
        lines.append(f"state[{key}]: {rounded}")
    return lines


def approvals_line(observation: Observation) -> str | None:
    """Summarize the steps the guard modified since the last policy call, or ``None``."""
    approvals = observation.extra.get("approvals")
    if not isinstance(approvals, list) or not approvals:
        return None
    total = len(approvals)
    details = [str(a.get("detail")) for a in approvals if isinstance(a, dict) and a.get("detail")]
    if not details:
        return f"approver: {total} step(s) modified."
    counts = Counter(details)
    formatted: list[str] = []
    for flag, count in counts.items():
        if count > 1:
            formatted.append(f"{flag} \u00d7{count}")
        else:
            formatted.append(flag)
    detail_str = f" ({', '.join(formatted)})"
    return f"approver: {total} step(s) modified{detail_str}."


def extra_lines(observation: Observation, key: str) -> list[str]:
    """Render ``extra[key]["lines"]``, tolerating absent or malformed data.

    The convention is embodiment-agnostic: a mapping whose ``lines`` are
    plain-language strings written for the policy (for example a per-arm motion
    diagnosis under ``motion_report``, or the hottest motors under
    ``motor_temps``). Anything else is ignored silently.
    """
    report = observation.extra.get(key)
    if not isinstance(report, Mapping):
        return []
    lines = report.get("lines")
    if not isinstance(lines, list):
        return []
    return [line.strip() for line in lines if isinstance(line, str) and line.strip()]


def operator_lines(observation: Observation) -> list[str]:
    """Render only well-formed feedback entries from the framework-reserved channel."""
    messages = observation.extra.get("operator_messages")
    if not isinstance(messages, list):
        return []
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        step = message.get("t")
        text = message.get("text")
        if not isinstance(step, int) or not isinstance(text, str):
            continue
        lines.append(f"operator feedback (step {step}): {text}")
    return lines


def observation_lines(
    observation: Observation,
    state_labels: tuple[str, tuple[str, ...]] | None = None,
    setpoint_lines: Sequence[str] = (),
) -> list[str]:
    """State, setpoint, guard summary, motion report, motor temperatures and operator feedback.

    The order keeps the readback next to the setpoint it is compared against,
    and the embodiment's own reports before live operator feedback.
    """
    lines = state_lines(observation, state_labels)
    lines.extend(setpoint_lines)
    approvals = approvals_line(observation)
    if approvals is not None:
        lines.append(approvals)
    lines.extend(extra_lines(observation, "motion_report"))
    lines.extend(extra_lines(observation, "motor_temps"))
    lines.extend(operator_lines(observation))
    return lines


class ToolAdapter:
    """The reference policy adapter: tools and text built from the embodiment's declarations.

    ``max_speed_frac``, ``images`` and ``pre_check`` configure
    [`control_tools`][inspect_robots.tools.control_tools].
    """

    def __init__(
        self,
        *,
        max_speed_frac: float = 0.1,
        images: str = "always",
        pre_check: PreCheck | None = None,
    ) -> None:
        self._max_speed_frac = max_speed_frac
        self._images = images
        self._pre_check = pre_check
        self._info: EmbodimentInfo | None = None
        self._control: ControlTools | None = None
        self._contributed = ToolLibrary()
        self._tools: ToolLibrary | None = None

    def bind(self, info: EmbodimentInfo) -> None:
        """Build the control and termination tools for ``info``; contributed tools are dropped."""
        self._control = control_tools(
            info.action_space,
            info.observation_space,
            info.control_hz,
            self._max_speed_frac,
            images=self._images,
            pre_check=self._pre_check,
        )
        self._info = info
        self._contributed = ToolLibrary()
        self._tools = None

    @property
    def control(self) -> ControlTools:
        """The bound control tools; raises ``RuntimeError`` before ``bind``."""
        if self._control is None:
            raise RuntimeError("ToolAdapter used before bind(); bind it to the embodiment first")
        return self._control

    def add_tools(self, tools: ToolLibrary) -> None:
        """Add the embodiment's tools after the built-in ones; a name clash raises ``ToolError``."""
        combined = self._contributed + tools
        _ = self.control.library + termination_tools(self.control) + combined
        self._contributed = combined
        self._tools = None

    @property
    def tools(self) -> ToolLibrary:
        """Control, termination and contributed tools, in listing order."""
        if self._tools is None:
            control = self.control
            self._tools = control.library + termination_tools(control) + self._contributed
        return self._tools

    def instructions(self) -> str:
        """The embodiment's notes (when it has any), bounds, and pinned dimensions."""
        control = self.control
        sections: list[str] = []
        notes = embodiment_notes(self._info) if self._info is not None else None
        if notes is not None:
            sections.append(notes)
        bounds = f"Embodiment bounds:\n{control.bounds_text}"
        pinned = control.pinned_labels
        if pinned:
            bounds += (
                f"\nPinned dimensions: {', '.join(pinned)} (fixed; do not attempt to move them)."
            )
        sections.append(bounds)
        return "\n\n".join(sections)

    def observation_lines(self, observation: Observation) -> list[str]:
        """The observation as text lines, with the held setpoint after the readback."""
        control = self.control
        return observation_lines(
            observation, control.state_labels(), control.setpoint_lines(observation)
        )

    def start_trial(self) -> None:
        """Forget the held setpoint so the next move starts from the measured state."""
        self.control.start_trial()
