"""Adapter conformance: mechanical checks on an embodiment's declared spaces.

Plan 0008 made two consumers of *declarations* load-bearing: the CLI's
default guardrails derive their limits from the action space, and the LLM
agent policy builds its whole tool surface from the spaces at bind time. An
adapter with missing semantics, missing bounds, unlabeled dims, or a
misaligned ``StateSpec`` silently degrades both. This module turns those
requirements into a checkable report so adapter repos can enforce them in CI
(one test: ``assert_embodiment_conformant(MyEmbodiment().info)``) and users
can audit an installed adapter via ``inspect-robots doctor``.

The checks are purely declarative — nothing here touches hardware — which is
also their limit: conformance proves an adapter is *guardrail-ready and
agent-ready*, not that its declarations are honest (a delta rig declaring
absolute-sized per-step bounds type-checks fine). The adapter authoring
guide covers the human half.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from inspect_robots.embodiment import EmbodimentInfo

_ABSOLUTE_MODES = frozenset({"joint_pos", "eef_abs_pose"})


@dataclass(frozen=True)
class ConformanceIssue:
    """One finding: ``severity`` is ``"error"`` (fails the check) or ``"warning"``."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ConformanceReport:
    """All findings for one embodiment; ``ok`` iff there are no errors."""

    embodiment: str
    issues: tuple[ConformanceIssue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether all declarations satisfy the required invariants."""
        return not any(i.severity == "error" for i in self.issues)

    def summary(self) -> str:
        """Render a CLI-ready multiline summary of the findings."""
        if not self.issues:
            return f"{self.embodiment}: conformant (no issues)"
        lines = [f"{self.embodiment}: {len(self.issues)} issue(s)"]
        lines += [f"  [{i.severity}] {i.code}: {i.message}" for i in self.issues]
        return "\n".join(lines)


def check_embodiment(info: EmbodimentInfo) -> ConformanceReport:
    """Check an embodiment's declarations against the plan-0008 requirements.

    Errors: missing action semantics; missing/non-finite bounds; missing or
    duplicate ``dim_labels``; absolute-target modes without exactly one
    ``StateSpec`` field shaped like the action space; a space the default
    guardrail chain refuses to limit. Warnings: ``control_hz`` undeclared
    (agent motion falls back to 10 Hz step counting) and zero-width bound
    dims. Purely declarative — safe to run anywhere, no hardware touched.
    """
    issues: list[ConformanceIssue] = []
    space = info.action_space
    semantics = space.semantics

    def error(code: str, message: str) -> None:
        issues.append(ConformanceIssue("error", code, message))

    def warning(code: str, message: str) -> None:
        issues.append(ConformanceIssue("warning", code, message))

    if semantics is None:
        error(
            "semantics",
            "action space declares no ActionSemantics; guardrails and the agent "
            "policy cannot tell absolute targets from displacements",
        )

    low, high = space.low, space.high
    if low is None or high is None or not bool(np.all(np.isfinite(high - low))):
        error(
            "bounds",
            "action space needs finite low/high bounds; without them the bounds "
            "clamp is skipped and no default delta limit can be derived",
        )
    elif bool(np.any(high == low)):
        warning("zero_width", "some action dims have low == high (zero commandable range)")

    if semantics is not None:
        labels = semantics.dim_labels
        if labels is None:
            error(
                "dim_labels",
                "action dims are unlabeled; label-addressed tooling (the LLM agent "
                "policy, logging) falls back to bare indices",
            )
        elif len(set(labels)) != len(labels):
            error("dim_labels", "dim_labels contains duplicates")

        if semantics.control_mode in _ABSOLUTE_MODES:
            spec = info.observation_space.state
            matching = [f.key for f in spec.fields if f.shape == (space.dim,)] if spec else []
            if len(matching) != 1:
                error(
                    "state_alignment",
                    "absolute-target control needs exactly one StateSpec field with "
                    f"shape ({space.dim},) as the proprioceptive reference; found "
                    f"{matching or 'none'}",
                )

        if low is not None and high is not None:
            from inspect_robots.approver import DeltaLimitApprover

            try:
                DeltaLimitApprover(space)
            except ValueError as exc:
                # Same refusal the CLI degradation path reports; here it is an
                # error because a conformant adapter must be limitable.
                error("guardrails", str(exc))

    if info.control_hz is None:
        warning(
            "control_hz",
            "control_hz is undeclared; agent motion durations fall back to 10 Hz step counting",
        )

    return ConformanceReport(embodiment=info.name, issues=tuple(issues))


def assert_embodiment_conformant(info: EmbodimentInfo) -> None:
    """Pytest-friendly wrapper: raise ``AssertionError`` with the full summary."""
    report = check_embodiment(info)
    if not report.ok:
        raise AssertionError(report.summary())
