"""Compatibility checking between a policy and an embodiment.

Before any rollout, RoboInspect verifies that a ``(policy, embodiment)`` pair can
actually run together: the action spaces agree in dimension and semantics, the
embodiment provides every observation the policy requires (resolving a name
remap), the control rates are reconcilable (R1), and — given a task — every
scene is realizable on the embodiment (R7).

Hard mismatches are ``error`` issues that fail fast; soft ones are warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from roboinspect.embodiment import Embodiment
from roboinspect.errors import CompatibilityError
from roboinspect.policy import Policy
from roboinspect.spaces import Box
from roboinspect.task import Task

_RATE_TOL = 1e-6


@dataclass(frozen=True)
class CompatIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str


@dataclass
class CompatibilityReport:
    """The outcome of a compatibility check."""

    issues: list[CompatIssue] = field(default_factory=list)
    remap: dict[str, str] = field(default_factory=dict)

    @property
    def errors(self) -> list[CompatIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[CompatIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            lines = "\n".join(f"  - [{i.code}] {i.message}" for i in self.errors)
            raise CompatibilityError(f"incompatible policy/embodiment:\n{lines}")


def _check_action_spaces(policy_box: Box, emb_box: Box, issues: list[CompatIssue]) -> None:
    if policy_box.dim != emb_box.dim:
        issues.append(
            CompatIssue(
                "error",
                "action_dim",
                f"policy emits {policy_box.dim}-D actions but embodiment expects {emb_box.dim}-D",
            )
        )
    ps, es = policy_box.semantics, emb_box.semantics
    if ps is None or es is None:
        issues.append(
            CompatIssue(
                "warning",
                "action_semantics_unknown",
                "action semantics missing on policy or embodiment; cannot verify control mode",
            )
        )
        return
    if ps.control_mode != es.control_mode:
        issues.append(
            CompatIssue(
                "error",
                "control_mode",
                f"policy control_mode {ps.control_mode!r} != embodiment {es.control_mode!r}",
            )
        )
    if ps.rotation_repr != es.rotation_repr:
        issues.append(
            CompatIssue(
                "error",
                "rotation_repr",
                f"policy rotation_repr {ps.rotation_repr!r} != embodiment {es.rotation_repr!r}",
            )
        )
    if ps.gripper != es.gripper:
        issues.append(
            CompatIssue(
                "warning",
                "gripper",
                f"policy gripper {ps.gripper!r} != embodiment {es.gripper!r}",
            )
        )
    if ps.frame != es.frame:
        issues.append(
            CompatIssue(
                "warning",
                "frame",
                f"policy frame {ps.frame!r} != embodiment {es.frame!r}",
            )
        )


def _resolve_keys(
    required: frozenset[str],
    provided: frozenset[str],
    remap: dict[str, str],
    kind: str,
    issues: list[CompatIssue],
) -> None:
    for key in sorted(required):
        target = remap.get(key, key)
        if target not in provided:
            issues.append(
                CompatIssue(
                    "error",
                    f"missing_{kind}",
                    f"policy requires {kind} {key!r} (→ {target!r}) which the "
                    f"embodiment does not provide; provides {sorted(provided)}",
                )
            )


def check_compatibility(
    policy: Policy,
    embodiment: Embodiment,
    task: Task | None = None,
    *,
    remap: dict[str, str] | None = None,
) -> CompatibilityReport:
    """Return a structured compatibility report (does not raise)."""
    remap = dict(remap or {})
    report = CompatibilityReport(remap=remap)
    issues = report.issues

    _check_action_spaces(policy.info.action_space, embodiment.info.action_space, issues)

    pobs = policy.info.observation_space
    eobs = embodiment.info.observation_space
    _resolve_keys(pobs.camera_names, eobs.camera_names, remap, "camera", issues)
    _resolve_keys(pobs.state_keys, eobs.state_keys, remap, "state", issues)

    # Control-rate reconciliation (R1): only warn, since the framework paces.
    p_hz = getattr(policy.info, "control_hz", None)
    e_hz = embodiment.info.control_hz
    if p_hz is not None and e_hz is not None and abs(p_hz - e_hz) > _RATE_TOL:
        issues.append(
            CompatIssue(
                "warning",
                "control_rate",
                f"policy desires {p_hz} Hz but embodiment runs at {e_hz} Hz; "
                "framework will pace to the effective rate",
            )
        )

    if task is not None:
        _check_scenes_realizable(task, embodiment, issues)

    return report


def _check_scenes_realizable(task: Task, embodiment: Embodiment, issues: list[CompatIssue]) -> None:
    sup_setups = embodiment.info.supported_setups
    sup_targets = embodiment.info.supported_target_kinds
    for scene in task.scenes:
        if scene.setup is not None and sup_setups and scene.setup not in sup_setups:
            issues.append(
                CompatIssue(
                    "error",
                    "scene_setup",
                    f"scene {scene.id!r} needs setup {scene.setup!r} which "
                    f"embodiment {embodiment.info.name!r} does not support",
                )
            )
        if scene.target is not None and sup_targets and scene.target.kind not in sup_targets:
            issues.append(
                CompatIssue(
                    "error",
                    "scene_target",
                    f"scene {scene.id!r} target kind {scene.target.kind!r} not "
                    f"supported by embodiment {embodiment.info.name!r}",
                )
            )


def assert_compatible(
    policy: Policy,
    embodiment: Embodiment,
    task: Task | None = None,
    *,
    remap: dict[str, str] | None = None,
) -> CompatibilityReport:
    """Check compatibility and raise
    [`CompatibilityError`][roboinspect.errors.CompatibilityError] on hard errors."""
    report = check_compatibility(policy, embodiment, task, remap=remap)
    report.raise_for_errors()
    return report
