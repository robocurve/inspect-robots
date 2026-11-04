"""Adapter conformance: mechanical checks on an embodiment's declared spaces.

Plan 0008 made two consumers of *declarations* load-bearing: the CLI's
default guardrails derive their limits from the action space, and the LLM
agent policy builds its whole tool surface from the spaces at bind time. An
adapter with missing semantics, missing bounds, unlabeled dims, or a
misaligned ``StateSpec`` silently degrades both. This module turns those
requirements into a checkable report so adapter repos can enforce them in CI
(one test: ``assert_embodiment_conformant(MyEmbodiment().info)``) and users
can audit an installed adapter via ``inspect-robots doctor``.

``check_embodiment`` is purely declarative and touches no hardware. The
separate, opt-in ``check_guardrail_contribution`` executes plugin code to
validate the optional runtime contribution hook. Conformance proves an
adapter is *guardrail-ready and agent-ready*, not that its declarations are
honest (a delta rig declaring absolute-sized per-step bounds type-checks
fine). The adapter authoring guide covers the human half.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from inspect_robots.embodiment import Embodiment, EmbodimentInfo
from inspect_robots.guard import GuardContribution
from inspect_robots.spaces import ABSOLUTE_CONTROL_MODES, Box, reference_state_fields

# A slot value with a URL scheme (http://, rtsp://, ...) names a stream, not a device node.
_STREAM_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")

DEVICE_KINDS = ("v4l2", "can", "serial")


@dataclass(frozen=True)
class DeviceSlot:
    """One device-shaped constructor argument the setup wizard interviews.

    ``arg`` is the ``[embodiment.args]`` key to write; ``kind`` selects the
    probe (``"v4l2"``: /dev/v4l listings with unplug-identify; ``"can"``:
    SocketCAN netdevs from sysfs with unplug-identify; ``"serial"``:
    /dev/serial/by-id listing). ``label`` is the human prompt ("left arm CAN
    channel"). Slots sharing a non-None ``group`` are all-or-none: the
    wizard refuses to write a partial subset of the group.
    """

    arg: str
    kind: str
    label: str
    group: str | None = None


def device_slots(factory: object) -> tuple[DeviceSlot, ...]:
    """The declared device slots, defensively read.

    Reads ``DEVICE_SLOTS`` off ``factory``; anything that is not an iterable
    of ``DeviceSlot`` instances (or contains a slot whose ``kind`` is not
    recognized) has the offending entries ignored, never crashes the wizard.
    Returns a tuple in declaration order.
    """
    try:
        slots = getattr(factory, "DEVICE_SLOTS", None)
    except Exception:
        return ()
    if not isinstance(slots, Iterable):
        return ()
    try:
        return tuple(
            slot for slot in slots if isinstance(slot, DeviceSlot) and slot.kind in DEVICE_KINDS
        )
    except Exception:
        return ()


@dataclass(frozen=True)
class OptionSlot:
    """One boolean behavior toggle the setup wizard interviews.

    ``arg`` is the ``[embodiment.args]`` key to write (``true``/``false``);
    ``label`` is the yes/no question shown to the operator ("Skip the
    operator start prompts (auto_start)"); ``default`` is the suggested
    answer when the key is absent from an existing config.
    """

    arg: str
    label: str
    default: bool = False


def option_slots(factory: object) -> tuple[OptionSlot, ...]:
    """The declared option slots, defensively read.

    Reads ``OPTION_SLOTS`` off ``factory``; anything that is not an iterable
    of ``OptionSlot`` instances has the offending entries ignored, never
    crashes the wizard. Returns a tuple in declaration order.
    """
    try:
        slots = getattr(factory, "OPTION_SLOTS", None)
    except Exception:
        return ()
    if not isinstance(slots, Iterable):
        return ()
    try:
        return tuple(
            slot
            for slot in slots
            if isinstance(slot, OptionSlot) and isinstance(slot.default, bool)
        )
    except Exception:
        return ()


@dataclass(frozen=True)
class NumberSlot:
    """One numeric constructor argument the setup wizard interviews.

    ``arg`` is the ``[embodiment.args]`` key to write and ``label`` is the
    prompt shown to the operator, including any unit. ``default`` is the
    suggestion when no valid carried value exists. ``minimum`` and
    ``maximum`` are inclusive bounds; ``None`` leaves that side unbounded.
    ``allow_none`` accepts a disabled value written as ``none``.
    """

    arg: str
    label: str
    default: float | int | None = None
    minimum: float | None = None
    maximum: float | None = None
    allow_none: bool = False


def _finite_number(value: object) -> bool:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _valid_number_slot(slot: NumberSlot) -> bool:
    if slot.minimum is not None and not _finite_number(slot.minimum):
        return False
    if slot.maximum is not None and not _finite_number(slot.maximum):
        return False
    if slot.minimum is not None and slot.maximum is not None and slot.minimum > slot.maximum:
        return False
    if slot.default is None:
        return slot.allow_none
    if not _finite_number(slot.default):
        return False
    if slot.minimum is not None and slot.default < slot.minimum:
        return False
    return slot.maximum is None or slot.default <= slot.maximum


def number_slots(factory: object) -> tuple[NumberSlot, ...]:
    """The declared numeric slots, defensively read.

    Reads ``NUMBER_SLOTS`` off ``factory`` and returns valid ``NumberSlot``
    instances in declaration order. Non-iterables, malformed entries,
    non-finite or boolean numeric fields, inverted bounds, and defaults that
    cannot pass their slot constraints are ignored without crashing setup.
    """
    try:
        slots = getattr(factory, "NUMBER_SLOTS", None)
    except Exception:
        return ()
    if not isinstance(slots, Iterable):
        return ()
    try:
        return tuple(
            slot for slot in slots if isinstance(slot, NumberSlot) and _valid_number_slot(slot)
        )
    except Exception:
        return ()


def missing_runtime_requirements(factory: object) -> dict[str, str]:
    """The declared runtime modules that are not importable here.

    Reads ``RUNTIME_REQUIREMENTS`` (module name -> remediation command) off
    ``factory`` and probes each with ``importlib.util.find_spec``. Top-level
    names (the intended use) are probed without executing anything; a dotted
    name imports its parent package when present, so declare top-level names.
    ANY probe failure counts as missing (broad ``except Exception``: a
    present-but-broken parent package propagates arbitrary errors from its
    ``__init__``, and this checker must never crash setup or doctor).
    Entries whose key or value is not ``str``, or a ``RUNTIME_REQUIREMENTS``
    that is not a ``Mapping``, are ignored (a plugin typo must not crash the
    preflight). Returns the missing subset, insertion-ordered.
    """
    requirements = getattr(factory, "RUNTIME_REQUIREMENTS", None)
    if not isinstance(requirements, Mapping):
        return {}

    missing: dict[str, str] = {}
    for name, remedy in requirements.items():
        if not isinstance(name, str) or not isinstance(remedy, str):
            continue
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None:
            missing[name] = remedy
    return missing


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


def check_device_slots(
    factory: object,
    configured: Mapping[str, object],
    *,
    sysfs_net: Path | None = None,
) -> list[ConformanceIssue]:
    """Check for configured device-slot values that no longer resolve on this host.

    For each ``DEVICE_SLOTS`` slot with a string value in ``configured``, return an
    error finding when a camera or serial path does not exist or cannot be checked, or a CAN
    interface is not present. A non-CAN value with a URL scheme (a camera stream such as
    ``http://host/api/stream.mjpeg?src=cam``) is skipped: there is no path to check, and the
    embodiment's own health check probes the stream live.

    ``configured`` is the resolved ``[embodiment.args]``. A non-string or absent value is skipped by
    this check. CAN validation checks presence only.

    Returns findings in slot order.
    """
    # Lazy import because _setup imports conformance at module load.
    from inspect_robots._setup import SYSFS_NET, _scan_can

    net = SYSFS_NET if sysfs_net is None else sysfs_net
    issues: list[ConformanceIssue] = []
    for slot in device_slots(factory):
        value = configured.get(slot.arg)
        if not isinstance(value, str):
            continue
        if slot.kind == "can":
            interfaces = _scan_can(net)
            if value not in interfaces:
                present = ", ".join(interfaces) or "none"
                issues.append(
                    ConformanceIssue(
                        "error",
                        "device",
                        f"{slot.label} ({slot.arg}): CAN interface {value!r} not found; "
                        f"present: {present}",
                    )
                )
        elif _STREAM_URL.match(value):
            # A camera slot may name a stream (e.g. an MJPEG URL from a camera
            # hub) instead of a device node; it has no path to stat, and the
            # embodiment's live health check is what probes it.
            continue
        else:
            try:
                exists = Path(value).exists()
            except OSError as exc:
                issues.append(
                    ConformanceIssue(
                        "error",
                        "device",
                        f"{slot.label} ({slot.arg}): {slot.kind} path {value!r} could not be "
                        f"checked: {exc.strerror or exc}",
                    )
                )
                continue
            if not exists:
                issues.append(
                    ConformanceIssue(
                        "error",
                        "device",
                        f"{slot.label} ({slot.arg}): {slot.kind} path {value!r} does not exist",
                    )
                )
    return issues


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

        if semantics.control_mode in ABSOLUTE_CONTROL_MODES:
            matching = reference_state_fields(
                info.observation_space.state, space.dim, semantics.control_mode
            )
            if len(matching) != 1:
                error(
                    "state_alignment",
                    "absolute-target control needs exactly one StateSpec field with "
                    f"shape ({space.dim},) as the proprioceptive reference; found "
                    f"{matching or 'none'}",
                )

        if low is not None and high is not None:
            from inspect_robots.guard import DeltaLimitGuard

            try:
                DeltaLimitGuard(space)
            except ValueError as exc:
                # Same refusal the CLI degradation path reports; here it is an
                # error because a conformant adapter must be limitable.
                error("guardrails", str(exc))

    if info.control_hz is None:
        warning(
            "control_hz",
            "control_hz is undeclared; agent motion durations fall back to 10 Hz step counting",
        )
    elif (
        isinstance(info.control_hz, bool)
        or not isinstance(info.control_hz, (int, float))
        or not math.isfinite(info.control_hz)
        or info.control_hz <= 0
    ):
        error(
            "control_hz",
            f"control_hz must be a positive finite number, got {info.control_hz!r}",
        )

    return ConformanceReport(embodiment=info.name, issues=tuple(issues))


def assert_embodiment_conformant(info: EmbodimentInfo) -> None:
    """Pytest-friendly wrapper: raise ``AssertionError`` with the full summary."""
    report = check_embodiment(info)
    if not report.ok:
        raise AssertionError(report.summary())


def _unmigrated_guard_hook(embodiment: object) -> bool:
    """True when only the pre-paper ``contribute_guardrails`` hook supplies safety guards.

    That is the old hook with no ``contribute_guards``, or with only the no-op
    ``EmbodimentBase.contribute_guards`` inherited rather than overridden.
    """
    from inspect_robots.embodiment import EmbodimentBase

    if not hasattr(embodiment, "contribute_guardrails"):
        return False
    method = getattr(type(embodiment), "contribute_guards", None)
    return method is None or method is EmbodimentBase.contribute_guards


def check_guardrail_contribution(embodiment: Embodiment, action_space: Box) -> ConformanceReport:
    """Validate an optional contribution hook by executing plugin code.

    An absent attribute passes. A present attribute must be callable and must
    return ``GuardContribution``. Exceptions raised by the hook propagate
    as plugin bugs. Unlike ``check_embodiment``, callers must opt in knowing
    this check can run arbitrary adapter code.
    """
    if _unmigrated_guard_hook(embodiment):
        issue = ConformanceIssue(
            "error",
            "guardrail_contribution",
            "defines contribute_guardrails(), which was renamed contribute_guards() and "
            "returns GuardContribution; the run command refuses this adapter until it is migrated",
        )
        return ConformanceReport(embodiment=embodiment.info.name, issues=(issue,))
    missing = object()
    hook = getattr(embodiment, "contribute_guards", missing)
    if hook is missing:
        return ConformanceReport(embodiment=embodiment.info.name)
    if not callable(hook):
        issue = ConformanceIssue(
            "error",
            "guardrail_contribution",
            f"contribute_guards is not callable (got {type(hook).__name__})",
        )
        return ConformanceReport(embodiment=embodiment.info.name, issues=(issue,))
    contribution = hook(action_space)
    if not isinstance(contribution, GuardContribution):
        issue = ConformanceIssue(
            "error",
            "guardrail_contribution",
            f"contribute_guards returned {type(contribution).__name__}, expected GuardContribution",
        )
        return ConformanceReport(embodiment=embodiment.info.name, issues=(issue,))
    return ConformanceReport(embodiment=embodiment.info.name)


def assert_guardrail_contribution_conformant(embodiment: Embodiment, action_space: Box) -> None:
    """Raise ``AssertionError`` with the full contribution report on failure."""
    report = check_guardrail_contribution(embodiment, action_space)
    if not report.ok:
        raise AssertionError(report.summary())


def _policy_interface(info: EmbodimentInfo) -> tuple[object, ...]:
    """What a policy sees of an embodiment: every declared field of both its spaces.

    Built from the spaces' dataclass fields rather than a hand-picked list, so
    any declared difference (bounds, rotation or gripper encoding, camera
    channels, state units or dtypes, a field added later) makes two
    interfaces distinct.
    """
    return (_frozen(info.action_space), _frozen(info.observation_space))


def _frozen(value: object) -> object:
    """A hashable, comparable rendering of a space description."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            tuple(
                (spec.name, _frozen(getattr(value, spec.name)))
                for spec in dataclasses.fields(value)
            ),
        )
    if isinstance(value, np.ndarray):
        # Bytes, not values: NaN bounds compare equal to themselves this way.
        return ("ndarray", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def check_policy_interfaces(infos: Iterable[EmbodimentInfo]) -> ConformanceReport:
    """Check that embodiments sharing a morphology offer distinct policy interfaces.

    The paper's unit is one (morphology, policy interface) pair: two
    embodiments on the same platform that a policy cannot tell apart are one
    embodiment registered twice, and their results would be pooled under two
    names. Embodiments without a ``morphology`` are not compared. Errors name
    each clashing pair. Purely declarative.
    """
    issues: list[ConformanceIssue] = []
    seen: dict[tuple[str, tuple[object, ...]], str] = {}
    names: list[str] = []
    for info in infos:
        names.append(info.name)
        if info.morphology is None:
            continue
        key = (info.morphology, _policy_interface(info))
        earlier = seen.setdefault(key, info.name)
        if earlier != info.name:
            issues.append(
                ConformanceIssue(
                    "error",
                    "policy_interface",
                    f"{earlier!r} and {info.name!r} share morphology {info.morphology!r} "
                    "and the same policy interface; give them different action or "
                    "observation spaces, or register one embodiment",
                )
            )
    return ConformanceReport(embodiment=", ".join(names), issues=tuple(issues))


_EVALUATION_HOOK_ARITY = {"health": 0, "prepare_trial": 1, "home": 0, "park": 0}


def check_evaluation_interface(embodiment: Embodiment) -> ConformanceReport:
    """Check the optional evaluation-interface hooks without calling them.

    Absent hooks pass. A present ``health``, ``home`` or ``park`` must be
    callable with no arguments, and ``prepare_trial`` with one (the scene).
    Signatures that cannot be inspected pass. A warning notes ``park``
    without ``health``: an embodiment that can move itself to rest can
    usually also say whether it is fit to run.
    """
    issues: list[ConformanceIssue] = []
    missing = object()
    offered: set[str] = set()
    for name, arity in _EVALUATION_HOOK_ARITY.items():
        hook = getattr(embodiment, name, missing)
        if hook is missing:
            continue
        if not callable(hook):
            issues.append(
                ConformanceIssue(
                    "error",
                    "evaluation_interface",
                    f"{name} is not callable (got {type(hook).__name__})",
                )
            )
            continue
        offered.add(name)
        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            continue
        try:
            signature.bind(*([None] * arity))
        except TypeError:
            expected = "the scene" if arity else "no arguments"
            issues.append(
                ConformanceIssue(
                    "error",
                    "evaluation_interface",
                    f"{name}({', '.join(signature.parameters)}) cannot be called with {expected}",
                )
            )
    if "park" in offered and "health" not in offered:
        issues.append(
            ConformanceIssue(
                "warning",
                "evaluation_interface",
                "park() is offered without health(); eval() cannot refuse to start "
                "a trial on an unhealthy platform",
            )
        )
    return ConformanceReport(embodiment=embodiment.info.name, issues=tuple(issues))
