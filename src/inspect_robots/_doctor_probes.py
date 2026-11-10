"""Read-only host probes for doctor; no component construction or device opens."""

from __future__ import annotations

import inspect
import shlex
import socket
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from inspect_robots.conformance import (
    ConformanceIssue,
    DeviceSlot,
    check_device_slots,
)

if TYPE_CHECKING:
    from socket import socket as Socket

# Only documented providers belong here. Never infer a package from a name.
_PROVIDERS = {
    "molmoact2": "inspect-robots-yam",
    "yam_arms": "inspect-robots-yam",
    "agent": "inspect-robots-agent",
    "capx": "inspect-robots-capx",
    "xpolicylab": "inspect-robots-xpolicylab",
    "ros": "inspect-robots-ros",
    "isaacsim": "inspect-robots-isaacsim",
}

# Plugin releases known to lack features the framework now relies on: the last
# lagging version and what it lacks. Requirement metadata cannot express these.
_KNOWN_LAGGING = {
    "inspect-robots-yam": ("0.6.0", "required pose parsing and action dim_labels"),
}


def _install_hint(name: str) -> str:
    """Suggest an install only for a known provider."""
    package = _PROVIDERS.get(name)
    return f"; install: uv pip install {package}" if package else ""


def _viewer_headless(env: Mapping[str, str], platform: str) -> bool:
    """Match setup's local-viewer availability, including forwarded displays."""
    return not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")) and (
        platform not in {"darwin", "win32"}
        or any(env.get(key) for key in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))
    )


def _display_probe(enabled: bool, env: Mapping[str, str], platform: str) -> list[ConformanceIssue]:
    if enabled and _viewer_headless(env, platform):
        command = "inspect-robots config set rerun false"
        if env.get("INSPECT_ROBOTS_CONFIG"):
            command += " --config " + shlex.quote(env["INSPECT_ROBOTS_CONFIG"])
        return [
            ConformanceIssue(
                "warning",
                "display",
                "Rerun viewer enabled without a display; run: "
                f"{command}, "
                "or pass --no-rerun to run; recordings remain available",
            )
        ]
    return []


def _version_probe(
    distributions: Iterable[metadata.Distribution] | None = None,
) -> list[ConformanceIssue]:
    """Check installed plugin constraints and known missing features from metadata."""
    installed = list(metadata.distributions() if distributions is None else distributions)
    plugins = [
        dist
        for dist in installed
        if any(ep.group.startswith("inspect_robots.") for ep in dist.entry_points)
    ]
    if not plugins:
        return []
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return [
            ConformanceIssue(
                "info", "version", "install version checks: uv pip install 'inspect-robots[doctor]'"
            )
        ]
    by_name: dict[str, metadata.Distribution] = {
        canonicalize_name(dist.metadata["Name"]): dist for dist in installed
    }
    issues: list[ConformanceIssue] = []
    for dist in plugins:
        name = dist.metadata["Name"]
        try:
            version = Version(dist.version)
            lagging = _KNOWN_LAGGING.get(canonicalize_name(name))
            if lagging is not None and version <= Version(lagging[0]):
                issues.append(
                    ConformanceIssue(
                        "error",
                        "version",
                        f"{name} {version} lacks {lagging[1]}; "
                        f"run: uv pip install --upgrade {shlex.quote(name)}; "
                        "if no newer release is available, a plugin release is needed",
                    )
                )
            # Check plugin -> framework and framework -> installed plugin constraints.
            framework = by_name.get("inspect-robots")
            pairs = [(dist, "inspect-robots")]
            if framework is not None:
                pairs.append((framework, canonicalize_name(name)))
            for owner, target in pairs:
                for raw in owner.requires or ():
                    requirement = Requirement(raw)
                    if canonicalize_name(requirement.name) != target:
                        continue
                    if requirement.marker is not None and not requirement.marker.evaluate():
                        continue
                    dependency = by_name.get(target)
                    if dependency is None:
                        issues.append(
                            ConformanceIssue(
                                "error",
                                "version",
                                f"{name} requires {requirement}; run: "
                                f"uv pip install --upgrade {shlex.quote(str(requirement))}",
                            )
                        )
                    elif not requirement.specifier.contains(dependency.version, prereleases=True):
                        issues.append(
                            ConformanceIssue(
                                "error",
                                "version",
                                f"{name} {version}: {owner.metadata['Name']} "
                                f"requires {requirement}, "
                                f"installed {dependency.metadata['Name']} {dependency.version}; "
                                f"run: uv pip install --upgrade {shlex.quote(name)} "
                                f"{shlex.quote(str(requirement))}",
                            )
                        )
        except (InvalidVersion, InvalidRequirement) as exc:
            issues.append(
                ConformanceIssue(
                    "warning",
                    "version",
                    f"cannot check metadata for {name}: {exc}; "
                    f"run: uv pip install --upgrade {shlex.quote(name)}",
                )
            )
    return issues


def _can_inventory(net: Path) -> dict[str, bool | None]:
    """Read SocketCAN names and IFF_UP; UNKNOWN operstate can still mean UP."""
    from inspect_robots._setup import _scan_can

    # Unlike setup's best-effort inventory, distinguish unreadable sysfs from no adapters.
    list(net.iterdir())
    states: dict[str, bool | None] = {}
    for name in _scan_can(net):
        try:
            states[name] = bool(int((net / name / "flags").read_text().strip(), 16) & 1)
        except (OSError, UnicodeError, ValueError):
            states[name] = None
    return states


def _can_probe(slot: DeviceSlot, value: str, *, net: Path, platform: str) -> list[ConformanceIssue]:
    """Check CAN presence and administrative state without opening the interface."""
    prefix = f"{slot.label} ({slot.arg}): CAN interface {value!r}"
    if not platform.startswith("linux"):
        return [ConformanceIssue("info", "device", f"{prefix}: check on Linux: ip -br link")]
    try:
        interfaces = _can_inventory(net)
    except OSError:
        return [
            ConformanceIssue(
                "warning", "device", f"{prefix}: cannot read CAN inventory; run: ip -br link"
            )
        ]
    if value not in interfaces:
        present = ", ".join(interfaces) or "none"
        guidance = (
            "check adapter power/cabling; if renamed, select the present name"
            if interfaces
            else "no CAN adapters found; check USB adapter power/cabling with lsusb"
        )
        return [
            ConformanceIssue(
                "error",
                "device",
                f"{prefix} not found; present: {present}; {guidance}; run: inspect-robots setup",
            )
        ]
    state = interfaces[value]
    if state is None:
        return [
            ConformanceIssue(
                "warning", "device", f"{prefix}: UP state unreadable; run: ip -br link"
            )
        ]
    if not state:
        return [
            ConformanceIssue(
                "error",
                "device",
                f"{prefix} is DOWN; after configuring the adapter bitrate, "
                f"run: sudo ip link set dev {shlex.quote(value)} up",
            )
        ]
    return []


def _device_probe(
    factory: object,
    configured: Mapping[str, object],
    *,
    net: Path = Path("/sys/class/net"),
    platform: str = sys.platform,
) -> list[ConformanceIssue]:
    """Extend device-slot validation with CAN state and actionable path fixes."""

    can_issues: list[ConformanceIssue] = []

    def can_check(slot: DeviceSlot, value: str) -> list[ConformanceIssue]:
        found = _can_probe(slot, value, net=net, platform=platform)
        can_issues.extend(found)
        return found

    issues = check_device_slots(factory, configured, can_check=can_check)
    # Every finding the CAN probe did not produce is a camera or serial path finding.
    return [
        issue
        if any(issue is can_issue for can_issue in can_issues)
        else replace(
            issue, message=issue.message + "; reconnect the device or run: inspect-robots setup"
        )
        for issue in issues
    ]


def _policy_server_probe(
    factory: object,
    configured: Mapping[str, object],
    *,
    connect: Callable[..., Socket] | None = None,
) -> list[ConformanceIssue]:
    """Probe URL arguments/defaults via TCP only, without constructing the policy."""
    parameters: Mapping[str, inspect.Parameter]
    try:
        parameters = inspect.signature(cast(Callable[..., object], factory)).parameters
    except (TypeError, ValueError):
        parameters = {}
    values = {key: parameter.default for key, parameter in parameters.items()}
    values.update(configured)
    issues: list[ConformanceIssue] = []
    connector = socket.create_connection if connect is None else connect
    for key, value in values.items():
        if key not in {"url", "server_url", "base_url"} and not key.endswith("_url"):
            continue
        if not isinstance(value, str) or not value:
            continue
        try:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https", "ws", "wss", "tcp"} or not parsed.hostname:
                raise ValueError("expected http(s), ws(s) or tcp URL with host")
            host = parsed.hostname
            port = parsed.port
            if port is None:
                if parsed.scheme == "tcp":
                    raise ValueError("tcp URL needs a port")
                port = 443 if parsed.scheme in {"https", "wss"} else 80
            # Never print credentials, path or query strings from a policy URL.
            endpoint = f"{host}:{port}"
            with connector((host, port), timeout=1.0):
                pass
        except ValueError as exc:
            issues.append(
                ConformanceIssue(
                    "error",
                    "policy-server",
                    f"{key}: invalid server URL ({exc}); "
                    "correct [policy.args] or run: inspect-robots setup",
                )
            )
        except OSError:
            issues.append(
                ConformanceIssue(
                    "error",
                    "policy-server",
                    f"{key}: cannot reach {endpoint} (1s TCP timeout); "
                    "start the policy server and verify its host/port; run: "
                    f"nc -vz -w 1 {shlex.quote(host)} {port}",
                )
            )
    return issues
