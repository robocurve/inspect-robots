"""Hardware-free regressions for the host preflight in doctor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from inspect_robots import cli, registry
from inspect_robots.conformance import DeviceSlot
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy


def test_doctor_aggregates_host_failures_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every cheap probe runs even when construction cannot succeed."""
    config = tmp_path / "rig.ini"
    config.write_text(
        "[defaults]\npolicy = preflight-policy\nembodiment = preflight-rig\nrerun = true\n"
        "[policy.args]\nserver_url = http://localhost:8202/act\n"
        f"[embodiment.args]\nwrist = {tmp_path / 'missing-camera'}\nleft = can_wrong\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(config))
    monkeypatch.delenv("INSPECT_ROBOTS_POLICY", raising=False)
    monkeypatch.delenv("INSPECT_ROBOTS_EMBODIMENT", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    class _Policy(ScriptedPolicy):
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
            "missing_preflight_transport": "uv pip install transport"
        }

        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("doctor must not construct policies")

    class _Rig(CubePickEmbodiment):
        DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = (
            DeviceSlot("wrist", "v4l2", "wrist camera"),
            DeviceSlot("left", "can", "left CAN"),
        )

        def __init__(self, **kwargs: Any) -> None:
            raise ValueError("bad pose")

    monkeypatch.setitem(registry._FACTORIES["policy"], "preflight-policy", _Policy)
    monkeypatch.setitem(registry._FACTORIES["embodiment"], "preflight-rig", _Rig)
    # Patch host inventory and the low-level connector, never use a real rig.
    from inspect_robots import _doctor_probes

    monkeypatch.setattr(_doctor_probes, "_can_inventory", lambda net: {"can_left": True})
    distribution = _dist(tmp_path, "inspect-robots-yam", "0.6.0")
    version_probe = _doctor_probes._version_probe
    monkeypatch.setattr(_doctor_probes, "_version_probe", lambda: version_probe([distribution]))
    import socket

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("offline")

    monkeypatch.setattr(socket, "create_connection", refused)
    with pytest.raises(ValueError, match="bad pose"):
        cli.main(["doctor"])
    out = capsys.readouterr().out
    for expected in (
        "runtime-requirement:",
        "[error] version:",
        "uv pip install --upgrade inspect-robots-yam",
        "[warning] display:",
        "inspect-robots config set rerun false",
        "--no-rerun",
        "policy-server:",
        "wrist camera (wrist): v4l2 path",
        "CAN interface 'can_wrong' not found",
        "inspect-robots setup",
    ):
        assert expected in out


@pytest.mark.parametrize(
    ("enabled", "env", "platform", "warns"),
    [
        (False, {}, "linux", False),
        (True, {}, "linux", True),
        (True, {"DISPLAY": ":0"}, "linux", False),
        (True, {"WAYLAND_DISPLAY": "wayland-0"}, "linux", False),
        (True, {}, "darwin", False),
        (True, {}, "win32", False),
        (True, {"SSH_CONNECTION": "remote"}, "darwin", True),
        (True, {"SSH_CLIENT": "remote"}, "win32", True),
        (True, {"SSH_TTY": "tty"}, "darwin", True),
        (True, {"DISPLAY": ":10", "SSH_CONNECTION": "remote"}, "linux", False),
    ],
)
def test_display_probe(enabled: bool, env: dict[str, str], platform: str, warns: bool) -> None:
    """Native desktops and forwarded displays follow setup's existing rules."""
    from inspect_robots._doctor_probes import _display_probe

    findings = _display_probe(enabled, env, platform)
    assert bool(findings) is warns
    if findings:
        assert findings[0].severity == "warning"
        assert "config set rerun false" in findings[0].message


def _dist(root: Path, name: str, version: str, *requires: str, plugin: bool = True) -> Any:
    from importlib.metadata import PathDistribution

    directory = root / name
    directory.mkdir()
    directory.joinpath("METADATA").write_text(
        f"Name: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires),
        encoding="utf-8",
    )
    if plugin:
        directory.joinpath("entry_points.txt").write_text(
            "[inspect_robots.policies]\nfake = fake:Policy\n", encoding="utf-8"
        )
    return PathDistribution(directory)


def test_version_constraints_and_known_required_features(tmp_path: Path) -> None:
    """Metadata bounds and the field report's known-bad feature release both fail."""
    from inspect_robots._doctor_probes import _version_probe

    core = _dist(tmp_path, "inspect-robots", "1.0", "example-plugin>=2", plugin=False)
    yam = _dist(tmp_path, "inspect-robots-yam", "0.6.0", "inspect-robots>=0.7")
    other = _dist(
        tmp_path,
        "example-plugin",
        "1.5rc1",
        "inspect-robots>=2",
        'inspect-robots>=9; python_version < "2"',
        'inspect-robots>=9; extra == "unused"',
        "numpy>=1",
    )
    issues = _version_probe([core, yam, other])
    assert len(issues) == 3
    assert "pose parsing and action dim_labels" in issues[0].message
    assert "uv pip install --upgrade inspect-robots-yam" in issues[0].message
    assert "inspect-robots>=2" in issues[1].message
    assert "example-plugin>=2" in issues[2].message
    assert all(issue.severity == "error" for issue in issues)


@pytest.mark.parametrize(
    ("version", "requires", "count"),
    [("0.6.1", (), 0), ("bad-version", (), 1), ("1.0", ("broken => 1",), 1)],
)
def test_version_metadata_failures(
    tmp_path: Path, version: str, requires: tuple[str, ...], count: int
) -> None:
    """Malformed metadata produces a finding rather than aborting later probes."""
    from inspect_robots._doctor_probes import _version_probe

    issues = _version_probe([_dist(tmp_path, "inspect-robots-yam", version, *requires)])
    assert len(issues) == count
    if issues:
        assert issues[0].severity == "warning"
        assert "uv pip install --upgrade" in issues[0].message


def test_version_missing_framework_and_unconstrained_plugin(tmp_path: Path) -> None:
    """A missing dependency has a concrete install remedy; no metadata is valid."""
    from inspect_robots._doctor_probes import _version_probe

    plugin = _dist(tmp_path, "plugin", "1.0", "inspect-robots>=1")
    assert "uv pip install --upgrade 'inspect-robots>=1'" in _version_probe([plugin])[0].message
    assert _version_probe([_dist(tmp_path, "other", "0.1")]) == []
    assert _version_probe([]) == []


def test_version_optional_parser_and_default_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core-only environments receive the optional install command without importing it."""
    import builtins
    from importlib import metadata

    from inspect_robots._doctor_probes import _version_probe

    dist = _dist(tmp_path, "plugin", "1")
    monkeypatch.setattr(metadata, "distributions", lambda: [dist])
    original = builtins.__import__

    def without_packaging(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("packaging"):
            raise ImportError("optional parser absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_packaging)
    assert "uv pip install 'inspect-robots[doctor]'" in _version_probe()[0].message


def _net(root: Path, name: str, flags: str | None, kind: str = "280") -> None:
    directory = root / name
    directory.mkdir(parents=True)
    directory.joinpath("type").write_text(kind, encoding="utf-8")
    if flags is not None:
        directory.joinpath("flags").write_text(flags, encoding="utf-8")


def test_can_inventory_and_state_findings(tmp_path: Path) -> None:
    """CAN flags distinguish administratively UP, DOWN and unreadable adapters."""
    from inspect_robots._doctor_probes import _can_inventory, _can_probe

    _net(tmp_path, "can_left", "0x41")
    _net(tmp_path, "can_right", "0x80")
    _net(tmp_path, "can_bad", "bad hex")
    _net(tmp_path, "can_missing_flags", None)
    _net(tmp_path, "eth0", "0x1", "1")
    slot = DeviceSlot("left", "can", "left CAN")
    assert _can_inventory(tmp_path) == {
        "can_bad": None,
        "can_left": True,
        "can_missing_flags": None,
        "can_right": False,
    }
    assert _can_probe(slot, "can_left", net=tmp_path, platform="linux") == []
    down = _can_probe(slot, "can_right", net=tmp_path, platform="linux")[0]
    assert down.severity == "error"
    assert "sudo ip link set dev can_right up" in down.message
    unknown = _can_probe(slot, "can_bad", net=tmp_path, platform="linux")[0]
    assert unknown.severity == "warning"
    assert "ip -br link" in unknown.message
    absent = _can_probe(slot, "can0", net=tmp_path, platform="linux")[0]
    assert "present: can_bad, can_left, can_missing_flags, can_right" in absent.message
    assert "if renamed" in absent.message
    assert "inspect-robots setup" in absent.message


def test_can_no_adapters_unreadable_and_nonlinux(tmp_path: Path) -> None:
    """An absent adapter, inaccessible inventory and unsupported OS get distinct guidance."""
    from inspect_robots._doctor_probes import _can_probe

    slot = DeviceSlot("left", "can", "left CAN")
    empty = _can_probe(slot, "can0", net=tmp_path, platform="linux")[0]
    assert "present: none" in empty.message
    assert "lsusb" in empty.message
    inaccessible = _can_probe(slot, "can0", net=tmp_path / "absent", platform="linux")[0]
    assert inaccessible.severity == "warning"
    assert "cannot read CAN inventory" in inaccessible.message
    skipped = _can_probe(slot, "can0", net=tmp_path, platform="darwin")[0]
    assert skipped.severity == "info"
    assert "check on Linux" in skipped.message


def test_device_probe_preserves_path_checks_and_fixes(tmp_path: Path) -> None:
    """Existing camera/serial validation and URL skipping are retained beside CAN state."""
    from inspect_robots._doctor_probes import _device_probe

    class _Rig:
        DEVICE_SLOTS = (
            DeviceSlot("left", "can", "left CAN"),
            DeviceSlot("camera", "v4l2", "camera"),
            DeviceSlot("stream", "v4l2", "stream"),
        )

    _net(tmp_path, "can0", "0x1")
    configured = {"left": "can0", "camera": str(tmp_path / "gone"), "stream": "http://cam/"}
    issues = _device_probe(_Rig, configured, net=tmp_path, platform="linux")
    assert len(issues) == 1
    assert "v4l2 path" in issues[0].message
    assert "reconnect the device or run: inspect-robots setup" in issues[0].message
    configured["camera"] = str(tmp_path)
    assert _device_probe(_Rig, configured, net=tmp_path, platform="linux") == []
    configured["left"] = "can_wrong"
    assert "CAN interface" in _device_probe(_Rig, configured, net=tmp_path)[0].message


def test_policy_server_defaults_overrides_and_tcp_only() -> None:
    """All configured service URLs use the injected connector and close it immediately."""
    from contextlib import nullcontext

    from inspect_robots._doctor_probes import _policy_server_probe

    calls: list[tuple[tuple[str, int], float]] = []

    def connect(address: tuple[str, int], *, timeout: float) -> Any:
        calls.append((address, timeout))
        return nullcontext()

    def factory(
        server_url: str = "http://localhost:8202/act", disabled_url: str | None = None
    ) -> None:
        raise AssertionError("must not construct")

    assert _policy_server_probe(factory, {}, connect=connect) == []
    assert calls == [(("localhost", 8202), 1.0)]
    calls.clear()
    assert _policy_server_probe(factory, {"server_url": None}, connect=connect) == []
    assert calls == []
    configured = {
        "server_url": "https://user:secret@remote/action?token=secret",
        "url": "ws://[::1]/",
        "base_url": "wss://remote",
        "sam3_url": "tcp://local:1234",
        "not_endpoint": "http://ignored/",
        "empty_url": "",
    }
    assert _policy_server_probe(factory, configured, connect=connect) == []
    assert calls == [
        (("remote", 443), 1.0),
        (("::1", 80), 1.0),
        (("remote", 443), 1.0),
        (("local", 1234), 1.0),
    ]


@pytest.mark.parametrize(
    "url",
    ["invalid", "ftp://host/", "http:///missing", "http://host:nope", "tcp://host", "http://["],
)
def test_policy_server_invalid_urls(url: str) -> None:
    """Malformed URLs fail before the connector is called."""
    from inspect_robots._doctor_probes import _policy_server_probe

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not connect")

    issue = _policy_server_probe(object(), {"server_url": url}, connect=forbidden)[0]
    assert issue.severity == "error"
    assert "invalid server URL" in issue.message
    assert "inspect-robots setup" in issue.message


def test_policy_server_refused_and_timeout_continue_without_secrets() -> None:
    """Every endpoint failure is reported; URL secrets are never echoed."""
    from inspect_robots._doctor_probes import _policy_server_probe

    def offline(address: tuple[str, int], *, timeout: float) -> Any:
        assert timeout == 1
        if address[1] == 80:
            raise ConnectionRefusedError("secret from transport")
        raise TimeoutError("secret from transport")

    issues = _policy_server_probe(
        object(),
        {"url": "http://user:secret@host/path?secret", "server_url": "https://other"},
        connect=offline,
    )
    assert len(issues) == 2
    assert "nc -vz -w 1 host 80" in issues[0].message
    assert "nc -vz -w 1 other 443" in issues[1].message
    assert "secret" not in str(issues)


def test_display_fix_targets_selected_config() -> None:
    """The suggested config command selects the same per-rig file as doctor."""
    from inspect_robots._doctor_probes import _display_probe

    issue = _display_probe(True, {"INSPECT_ROBOTS_CONFIG": "/rig configs/arm.ini"}, "linux")[0]
    assert "config set rerun false --config '/rig configs/arm.ini'" in issue.message


def test_doctor_known_provider_install_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Known missing policy and embodiment names both identify their package."""
    config = tmp_path / "rig.ini"
    config.write_text("[defaults]\npolicy = molmoact2\nembodiment = yam_arms\n", encoding="utf-8")
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(config))
    monkeypatch.delenv("INSPECT_ROBOTS_POLICY", raising=False)
    monkeypatch.delenv("INSPECT_ROBOTS_EMBODIMENT", raising=False)
    monkeypatch.delitem(registry._FACTORIES["policy"], "molmoact2", raising=False)
    monkeypatch.delitem(registry._FACTORIES["embodiment"], "yam_arms", raising=False)
    with pytest.raises(SystemExit, match="no embodiment named 'yam_arms'"):
        cli.main(["doctor"])
    assert capsys.readouterr().out.count("install: uv pip install inspect-robots-yam") == 2


def test_doctor_warnings_and_platform_skips_do_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing viewer display and skipped CAN check allow conformant rigs to pass."""
    config = tmp_path / "rig.ini"
    config.write_text(
        "[defaults]\nembodiment = can-rig\nrerun = true\n[embodiment.args]\nleft = can0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(config))
    monkeypatch.delenv("INSPECT_ROBOTS_EMBODIMENT", raising=False)
    monkeypatch.delenv("INSPECT_ROBOTS_POLICY", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("SSH_TTY", "remote")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    closed: list[bool] = []

    class _Rig(CubePickEmbodiment):
        DEVICE_SLOTS = (DeviceSlot("left", "can", "left CAN"),)

        def __init__(self, left: str) -> None:
            super().__init__()

        def reset(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("never reset in doctor")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setitem(registry._FACTORIES["embodiment"], "can-rig", _Rig)
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[warning] display:" in out
    assert "[info] device:" in out
    assert closed == [True]


def test_doctor_policy_url_args_ownership_and_error_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Args from another policy cannot redirect the probe; a refused port fails doctor."""
    import socket

    config = tmp_path / "rig.ini"
    config.write_text(
        "[defaults]\npolicy = original\nembodiment = cubepick\n"
        "[policy.args]\nserver_url = http://wrong-host:4321/\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(config))
    monkeypatch.setenv("INSPECT_ROBOTS_POLICY", "replacement")
    monkeypatch.delenv("INSPECT_ROBOTS_EMBODIMENT", raising=False)
    calls: list[tuple[str, int]] = []

    def factory(server_url: str = "http://correct-host:1234/") -> None:
        raise AssertionError("never construct policies")

    def refused(address: tuple[str, int], *, timeout: float) -> Any:
        calls.append(address)
        raise ConnectionRefusedError

    monkeypatch.setitem(registry._FACTORIES["policy"], "replacement", factory)
    monkeypatch.setattr(socket, "create_connection", refused)
    assert cli.main(["doctor"]) == 1
    assert calls == [("correct-host", 1234)]
    out = capsys.readouterr().out
    assert "policy-server: server_url: cannot reach correct-host:1234" in out
    assert "cubepick: conformant" in out


def test_doctor_resolved_components_share_preflight_without_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resolved sim/scorer/grader imports join policy-server findings in one pass."""
    import socket

    config = tmp_path / "rig.ini"
    config.write_text(
        "[defaults]\npolicy = served-policy\nembodiment = cubepick\n"
        "sim_embodiment = unused-sim\nscorer = unused-scorer\ngrader = unused-grader\n"
        "[policy.args]\nserver_url = http://localhost:8202/\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(config))
    for variable in (
        "INSPECT_ROBOTS_POLICY",
        "INSPECT_ROBOTS_EMBODIMENT",
        "INSPECT_ROBOTS_SIM_EMBODIMENT",
    ):
        monkeypatch.delenv(variable, raising=False)

    class _UnusedComponent:
        RUNTIME_REQUIREMENTS: ClassVar[dict[str, str]] = {
            "missing_preflight_runtime": "uv pip install component-runtime"
        }

        def __init__(self) -> None:
            raise AssertionError("non-embodiment components must not be constructed")

    def policy(server_url: str) -> None:
        raise AssertionError("policies must not be constructed")

    calls: list[tuple[str, int]] = []

    def refused(address: tuple[str, int], *, timeout: float) -> Any:
        calls.append(address)
        raise ConnectionRefusedError

    for kind, name in (
        ("embodiment", "unused-sim"),
        ("scorer", "unused-scorer"),
        ("grader", "unused-grader"),
    ):
        monkeypatch.setitem(registry._FACTORIES[kind], name, _UnusedComponent)
    monkeypatch.setitem(registry._FACTORIES["policy"], "served-policy", policy)
    monkeypatch.setattr(socket, "create_connection", refused)
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert out.count("missing_preflight_runtime missing → uv pip install component-runtime") == 3
    assert "[error] policy-server:" in out
    assert calls == [("localhost", 8202)]
    assert "cubepick: conformant" in out


def test_doctor_probes_configured_server_when_policy_is_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An absent plugin must not hide an independently configured unreachable server."""
    import socket

    config = tmp_path / "rig.ini"
    config.write_text(
        "[defaults]\npolicy = absent-served-policy\nembodiment = cubepick\n"
        "[policy.args]\nserver_url = http://localhost:8202/\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INSPECT_ROBOTS_CONFIG", str(config))
    monkeypatch.delenv("INSPECT_ROBOTS_POLICY", raising=False)
    monkeypatch.delenv("INSPECT_ROBOTS_EMBODIMENT", raising=False)

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionRefusedError

    monkeypatch.setattr(socket, "create_connection", refused)
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "resolution: no policy named 'absent-served-policy'" in out
    assert "policy-server: server_url: cannot reach localhost:8202" in out
    assert "cubepick: conformant" in out
