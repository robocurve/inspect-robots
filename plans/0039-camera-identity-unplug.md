# Camera identity unplug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the setup wizard's camera listing and unplug-to-identify flow operate on physical USB cameras instead of symlink names, so a camera whose color node lost udev's by-id name race (issue #261, RealSense D435) or whose by-path name is duplicated by `usbv2-`/`usbv3-` aliases is still listed, identifiable by one unplug, and stored under a name that survives replugs.

**Architecture:** A new inventory scan resolves every symlink in both `/dev/v4l` listing directories to its target, probes each target once for color capture, and attaches the target's sysfs USB device (its physical-camera identity) and serial. Listing rows and the stored config value are derived from records by a trust ladder (by-id name if present and its serial is unique, else the plain by-path name, else the raw node), and unplug-identify diffs the set of physical cameras, re-deriving the stored name from the post-replug scan because udev re-rolls symlinks on every plug event. CAN and serial slots keep the existing string-diff flow untouched.

**Tech Stack:** Python 3.10+, stdlib only (`pathlib`, `re`, `collections.Counter`), pytest with injected `input_fn`/`out` seams and tmp_path fake `/dev` + `/sys` trees.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict, covers `src` and `tests`), `uv run pytest --cov` at **100% coverage**.
- Every public module/class/function needs a docstring; state the contract (ruff D1).
- Repo root is the `ir-wt-unplug-identify` worktree at `~/robocurve/ir-wt-unplug-identify`; run everything via `uv run ...` there.
- Wizard behavior for rigs whose cameras all have healthy, unique by-id links must be byte-for-byte unchanged: every existing golden-config test in `tests/test_setup.py` must pass untouched, EXCEPT tests that specifically pin the old broken behavior (the by-path-extra "press 'p'" hint wording and any direct `_scan_cameras`/`_identify_by_replug` call sites); those are updated deliberately and called out in Task 4.
- Non-Linux / no-sysfs environments (Windows workstations editing rig configs, CI) must keep working via one crisp regime rule: **an empty inventory (no node probed color-capable) means legacy behavior everywhere** — raw per-directory listings, the old `active_is_by_id` and `advertise_path_toggle` formulas, the old `_print_camera_path_hint` (kept, not deleted), and 'u' delegating to the legacy string-diff `_identify_by_replug` with its historical message copy. Every existing test whose fixtures probe `None` (touch()-ed files) therefore passes byte-for-byte, including the run_setup 'u'-flow tests at tests/test_setup.py:1378 and :1412. A non-empty inventory switches to the new device-identity behavior.
- Commit messages: imperative, scoped; reference #261 as motivation where apt.

## Reference: current wiring (main @ f751d5e1)

- `_setup.py:46-49`: injectable listing roots (`V4L_BY_ID`, `V4L_BY_PATH`, `SYSFS_NET`, `SERIAL_BY_ID`); `run_setup` (line 974) takes them as keyword params with these defaults, and `cli.py:2110-2118` (`_cmd_setup`) passes none of them — a new `sysfs_video` param with a default is fully backward compatible.
- `_setup.py:752-790`: `_v4l2_color_capture(path) -> bool | None` — the per-node color probe. It works on any path (symlink or raw node); tests monkeypatch it via `monkeypatch.setattr("inspect_robots._setup._v4l2_color_capture", ...)` (see `tests/test_setup.py:166-185`) or fake the ioctls themselves (`_fake_v4l2_ioctl`, line 236).
- `_setup.py:793-800`: `_scan_cameras(v4l_dir)` — per-directory color filter with the `color_entries or entries` raw fallback. Deleted in Task 4 (the fallback semantics move into `_camera_rows`).
- `_setup.py:275-322`: `_identify_by_replug(role, devices, *, input_fn, out, rescan, nouns...)` — diffs the caller's stale `devices` snapshot against one `rescan()`; the `no ... disappeared` and `unplug only one` messages live here. Kept for CAN/serial (Task 3 makes it rescan at entry); cameras get their own `_identify_camera_by_replug`.
- `_setup.py:325-345`: `_device_slot_prompt` — builds the prompt string, including the `(current)` / `(current, not detected)` suffix via `current in devices`.
- `_setup.py:348-505`: `_prompt_device_slot` — the per-slot loop: 's'/'p'/'u'/Enter/absolute-path/number handling, the `nouns` dict for 'u' (line 391), the current-not-detected warning branch (lines 409-430), and the already-assigned confirmation. Its `rescan_by_id`/`rescan_by_path` params exist only for the 'u' path.
- `_setup.py:508-574`: `_camera_section` — fallback three-camera interview (`CAM_ROLES`, line 44); computes `by_id_devices`/`by_path_devices` once, `active_is_by_id = bool(by_id_devices) or not by_path_devices` (line 519), `advertise_path_toggle = len(by_path_devices) > len(by_id_devices)` (line 522).
- `_setup.py:606-722`: `_device_section` — plugin-declared slots; same camera scans (lines 620-621, 624-625) plus CAN/serial; `prompt_slot` builds per-kind rescan callables (lines 643-665).
- `_setup.py:249-272`: `_print_camera_path_hint` — the "N by-path camera nodes have no by-id entry" warning, computed by resolved-target set difference. Replaced in Task 4 by `_print_camera_name_hint` (record-based, new copy).
- `tests/test_setup.py:123-160`: `_register_device_slots` / `_empty_registry` fixtures; line ~553: the golden-config harness (env with `XDG_CONFIG_HOME=tmp_path`, `DISPLAY: ":0"`, scripted `input_fn` that records prompts, `io.StringIO()` out) — copy this scaffold for every new run_setup-level test. Prompt strings go to `input_fn`, NOT `out`; assert prompt text against the recorded prompts list.
- `docs/guide/cli.md:116-129`: the wizard section documenting 'u' and 'p'.
- Sysfs ground truth (from the #261 rig): `/sys/class/video4linux/video10/device` resolves to `.../usb4/4-9/4-9:1.3`; walking up parents, the first directory containing an `idVendor` file is the USB device dir `.../usb4/4-9`, whose `serial` file reads `310323023943`. Fake trees in tests mirror exactly this shape.

---

### Task 1: inventory primitives (`_CameraNode`, sysfs walk, alias preference, `_camera_inventory`)

**Files:**
- Modify: `src/inspect_robots/_setup.py` (new block between `_v4l2_color_capture` and `_scan_cameras`; add `from collections import Counter`, `from dataclasses import dataclass` imports; `re` is already imported)
- Test: `tests/test_setup.py`

**Interfaces:**
- Produces (Tasks 2-5 consume all of these):
  - `_CameraNode(node: str, camera: str | None, serial: str | None, by_id: str | None, by_path: str | None)` (frozen dataclass)
  - `_usb_device_dir(node_name: str, sysfs_video: Path) -> Path | None`
  - `_prefer_plain_alias(candidates: list[Path]) -> Path`
  - `_camera_inventory(by_id_dir: Path, by_path_dir: Path, sysfs_video: Path) -> list[_CameraNode]`
  - Module constant `SYSFS_VIDEO: Path = Path("/sys/class/video4linux")` next to the other roots at line 46.

- [ ] **Step 1: Write the failing tests**

Add a fixture builder near the existing camera fixtures. It fabricates the #261 rig in tmp_path and is reused by Tasks 2-5:

```python
def _rig(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A fake /dev + /sys tree reproducing the #261 D435 rig.

    dev/video10 is the D435's only color node; its by-id link is MISSING
    (lost udev's name race) while by-id index0/index1 point at the depth
    node (video0, no color) and metadata node (video11). by-path has the
    plain usb- name AND a usbv3- alias for video10. dev/video8 is a D405
    with a healthy by-id link. Returns (by_id, by_path, sysfs_video, dev).
    """
    dev = tmp_path / "dev"
    by_id = tmp_path / "by-id"
    by_path = tmp_path / "by-path"
    sysfs_video = tmp_path / "sys-video"
    devices = tmp_path / "sys-devices"
    for directory in (dev, by_id, by_path, sysfs_video):
        directory.mkdir()
    for node in ("video0", "video8", "video10", "video11"):
        (dev / node).touch()
    _symlink(by_id / "usb-D435_310323023943-video-index0", dev / "video0")
    _symlink(by_id / "usb-D435_310323023943-video-index1", dev / "video11")
    _symlink(by_id / "usb-D405_429423070256-video-index4", dev / "video8")
    _symlink(by_path / "pci-0000:80:14.0-usb-0:9:1.3-video-index0", dev / "video10")
    _symlink(by_path / "pci-0000:80:14.0-usbv3-0:9:1.3-video-index0", dev / "video10")
    _symlink(by_path / "pci-0000:80:14.0-usb-0:9:1.0-video-index0", dev / "video0")
    _symlink(by_path / "pci-0000:80:14.0-usb-0:1.4:1.0-video-index4", dev / "video8")
    _usb_device(
        devices, "4-9", "310323023943", sysfs_video,
        {"video0": "1.0", "video10": "1.3", "video11": "1.3"},
    )
    _usb_device(devices, "1-4", "429423070256", sysfs_video, {"video8": "1.0"})
    return by_id, by_path, sysfs_video, dev


def _symlink(link: Path, target: Path) -> None:
    """A symlink helper that skips where symlinks are unavailable (Windows)."""
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - Windows without symlink privilege
        pytest.skip("symlinks unavailable")


def _usb_device(
    devices: Path, port: str, serial: str, sysfs_video: Path, nodes: dict[str, str]
) -> None:
    """One fake sysfs USB device mirroring real sysfs shape.

    ``nodes`` maps video-node name to USB interface suffix ("video10": "1.3");
    each node's ``device`` link points at the interface directory itself
    (``<port>/<port>:1.3``), exactly like real sysfs, and the USB device dir
    above it holds ``idVendor`` + ``serial``.
    """
    usb_dir = devices / port
    usb_dir.mkdir(parents=True, exist_ok=True)
    (usb_dir / "idVendor").write_text("8086", encoding="utf-8")
    (usb_dir / "serial").write_text(serial + "\n", encoding="utf-8")
    for node, suffix in nodes.items():
        interface = usb_dir / f"{port}:{suffix}"
        interface.mkdir(exist_ok=True)
        entry = sysfs_video / node
        entry.mkdir()
        _symlink(entry / "device", interface)
```

Call sites in `_rig`: `_usb_device(devices, "4-9", "310323023943", sysfs_video, {"video0": "1.0", "video10": "1.3", "video11": "1.3"})` and `_usb_device(devices, "1-4", "429423070256", sysfs_video, {"video8": "1.0"})` — video10 and video11 sharing the `1.3` interface dir is deliberate (that is how a real UVC interface's capture + metadata nodes look).

A color-probe fake keyed by RESOLVED TARGET name. It returns True/False only; the empty-inventory (legacy-regime) tests instead monkeypatch the probe to `lambda _path: None` — the inconclusive verdict every touch()-ed fixture file gets on a real run. Beware False-vs-None in `_prompt_device_slot`'s warned-current branch (`_setup.py:422`): a False-returning fake makes any Enter-accepted carried current print "offers no color capture format", so run_setup tests that carry a config keep their currents either color-capable or nonexistent:

```python
def _color_by_node(monkeypatch: pytest.MonkeyPatch, color: set[str]) -> None:
    """Fake _v4l2_color_capture: True iff the path's basename is in ``color``."""
    monkeypatch.setattr(
        "inspect_robots._setup._v4l2_color_capture",
        lambda path: Path(path).name in color,
    )
```

Then the tests:

```python
def test_camera_inventory_groups_names_by_resolved_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_id, by_path, sysfs_video, dev = _rig(tmp_path)
    _color_by_node(monkeypatch, {"video10", "video8"})
    inventory = _camera_inventory(by_id, by_path, sysfs_video)
    by_node = {Path(record.node).name: record for record in inventory}
    assert set(by_node) == {"video10", "video8"}  # depth/metadata nodes excluded
    d435 = by_node["video10"]
    assert d435.by_id is None  # the race loser has no by-id name
    assert d435.by_path is not None and "-usbv" not in Path(d435.by_path).name
    assert d435.serial == "310323023943"
    assert d435.camera is not None and d435.camera.endswith("4-9")
    d405 = by_node["video8"]
    assert d405.by_id is not None and Path(d405.by_id).name.startswith("usb-D405")


def test_camera_inventory_probes_each_target_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # video10 is reachable via two by-path aliases; count probe calls per
    # target with a recording fake and assert each target is probed exactly
    # once (aliases must not triple the ioctl traffic).
    ...


def test_usb_device_dir_walks_to_idvendor_and_survives_missing_sysfs(
    tmp_path: Path,
) -> None:
    by_id, by_path, sysfs_video, dev = _rig(tmp_path)
    assert _usb_device_dir("video10", sysfs_video) is not None
    assert _usb_device_dir("video10", tmp_path / "absent") is None
    assert _usb_device_dir("nonexistent-node", sysfs_video) is None


def test_usb_device_dir_exhausted_walk_returns_none(tmp_path: Path) -> None:
    # A device link resolving into a tree with NO idVendor anywhere above it
    # (e.g. a platform/CSI camera): the ancestor walk must exhaust and
    # return None instead of finding a phantom USB device.
    sysfs_video = tmp_path / "sys-video"
    platform = tmp_path / "sys-devices" / "platform" / "csi0"
    platform.mkdir(parents=True)
    entry = sysfs_video / "video99"
    entry.mkdir(parents=True)
    _symlink(entry / "device", platform)
    assert _usb_device_dir("video99", sysfs_video) is None


def test_prefer_plain_alias_picks_usb_over_usbv_variants() -> None:
    plain = Path("pci-0000:80:14.0-usb-0:9:1.3-video-index0")
    v2 = Path("pci-0000:80:14.0-usbv2-0:9:1.3-video-index0")
    v3 = Path("pci-0000:80:14.0-usbv3-0:9:1.3-video-index0")
    assert _prefer_plain_alias([v3, plain, v2]) == plain
    assert _prefer_plain_alias([v3, v2]) == v2  # deterministic without a plain name


def test_camera_inventory_missing_directories_and_serial_are_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # by_id dir absent entirely + a camera whose sysfs USB dir has no serial
    # file: inventory still returns the by-path record with serial None.
    ...
```

Flesh the `...` bodies against the fixture; keep every test mypy-strict (annotate returns `-> None`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_setup.py -k "camera_inventory or usb_device_dir or prefer_plain" -v`
Expected: FAIL at import (`ImportError: cannot import name '_CameraNode'` — extend the test file's existing `from inspect_robots._setup import ...` block).

- [ ] **Step 3: Implement**

In `src/inspect_robots/_setup.py`, after `_v4l2_color_capture`:

```python
SYSFS_VIDEO: Path = Path("/sys/class/video4linux")   # goes next to line 46-49 roots

_USB_SPEED_ALIAS = re.compile(r"-usbv\d+-")


@dataclass(frozen=True)
class _CameraNode:
    """One color-capable camera node and every name that reaches it.

    ``node`` is the resolved real device path and the diffing identity when
    sysfs cannot group nodes into physical cameras; ``camera`` is the sysfs
    USB device directory owning the node (one per physical camera) or
    ``None`` where sysfs is unavailable; ``serial`` is the USB serial when
    readable; ``by_id``/``by_path`` are the preferred symlinks resolving to
    the node in each listing directory, when any exist.
    """

    node: str
    camera: str | None
    serial: str | None
    by_id: str | None
    by_path: str | None


def _usb_device_dir(node_name: str, sysfs_video: Path) -> Path | None:
    """Sysfs USB device directory owning a video node, if resolvable.

    Walks from ``<sysfs_video>/<node>/device`` up to the first ancestor
    holding an ``idVendor`` file — the USB *device* (not interface) dir,
    which is the physical-camera identity shared by all the device's nodes.
    """
    try:
        device = (sysfs_video / node_name / "device").resolve(strict=True)
    except OSError:
        return None
    # is_file() swallows ENOENT/ENOTDIR itself, so the walk needs no
    # per-ancestor exception handling (a try/except here would be dead code
    # under the 100% branch gate).
    for ancestor in (device, *device.parents):
        if (ancestor / "idVendor").is_file():
            return ancestor
    return None


def _prefer_plain_alias(candidates: list[Path]) -> Path:
    """The canonical name among symlinks to one node.

    Newer systemd publishes ``usbv2-``/``usbv3-`` speed-qualified aliases
    next to the plain ``usb-`` by-path name; prefer the plain name, then
    tie-break lexicographically for determinism.
    """
    return min(candidates, key=lambda path: (bool(_USB_SPEED_ALIAS.search(path.name)), path.name))


def _camera_inventory(
    by_id_dir: Path, by_path_dir: Path, sysfs_video: Path
) -> list[_CameraNode]:
    """Color-capable camera nodes with their physical identity and names.

    Symlinks from both listing directories are resolved and grouped by
    target, each target is probed once, and each color-capable target
    becomes one record. Grouping by resolved target rather than by name is
    what lets udev alias duplicates and missing by-id links be survivable.
    """
    names: dict[Path, dict[str, list[Path]]] = {}
    for directory, kind in ((by_id_dir, "by_id"), (by_path_dir, "by_path")):
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            target = entry.resolve(strict=False)
            names.setdefault(target, {"by_id": [], "by_path": []})[kind].append(entry)
    records: list[_CameraNode] = []
    for target in sorted(names):
        if _v4l2_color_capture(target) is not True:
            continue
        usb_dir = _usb_device_dir(target.name, sysfs_video)
        serial: str | None = None
        if usb_dir is not None:
            try:
                serial = (usb_dir / "serial").read_text(encoding="utf-8").strip() or None
            except OSError:
                serial = None
        by_id = names[target]["by_id"]
        by_path = names[target]["by_path"]
        records.append(
            _CameraNode(
                node=str(target),
                camera=None if usb_dir is None else str(usb_dir),
                serial=serial,
                by_id=str(_prefer_plain_alias(by_id)) if by_id else None,
                by_path=str(_prefer_plain_alias(by_path)) if by_path else None,
            )
        )
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_setup.py -k "camera_inventory or usb_device_dir or prefer_plain" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: camera inventory grouped by resolved target and sysfs USB device (#261)"
```

---

### Task 2: trust-ladder naming (`_duplicated_serials`, `_preferred_name`, `_camera_rows`)

**Files:**
- Modify: `src/inspect_robots/_setup.py` (directly after `_camera_inventory`)
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: `_CameraNode`, `_camera_inventory` (Task 1).
- Produces (Tasks 3-5 consume):
  - `_duplicated_serials(inventory: list[_CameraNode]) -> set[str]`
  - `_preferred_name(records: list[_CameraNode], duplicated: set[str], *, prefer_by_id: bool) -> str`
  - `_camera_rows(inventory: list[_CameraNode], directory: Path, *, by_id: bool) -> list[str]`

- [ ] **Step 1: Write the failing tests**

```python
def test_camera_rows_by_id_falls_back_to_by_path_for_race_losers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_id, by_path, sysfs_video, dev = _rig(tmp_path)
    _color_by_node(monkeypatch, {"video10", "video8"})
    inventory = _camera_inventory(by_id, by_path, sysfs_video)
    rows = _camera_rows(inventory, by_id, by_id=True)
    names = [Path(row).name for row in rows]
    # the D435 appears via its plain by-path name; the D405 via by-id
    assert "pci-0000:80:14.0-usb-0:9:1.3-video-index0" in names
    assert "usb-D405_429423070256-video-index4" in names
    assert len(rows) == 2  # one row per camera node, aliases collapsed


def test_camera_rows_by_path_view_dedupes_usbv_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # by_id=False: exactly one row for video10 (the plain usb- name), one for
    # video8; no usbv3- row.
    ...


def test_camera_rows_shared_serial_distrusts_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build two Innomaker-style cameras (video13, video15), DISTINCT sysfs
    # USB dirs ("3-2", "3-4") but the SAME serial "SN0001"; give video15 a
    # by-id link and both by-path links. The by-id view must list BOTH via
    # by-path names (the by-id name is ambiguous across replugs), and
    # _duplicated_serials returns {"SN0001"}.
    ...


def test_camera_rows_raw_fallback_when_no_color_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Probe returns None everywhere (non-Linux): _camera_inventory is empty
    # and _camera_rows returns the raw sorted listing of the directory it is
    # asked about — the exact `color_entries or entries` behavior today —
    # and [] for a missing directory.
    ...


def test_preferred_name_ladder() -> None:
    trusted = _CameraNode(node="/dev/video8", camera="c1", serial="A", by_id="/i/a", by_path="/p/a")
    raceless = _CameraNode(node="/dev/video10", camera="c2", serial="B", by_id=None, by_path="/p/b")
    bare = _CameraNode(node="/dev/video3", camera=None, serial=None, by_id=None, by_path=None)
    assert _preferred_name([trusted], set(), prefer_by_id=True) == "/i/a"
    assert _preferred_name([trusted], {"A"}, prefer_by_id=True) == "/p/a"  # shared serial
    assert _preferred_name([trusted], set(), prefer_by_id=False) == "/p/a"
    assert _preferred_name([raceless], set(), prefer_by_id=True) == "/p/b"
    assert _preferred_name([bare], set(), prefer_by_id=True) == "/dev/video3"
    # multi-node camera: the by-id-bearing node wins the camera's name
    assert _preferred_name([raceless, trusted], set(), prefer_by_id=True) == "/i/a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_setup.py -k "camera_rows or preferred_name or duplicated" -v`
Expected: FAIL at import.

- [ ] **Step 3: Implement**

```python
def _duplicated_serials(inventory: list[_CameraNode]) -> set[str]:
    """Serials shared by more than one physical camera in the inventory.

    A by-id name embedding such a serial is ambiguous: udev lets the two
    devices overwrite each other's links on every replug, so the name can
    silently swap cameras.
    """
    per_camera = {record.camera or record.node: record for record in inventory}
    counts = Counter(
        record.serial for record in per_camera.values() if record.serial is not None
    )
    return {serial for serial, count in counts.items() if count > 1}


def _preferred_name(
    records: list[_CameraNode], duplicated: set[str], *, prefer_by_id: bool
) -> str:
    """The trust-ladder name for one camera's color nodes.

    Ladder: a by-id name whose serial is not shared by another camera (it
    survives port moves), else the plain by-path name (stable per physical
    USB port), else the raw node path. ``prefer_by_id=False`` (the 'p'
    port-name view) skips the first rung.
    """
    ranked: list[tuple[int, str]] = []
    for record in records:
        by_id = record.by_id
        trusted = by_id is not None and (
            record.serial is None or record.serial not in duplicated
        )
        if prefer_by_id and by_id is not None and trusted:
            ranked.append((0, by_id))
        elif record.by_path is not None:
            ranked.append((1, record.by_path))
        else:
            ranked.append((2, record.node))
    return min(ranked)[1]


def _camera_rows(inventory: list[_CameraNode], directory: Path, *, by_id: bool) -> list[str]:
    """Listing rows for one view, one row per color-capable node.

    Every camera is always listed: a node whose by-id link is missing or
    ambiguous shows its port-stable by-path name instead of vanishing. When
    no color-capable node is confirmed at all (non-Linux, or probing
    unavailable) the raw directory listing is offered unfiltered, matching
    the historical ``_scan_cameras`` fallback.
    """
    duplicated = _duplicated_serials(inventory)
    rows = sorted(
        {_preferred_name([record], duplicated, prefer_by_id=by_id) for record in inventory}
    )
    if rows:
        return rows
    try:
        return [str(entry) for entry in sorted(directory.iterdir())]
    except OSError:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_setup.py -k "camera_rows or preferred_name or duplicated" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: trust-ladder camera naming with alias dedupe and raw fallback (#261)"
```

---

### Task 3: camera-identity unplug (`_identify_camera_by_replug`; fresh-before rescans)

**Files:**
- Modify: `src/inspect_robots/_setup.py` (`_identify_by_replug` at 275; new function after it)
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: `_CameraNode`, `_camera_inventory`, `_duplicated_serials`, `_preferred_name`, `_camera_rows`.
- Produces (Task 4 consumes): `_identify_camera_by_replug(role: str, *, input_fn, out, rescan: Callable[[], list[_CameraNode]], prefer_by_id: bool, by_id_dir: Path, by_path_dir: Path, unplug_label: str | None = None) -> str | None`. Also changes `_identify_by_replug` to drop its `devices` parameter and take the "before" snapshot from `rescan()` at entry (a camera/CAN device plugged in after the section listing was printed is otherwise undetectable forever).
- Regime rule (Global Constraints): when the entry rescan yields an EMPTY inventory, `_identify_camera_by_replug` delegates to the legacy `_identify_by_replug` over the active directory's rows (which are the raw listing in that regime), with the default camera nouns — so no-probe environments keep the historical flow and message copy ("no camera device disappeared") byte-for-byte. The new copy ("no camera disappeared" / "N cameras disappeared") appears only on probed rigs, where the unit really is a physical camera; this copy divergence between regimes is deliberate.

- [ ] **Step 1: Write the failing tests**

Drive the fake rig through unplug cycles by mutating the tmp tree from inside a scripted `input_fn` (the existing direct `_identify_by_replug` tests show the pattern — find them with `grep -n "_identify_by_replug" tests/test_setup.py` and mirror the idiom):

```python
def test_identify_camera_finds_by_id_invisible_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#261 exact scenario: D435 absent from by-id view, one unplug finds it."""
    by_id, by_path, sysfs_video, dev = _rig(tmp_path)
    _color_by_node(monkeypatch, {"video10", "video8"})
    # unplugging the D435 removes BOTH by-path aliases, its dev node, and its
    # sysfs entry; replugging restores them (helper _unplug/_replug below).
    ...
    selected = _identify_camera_by_replug(
        "top", input_fn=script, out=out,
        rescan=lambda: _camera_inventory(by_id, by_path, sysfs_video),
        prefer_by_id=True,
    )
    assert selected is not None
    assert Path(selected).name == "pci-0000:80:14.0-usb-0:9:1.3-video-index0"


def test_identify_camera_alias_pair_counts_as_one_camera(...) -> None:
    # by-path view (prefer_by_id=False): the usb- and usbv3- aliases vanish
    # together on unplug; the flow must NOT print "2 camera devices
    # disappeared" — it identifies one camera and returns the plain name.


def test_identify_camera_rederives_name_after_replug_reroll(...) -> None:
    # Before: video10 has no by-id link. During _replug, CREATE
    # "usb-D435_310323023943-video-index0" -> video10 (the race re-rolled the
    # good way) and REMOVE the old index0 link to video0. The returned name
    # must be the FRESH by-id link, not the stale by-path fallback.


def test_identify_camera_two_cameras_unplugged_is_rejected(...) -> None:
    # Remove both the D435 and D405 trees before the rescan: message contains
    # "2 cameras disappeared; unplug only one" and the function returns None.


def test_identify_camera_nothing_unplugged_reports_and_returns_none(...) -> None:
    # No mutation between prompts: "no camera disappeared" printed, None.


def test_identify_camera_not_reappearing_warns_and_keeps_assignment(...) -> None:
    # Unplug succeeds, but the camera never comes back through the extra
    # "press Enter to rescan" retry: warning printed, the pre-unplug
    # trust-ladder name is still returned (assignment kept).


def test_identify_camera_late_arrival_is_detectable(...) -> None:
    # A camera plugged in AFTER the wizard's listing was printed (i.e. absent
    # from any earlier snapshot) is still identified: the flow's "before" set
    # comes from a fresh rescan, not the section-start listing.


def test_identify_camera_empty_inventory_delegates_to_legacy_flow(...) -> None:
    # Probe monkeypatched to `lambda _path: None` (no inventory): removing a
    # by-id entry between prompts identifies it through the legacy row diff,
    # the failure copy on a no-op unplug is the HISTORICAL "no camera device
    # disappeared", and the returned value is the raw listing row. Exercise
    # both prefer_by_id=True (diffs the by-id dir) and False (by-path dir).
```

Also update any existing direct-call tests of `_identify_by_replug` for the dropped `devices` param (the "before" list now comes from an extra leading `rescan()`; scripted rescan fakes that pop from a list need one more element at the front).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_setup.py -k identify_camera -v`
Expected: FAIL at import (`_identify_camera_by_replug` undefined).

- [ ] **Step 3: Implement**

Change `_identify_by_replug`'s signature from `(role, devices, *, ...)` to `(role, *, ...)` and open the body with `devices = rescan()` before the unplug prompt (docstring: the before snapshot is taken fresh so devices attached after the listing printed still diff). Then add:

```python
def _identify_camera_by_replug(
    role: str,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
    rescan: Callable[[], list[_CameraNode]],
    prefer_by_id: bool,
    by_id_dir: Path,
    by_path_dir: Path,
    unplug_label: str | None = None,
) -> str | None:
    """Identify one physical camera by diffing USB devices while unplugged.

    Diffs cameras (sysfs USB device, falling back to the resolved node)
    rather than listing rows, so a camera invisible in the by-id view or
    duplicated by by-path aliases is still identified by a single unplug.
    The stored name is re-derived from the post-replug scan because udev
    re-rolls symlinks on every plug event. With an empty inventory (no node
    probed color-capable: non-Linux, or probing unavailable) the legacy
    row-diff flow runs instead, over the active directory's raw listing.
    """

    def grouped(inventory: list[_CameraNode]) -> dict[str, list[_CameraNode]]:
        cameras: dict[str, list[_CameraNode]] = {}
        for record in inventory:
            cameras.setdefault(record.camera or record.node, []).append(record)
        return cameras

    def name_of(cameras: dict[str, list[_CameraNode]], key: str) -> str:
        flat = [record for records in cameras.values() for record in records]
        return _preferred_name(cameras[key], _duplicated_serials(flat), prefer_by_id=prefer_by_id)

    before_inventory = rescan()
    if not before_inventory:
        directory = by_id_dir if prefer_by_id else by_path_dir
        return _identify_by_replug(
            role,
            input_fn=input_fn,
            out=out,
            rescan=lambda: _camera_rows(rescan(), directory, by_id=prefer_by_id),
            unplug_label=unplug_label,
        )

    label = unplug_label if unplug_label is not None else f"{role} camera"
    before = grouped(before_inventory)
    input_fn(f"Unplug the {label} now, then press Enter...")
    after = grouped(rescan())
    gone = [key for key in before if key not in after]
    if not gone:
        print(
            _paint("no camera disappeared; unplug one camera and try again", _YELLOW, out),
            file=out,
        )
        return None
    if len(gone) > 1:
        print(
            _paint(
                f"{len(gone)} cameras disappeared; unplug only one and try again", _YELLOW, out
            ),
            file=out,
        )
        return None

    key = gone[0]
    display = Path(name_of(before, key)).name
    print(f"That was: {_paint(display, _GREEN, out)}", file=out)
    input_fn("Plug it back in, then press Enter...")
    final = grouped(rescan())
    if key not in final:
        input_fn(f"{display} was not detected; press Enter to rescan...")
        final = grouped(rescan())
    if key not in final:
        print(
            _paint(
                f"warning: {display} was still not detected; keeping the assignment",
                _YELLOW,
                out,
            ),
            file=out,
        )
        return name_of(before, key)
    return name_of(final, key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_setup.py -v`
Expected: PASS (including the adjusted `_identify_by_replug` direct tests).

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: identify cameras by USB device, re-derive name after replug (#261)"
```

---

### Task 4: wire the inventory through the wizard

**Files:**
- Modify: `src/inspect_robots/_setup.py` (`_print_camera_path_hint` 249; `_prompt_device_slot` 348; `_camera_section` 508; `_device_section` 606; `run_setup` 974; delete `_scan_cameras` 793)
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run_setup(..., sysfs_video: Path = SYSFS_VIDEO)` — new keyword param, threaded to both sections.
  - `_prompt_device_slot` replaces `rescan_by_id`/`rescan_by_path` with one `identify: Callable[[bool], str | None]` (the bool is the live `active_is_by_id`); its `nouns` dict moves into the CAN/serial identify closure as a per-kind dict LOOKUP (data, not an if/else — a serial-slot branch would otherwise need its own 'u' test to satisfy branch coverage).
  - `_camera_identify(role, unplug_label, *, input_fn, out, rescan, by_id_dir, by_path_dir) -> Callable[[bool], str | None]` — module-level (both sections call the SAME helper; a per-section closure would leave the `_device_section` copy's body uncovered since no existing test presses 'u' on a v4l2 device slot).
  - `_print_camera_name_hint(inventory, active_is_by_id, out)` — used when the inventory is non-empty; the legacy `_print_camera_path_hint` is KEPT and still used in the empty-inventory regime, so its existing tests pass untouched.
  - `advertise_path_toggle` and `active_is_by_id`: records-based formulas when the inventory is non-empty, the historical row-count/row-emptiness formulas otherwise (see Step 3 code — the empty-inventory arm is exactly today's lines 519/522 semantics).

- [ ] **Step 1: Write the failing tests**

End-to-end through `run_setup` with the golden-config harness plus the `_rig` fixture (pass `by_id_dir=`/`by_path_dir=`/`sysfs_video=` into `run_setup`):

```python
def test_run_setup_lists_race_loser_camera_and_selects_it_by_number(...) -> None:
    # The #261 transcript, fixed: the by-id listing has 2 rows (D435 via its
    # by-path name); selecting its number writes the by-path name into
    # [embodiment.args] top_cam_device; the hint line explains the fallback
    # ("no usable by-id entry"). Camera section only (no DEVICE_SLOTS).


def test_run_setup_unplug_identifies_race_loser_camera(...) -> None:
    # Answer 'u' for the top camera and mutate the tree from the scripted
    # input_fn exactly as in Task 3's direct test; the written config holds
    # the plain by-path name.


def test_run_setup_shared_serial_cameras_both_listed_by_path(...) -> None:
    # Two same-serial cameras (Task 2 fixture): both rows are by-path names;
    # assigning both to different roles works without the duplicate-warning.


def test_run_setup_all_race_losers_starts_in_port_view_hint_without_p(...) -> None:
    # A rig whose ONLY camera is the D435 (no camera has a by-id link, probe
    # True): active_is_by_id computes False, the listing shows the by-path
    # row, and the hint prints WITHOUT the "press 'p'" suffix. This is the
    # only test exercising the records-based active_is_by_id=False decision
    # and _print_camera_name_hint's active_is_by_id=False arc.


def test_run_setup_healthy_rig_prompts_and_config_unchanged(...) -> None:
    # Regression guard: a rig whose every camera has a healthy unique by-id
    # link (fixture with by-id + by-path + sysfs, no races) produces the
    # same listing rows, NO hint line, NO 'p' advertisement, and the same
    # written config as the equivalent pre-change fixture. Mirror an
    # existing golden test's fixture shape and assert its exact expectations
    # still hold with sysfs present.
```

Update deliberately-changed existing tests, and ONLY these kinds:
- tests pinning the old hint copy ("by-path camera nodes have no by-id entry") — new copy is asserted in the first test above;
- tests pinning `advertise_path_toggle` purely on row counts;
- direct `_scan_cameras` unit tests — fold their fallback/missing-dir cases into `_camera_rows` tests (Task 2 already covers them; delete the leftovers);
- direct `_prompt_device_slot` calls, for the new `identify` parameter.

Every other existing test must pass byte-for-byte. If one fails, treat it as a bug in the new code, not a test to update (the Global Constraints exception list is exhaustive).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_setup.py -v`
Expected: new tests FAIL (rows missing, 'u' cannot find the D435); existing tests still pass.

- [ ] **Step 3: Implement**

In `_camera_section` (and symmetrically in `_device_section` for its v4l2 arm) — note the regime split; the empty-inventory arm reproduces today's lines 519/522 exactly, because in that regime the rows ARE the raw per-directory listings:

```python
    inventory = _camera_inventory(by_id_dir, by_path_dir, sysfs_video)
    duplicated = _duplicated_serials(inventory)
    by_id_devices = _camera_rows(inventory, by_id_dir, by_id=True)
    by_path_devices = _camera_rows(inventory, by_path_dir, by_id=False)
    if inventory:
        active_is_by_id = any(record.by_id for record in inventory) or not any(
            record.by_path for record in inventory
        )
        advertise_path_toggle = any(
            record.by_id is None
            or (record.serial is not None and record.serial in duplicated)
            for record in inventory
        )
    else:
        active_is_by_id = bool(by_id_devices) or not by_path_devices
        advertise_path_toggle = len(by_path_devices) > len(by_id_devices)
    rescan_inventory = partial(_camera_inventory, by_id_dir, by_path_dir, sysfs_video)
```

The hint call site branches on the same regime (both sections):

```python
    if inventory:
        _print_camera_name_hint(inventory, active_is_by_id, out)
    else:
        _print_camera_path_hint(by_id_devices, by_path_devices, active_is_by_id, out)
```

Module-level camera identify factory (both sections' v4l2 slots bind through this one function; `role` is the camera role or slot label):

```python
def _camera_identify(
    role: str,
    unplug_label: str | None,
    *,
    input_fn: Callable[[str], str],
    out: IO[str],
    rescan: Callable[[], list[_CameraNode]],
    by_id_dir: Path,
    by_path_dir: Path,
) -> Callable[[bool], str | None]:
    """Bind one camera slot's 'u' handler over the shared inventory rescan."""

    def identify(prefer_by_id: bool) -> str | None:
        return _identify_camera_by_replug(
            role,
            input_fn=input_fn,
            out=out,
            rescan=rescan,
            prefer_by_id=prefer_by_id,
            by_id_dir=by_id_dir,
            by_path_dir=by_path_dir,
            unplug_label=unplug_label,
        )

    return identify
```

CAN/serial identify closure in `_device_section.prompt_slot` (replacing the `nouns` dict that lived in `_prompt_device_slot`). The per-kind nouns stay a dict LOOKUP so no untested branch appears, and the rescan is the existing `rescan_primary` callable already built per kind in `prompt_slot` (`_setup.py:659/665`):

```python
        _NOUNS = {
            "can": ("CAN interface", "CAN interfaces", "CAN interface"),
            "serial": ("serial device", "serial devices", "serial device"),
        }
        ...
        noun, plural_noun, retry_noun = _NOUNS[slot.kind]

        def identify(_prefer_by_id: bool) -> str | None:
            return _identify_by_replug(
                slot.label,
                input_fn=input_fn,
                out=out,
                rescan=rescan_primary,
                noun=noun,
                plural_noun=plural_noun,
                retry_noun=retry_noun,
                unplug_label=slot.label,
            )
```

(Place `_NOUNS` at module level next to the other constants; the closure body is exercised by the existing CAN 'u' run_setup test.)

For v4l2 slots, `prompt_slot` passes `_camera_identify(slot.label, slot.label, input_fn=input_fn, out=out, rescan=rescan_inventory, by_id_dir=by_id_dir, by_path_dir=by_path_dir)` (device-slot labels name the camera in the unplug prompt, exactly like today's `unplug_label=label` path), while `_camera_section` passes `_camera_identify(role, None, ...)` so the prompt renders the historical `f"{role} camera"`.

`_prompt_device_slot`'s 'u' branch shrinks to:

```python
        if entered.lower() == "u":
            selected = identify(active_is_by_id)
            if selected is None:
                continue
```

`_print_camera_name_hint` replaces `_print_camera_path_hint`:

```python
def _print_camera_name_hint(
    inventory: list[_CameraNode], active_is_by_id: bool, out: IO[str]
) -> None:
    """Explain fallback rows: cameras whose by-id name is missing or ambiguous."""
    duplicated = _duplicated_serials(inventory)
    fallback = [
        record
        for record in inventory
        if record.by_id is None
        or (record.serial is not None and record.serial in duplicated)
    ]
    if not fallback:
        return
    names = ", ".join(
        Path(record.by_path or record.node).name for record in fallback
    )
    message = (
        f"{len(fallback)} camera node(s) have no usable by-id entry "
        "(udev name race between USB interfaces, or a serial shared by "
        f"two cameras): {names}; their port-stable by-path names are "
        "listed instead"
    )
    if active_is_by_id:
        message += "; press 'p' to see every camera by port name"
    print(_paint(message, _YELLOW, out), file=out)
```

`run_setup` gains `sysfs_video: Path = SYSFS_VIDEO` and passes it to both sections. Delete `_scan_cameras`. Keep `_scan_can`/`_scan_serial` and `_print_camera_path_hint` untouched.

Accepted cosmetics (do NOT fix; noting them so nobody "improves" them mid-implementation): the listing header still reads `Found N camera device(s) under {by_id_dir}` even when fallback rows are by-path paths (the hint line right below explains why), and mixed rows sort by full path so by-id rows precede by-path rows regardless of camera order. `_preferred_name` keeps its `by_id is not None` check alongside `trusted` purely for mypy narrowing — keep the comment saying so.

- [ ] **Step 4: Run tests to verify they pass, then the full gate set**

Run: `uv run pytest tests/test_setup.py -v`, then
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`
Expected: all green, 100% coverage (delete any now-unreachable code rather than excluding it).

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: list and identify cameras through the device inventory (#261)"
```

---

### Task 5: reconcile a saved-but-missing camera path by serial

**Files:**
- Modify: `src/inspect_robots/_setup.py` (the `warned_current` branch of `_prompt_device_slot`, lines 409-430)
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: `_CameraNode`, `_preferred_name`, `_duplicated_serials`.
- Produces: `_reconcile_missing_current(current: str, inventory: list[_CameraNode], *, prefer_by_id: bool) -> str | None` — a hint path for a saved by-id name whose serial matches a scanned camera. `_prompt_device_slot` gains an `inventory: list[_CameraNode]` parameter (empty for CAN/serial callers).

- [ ] **Step 1: Write the failing tests**

```python
def test_reconcile_missing_current_matches_serial_from_by_id_name() -> None:
    record = _CameraNode(
        node="/dev/video10",
        camera="/sys/devices/4-9",
        serial="310323023943",
        by_id=None,
        by_path="/dev/v4l/by-path/pci-0000:80:14.0-usb-0:9:1.3-video-index0",
    )
    saved = "/dev/v4l/by-id/usb-Intel_..._435_310323023943-video-index0"
    assert _reconcile_missing_current(saved, [record], prefer_by_id=True) == record.by_path
    assert _reconcile_missing_current("/dev/video99", [record], prefer_by_id=True) is None
    assert _reconcile_missing_current(saved, [], prefer_by_id=True) is None
    # two distinct cameras carrying the saved serial: refuse to guess
    twin = _CameraNode(
        node="/dev/video12",
        camera="/sys/devices/3-4",
        serial="310323023943",
        by_id=None,
        by_path="/dev/v4l/by-path/pci-0000:80:14.0-usb-0:3.4:1.0-video-index0",
    )
    assert _reconcile_missing_current(saved, [record, twin], prefer_by_id=True) is None


def test_run_setup_missing_current_prints_reconciliation_hint(...) -> None:
    # _rig + a carried config whose top_cam_device is the dead by-id name
    # embedding serial 310323023943: pressing Enter first triggers the
    # existing "does not exist here" warning PLUS a hint naming the by-path
    # row; entering that row's number then writes it. Assert both messages
    # arrive on `out` and the final config carries the by-path name.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_setup.py -k reconcile -v`
Expected: FAIL at import.

- [ ] **Step 3: Implement**

```python
_BY_ID_SERIAL = re.compile(r"_([A-Za-z0-9]+)-video-index\d+$")


def _reconcile_missing_current(
    current: str, inventory: list[_CameraNode], *, prefer_by_id: bool
) -> str | None:
    """The scanned camera matching a dead by-id name's embedded serial.

    udev by-id names end ``_<serial>-video-index<n>``; when the saved link
    is gone but a camera with that serial is on the bus, the operator should
    be told where it lives now instead of just "not detected".
    """
    match = _BY_ID_SERIAL.search(Path(current).name)
    if match is None:
        return None
    serial = match.group(1)
    records = [record for record in inventory if record.serial == serial]
    if not records:
        return None
    if len({record.camera or record.node for record in records}) > 1:
        # Two cameras carry this serial: pointing at either would be a guess.
        return None
    return _preferred_name(records, _duplicated_serials(inventory), prefer_by_id=prefer_by_id)
```

In the `warned_current` branch (after the existing two warnings, before `continue`), when a warning fired and `kind == "v4l2"`:

```python
                if warning is not None:
                    print(_paint(warning, _YELLOW, out), file=out)
                    replacement = _reconcile_missing_current(
                        current, inventory, prefer_by_id=active_is_by_id
                    )
                    if replacement is not None:
                        print(
                            _paint(
                                f"a camera with that serial is connected; its "
                                f"stable path is now {replacement}",
                                _YELLOW,
                                out,
                            ),
                            file=out,
                        )
                    warned_current = True
                    continue
```

- [ ] **Step 4: Run tests to verify they pass, then the full gate set**

Run: `uv run pytest tests/test_setup.py -v && uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`
Expected: green, 100%.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots/_setup.py tests/test_setup.py
git commit -m "setup: point a dead by-id camera path at its serial's new home (#261)"
```

---

### Task 6: docs

**Files:**
- Modify: `docs/guide/cli.md` (wizard section, lines 116-129)
- Modify: `CHANGELOG.md` (`## [Unreleased]` → `### Fixed`; match the file's heading structure)
- Modify: `src/inspect_robots/CLAUDE.md` (the `_setup.py` module-map row, line 36)

- [ ] **Step 1: cli guide**

Rewrite the two camera paragraphs: the wizard lists every color-capable camera, preferring `/dev/v4l/by-id` names and falling back to port-stable `/dev/v4l/by-path` names when a camera's by-id link is missing (multi-interface cameras like the RealSense D435 lose udev's name race between their depth and RGB interfaces) or when two cameras share one serial; `u` identifies the physical USB device that disappeared, so it works even for cameras the by-id listing cannot name, and the stored path is chosen after replug because udev reassigns links on every plug; `p` switches the listing to port names. Keep the CAN paragraph as is. Match the surrounding prose style and the repo writing rules (no em dashes in prose).

- [ ] **Step 2: CHANGELOG**

```markdown
- `inspect-robots setup` camera slots (#261, plan 0039): the wizard now lists
  and unplug-identifies cameras as physical USB devices. A camera whose color
  node lost udev's by-id name race (multi-interface cameras such as the
  RealSense D435) or whose by-path name is duplicated by systemd
  `usbv2-`/`usbv3-` aliases no longer vanishes from the listing or defeats
  `u`; shared-serial cameras are listed by port-stable by-path names, and a
  saved-but-dead by-id path now points the operator at the camera's current
  location by serial.
```

Match the existing entry indentation/format exactly.

- [ ] **Step 3: Module map**

Extend the `_setup.py` row's plan list with 0039 and swap "fallback camera discovery" phrasing for the new inventory summary (color-probed camera inventory grouped by sysfs USB device; trust-ladder naming; device-level unplug identify), matching the row's phrasing density.

- [ ] **Step 4: Gates + commit**

Run: `uv run pytest -q` (green tree), then:

```bash
git add docs/guide/cli.md CHANGELOG.md src/inspect_robots/CLAUDE.md
git commit -m "docs: describe device-identity camera listing and unplug flow (#261)"
```

---

## Out of scope

- Automatic udev-settle polling after replug. Known residual race, accepted deliberately: the "press Enter to rescan" retry only fires when the camera's KEY is absent from the post-replug scan; a partially settled camera (dev node + by-path present, by-id link still pending) is already in the inventory, so the trust ladder can land on the by-path rung moments before a by-id link would have appeared. The stored by-path name is still correct and port-stable — the cost is a less-portable name, not a broken config — and a timed poll would buy determinism at the price of plumbing a sleep through four functions. Revisit only if rigs show it mattering.
- A udev rule generator for stable camera names (the CAN pinning suggestion's analogue): worth a follow-up issue if by-path names prove insufficient on rigs where cameras move between ports.
- Fixing udev's multi-interface by-id collision itself (systemd upstream) or the `realsense_capture` serial-based open path in inspect-robots-yam: unaffected by this change.
- CAN/serial identify improvements beyond the fresh-before rescan (their single-name-per-device model does not have the camera problem).
