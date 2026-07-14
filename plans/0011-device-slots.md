# 0011 — Plugin-declared device slots: the wizard interviews buses like cameras

Fixes #61 (check 5 of #50). The wizard's camera flow is good because it
probes real devices; CAN buses get no probe, so a fresh config on a
udev-pinned rig silently inherits the plugin default (`can0`/`can1`) and the
first run dies with `Could not access SocketCAN device can0`. Components
declare their device-shaped config args; the wizard runs the
probe-and-interview appropriate to each kind. Cameras stop being a special
case and become one declared kind. Builds on plan 0010's declaration
mechanism (`RUNTIME_REQUIREMENTS`); a component may carry both.

## 1. Protocol: `DeviceSlot` in `conformance.py`

```python
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
```

Both live in `inspect_robots.conformance` next to
`missing_runtime_requirements`; submodule import, no `__all__` change.
Recognized kinds are a module constant `DEVICE_KINDS = ("v4l2", "can",
"serial")`.

Example declaration (the yam plugin's, tracked separately in
robocurve/inspect-robots-yam#30):

```python
DEVICE_SLOTS = (
    DeviceSlot(arg="left_channel",  kind="can",  label="left arm CAN channel"),
    DeviceSlot(arg="right_channel", kind="can",  label="right arm CAN channel"),
    DeviceSlot(arg="top_cam_device",   kind="v4l2", label="top camera", group="cameras"),
    DeviceSlot(arg="left_cam_device",  kind="v4l2", label="left camera", group="cameras"),
    DeviceSlot(arg="right_cam_device", kind="v4l2", label="right camera", group="cameras"),
)
```

## 2. Probes (`_setup.py`, pure and injectable)

- **v4l2**: existing `_scan_cameras` on `by_id_dir`/`by_path_dir`, unchanged.
- **can**: `_scan_can(sysfs_net: Path) -> list[str]` — a CAN interface is a
  netdev with ARPHRD type 280, so listing is a sysfs read: every child of
  `/sys/class/net` whose `type` file reads `280`. Sorted interface names
  (`can_left`, `can0`, ...). No privileges, no subprocess. Missing dir or
  unreadable `type` files → skipped/`[]`. `run_setup` gains
  `sysfs_net: Path = Path("/sys/class/net")`.
- **serial**: `_scan_serial(serial_by_id_dir: Path) -> list[str]` — sorted
  absolute paths under `/dev/serial/by-id`; missing dir → `[]`. `run_setup`
  gains `serial_by_id_dir: Path = Path("/dev/serial/by-id")`.
- **CAN serial numbers** (for §5): `_can_serial(sysfs_net: Path, ifname:
  str) -> str | None` — reads
  `(sysfs_net / ifname / "device").resolve().parent / "serial"`, stripped;
  any failure → `None` (broad except: sysfs layouts vary).

## 3. Slot-driven interview

`run_setup` decides the device section's shape after the defaults prompts:

- The configured embodiment name is registered AND `device_slots(factory)`
  is non-empty → **slot mode**: interview exactly the declared slots, in
  declaration order.
- Otherwise → **fallback mode**: today's hardcoded camera section,
  unchanged (quickstart-before-plugin-install keeps working).

Slot mode reuses the existing role-prompt machinery, generalized:

- One "Configure devices?" yes/no gate (default yes when any probe found
  devices or the existing config assigns any slot arg; default no
  otherwise), mirroring today's camera gate.
- Per kind, on first use, print the listing (`Found N camera device(s)
  under ...` stays; CAN: `Found N CAN interface(s) under /sys/class/net:`;
  serial: `Found N serial device(s) under /dev/serial/by-id:`).
- Each slot prompts with its `label` (not the arg):
  `left arm CAN channel — number, 'u' to identify by unplugging, 's' to
  skip[, 'p' to switch listing]: `. `u` is the same rescan-diff flow for
  every kind ("Unplug the left arm CAN channel now..." wording comes from
  the label; for CAN the operator unplugs the USB-CAN adapter). The `p`
  by-id/by-path toggle exists only for v4l2. Manual entry: absolute path
  for v4l2/serial (advisory existence warning, as today); for CAN a bare
  interface name (no `/`), advisory warning when not in the listing.
- "(current)" Enter-accept defaults from the existing config, per slot arg,
  as today.
- Duplicate-assignment confirm works across all slots of the same kind.
- All-or-none applies per named `group` (the generalized message:
  `"{embodiment} needs all {group} slots or none; writing none unless you
  go back"`). Ungrouped slots are independent.

Managed keys become dynamic: `_render_config` and the decline-preserve path
currently hardcode `camera_keys`; both take the interviewed keys as a
parameter (`managed_args: tuple[str, ...]`). Slot mode passes the declared
slot args; fallback mode passes the camera keys (behavior unchanged).
Declining the section preserves existing assignments for managed args, the
all-or-none "write none" branch drops the group's args, and carried
non-managed `[embodiment.args]` keys pass through raw, all exactly as today.

## 4. What slot mode does NOT do (YAGNI)

- No policy-side slots: only the embodiment is interviewed (policy args are
  checkpoints/URLs, not devices).
- No slot-declared defaults or validators; a slot is (arg, kind, label,
  group) and nothing else until a second plugin needs more.
- No attempt to probe whether a CAN interface is UP (`operstate`): existence
  is the wizard's contract; `doctor`/#50 owns runnability.

## 5. udev pinning suggestion (issue Layer 3, print-only)

After slot-mode assignment, when TWO OR MORE assigned CAN interfaces have
kernel-default order-dependent names (regex `^can\d+$`):

- Read each one's adapter serial via `_can_serial`.
- All readable and pairwise distinct → print (yellow) a warning that
  order-dependent names can swap which physical arm receives commands on
  replug, then the exact rules snippet, one line per assignment, with a
  suggested stable name derived from the slot arg (`left_channel` →
  `can_left`: strip a trailing `_channel`/`_bus`, prefix `can_` unless the
  remainder already starts with `can`):

```
these CAN interfaces have order-dependent names; a replug can swap them.
pin them by adapter serial (paste into /etc/udev/rules.d/70-can-names.rules,
then replug or reboot), and re-run setup to record the pinned names:
  SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="3B004B", NAME="can_left"
  SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="3B004C", NAME="can_right"
```

- Serials unreadable or duplicated (identical adapters without unique
  serials) → print only the swap warning, no snippet.
- Pinned names (anything not matching `^can\d+$`) → nothing printed.

The wizard never writes to `/etc` (sudo); this is guidance text only.

## 6. Docs

- `docs/guide/adapters.md`: extend the plan-0010 "Declare runtime
  requirements" area with a "Declare device slots" section (dataclass
  example above, kinds, group semantics, submodule import).
- `docs/guide/cli.md` setup section: one short paragraph (slots drive the
  interview when declared; CAN listing + unplug-identify; udev suggestion).
- CHANGELOG "Added" entry referencing #61; module map row for
  `conformance.py` extended; `_setup.py` row mentions device slots.

## 7. Tests (100% branch, mypy strict)

`tests/test_conformance.py`:
- `device_slots`: absent → `()`; valid tuple round-trips in order; list
  accepted; non-iterable / entries that are not DeviceSlot / unrecognized
  kind → offending entries ignored (whole-value garbage → `()`); `None`
  factory → `()`.

`tests/test_setup.py` (fake sysfs tree in tmp_path: `net/<if>/type` files
with `280`/`1`, `net/<if>/device` symlink into a `usb/...` dir whose parent
holds `serial`):
- `_scan_can`: filters type==280, sorted, missing dir → `[]`, unreadable
  type file skipped. `_scan_serial`: listing, missing dir → `[]`.
  `_can_serial`: reads through the device symlink; missing serial → None.
- Slot mode activates: registered factory with DEVICE_SLOTS → prompts use
  labels, CAN listing printed, number pick writes `left_channel = can_left`
  into `[embodiment.args]`; fallback mode when unregistered or no
  declaration (existing camera tests keep passing untouched).
- CAN manual entry: bare name accepted with advisory warning when unlisted;
  absolute-path entry rejected vocabulary (re-prompt) since CAN wants a
  name; v4l2/serial keep path entry.
- `u` flow for a CAN slot via injected rescan (shrink/restore lists).
- Group all-or-none: partial `cameras` group → generalized message, both
  branches; ungrouped CAN slots unaffected by the guard.
- Decline preserves existing slot-arg assignments; carried non-managed keys
  survive (managed_args parameterization).
- udev suggestion: two `can\d+` assignments with distinct serials → warning
  + two rule lines with the right serials and derived names; identical
  serials → warning only; pinned names → silence.
- `(current)` default for a slot arg.

`tests/test_registry_cli.py`: none needed (doctor untouched).

## 8. Execution

Single PR on `feat/device-slots`, stacked on `feat/runtime-requirements`
(rebase onto main once #62 merges). Commits: (1) protocol + probes,
(2) slot-driven interview + managed-args generalization, (3) udev
suggestion, (4) docs. Gates per commit; Fable review-edit loop before
merge.
