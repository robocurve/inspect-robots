# 0006 — `--sim`: swap the default embodiment for its sim counterpart

## 1. Goal

Plan 0005 gave us `inspect-robots "place the spoon on the plate"` running on
the user's configured default embodiment — typically real hardware. This plan
adds the one-flag escape hatch to run the same command in simulation:

```bash
inspect-robots "place the spoon on the plate" --sim
```

Real hardware stays the default (it is whatever the user configured as
`embodiment`); `--sim` swaps in a second, separately configured embodiment.
The framework cannot guess that `yam-bimanual` maps to
`yam-bimanual-isaacsim` — per 0005's no-magic stance, the mapping is explicit
user configuration:

```ini
[defaults]
embodiment = yam-bimanual            ; the default: real hardware
sim_embodiment = yam-bimanual-isaac  ; what --sim swaps in

[sim_embodiment.args]                ; default -E pairs for the sim embodiment
headless = true
```

Non-goals (YAGNI): no auto-discovery of sim counterparts, no per-task sim
mapping, no `--real` flag (real is the absence of `--sim`), no change to the
ad-hoc default scorer under `--sim` (a sim may have a success oracle, but
which scorer that is stays the user's choice — set `scorer =` in config or
pass `--scorer`).

## 2. Grounding (state after plan 0005)

- `_defaults.py`: `Defaults` frozen dataclass with `policy`/`embodiment`
  (+ `_source` strings), `scorer`, `max_steps`, `policy_args`,
  `embodiment_args`; `load_defaults(env)` reads
  `<config-home>/inspect-robots/config.ini` then applies
  `INSPECT_ROBOTS_POLICY`/`INSPECT_ROBOTS_EMBODIMENT` env overrides;
  value/type validation raises `SystemExit` naming the file/key.
- `cli.py`: `_cmd_run` validates mode flags, resolves names via
  `_pick_component(kind, flag, default, source)` (guidance `SystemExit` when
  nothing is configured), merges `defaults.embodiment_args` under explicit
  `-E` pairs, prints the resolved header before the embodiment moves.
- Tests: `tests/test_defaults.py` (pure `load_defaults`),
  `tests/test_registry_cli.py` (CLI e2e on the mock, hermetic
  `_hermetic_defaults` autouse fixture).

## 3. Design

### 3.1 Semantics

- `--sim` is a `run` flag (`store_true`). It applies to **both** `--task` and
  `--instruction` runs — it only changes which embodiment default is used, and
  is orthogonal to task selection. The bare-instruction sugar already passes
  trailing flags through (`inspect-robots "wipe table" --sim` →
  `run --instruction "wipe table" --sim`).
- Resolution for the embodiment name becomes:
  - without `--sim`: `--embodiment` flag > `$INSPECT_ROBOTS_EMBODIMENT` >
    config `embodiment` (unchanged from 0005)
  - with `--sim`: `$INSPECT_ROBOTS_SIM_EMBODIMENT` > config `sim_embodiment`
- `--sim` **conflicts with an explicit `--embodiment`**: `SystemExit` ("--sim
  selects your configured sim_embodiment; passing --embodiment already picks
  the embodiment — drop one"). Silently letting the flag win would make `--sim`
  a no-op lie in scripts.
- `--sim` with `$INSPECT_ROBOTS_EMBODIMENT` set is **not** an error: the sim
  chain simply doesn't consult that variable. This asymmetry with the
  `--embodiment` conflict is deliberate: the env var is a persistent default
  (a shell profile), not a per-invocation intent — erroring would make
  `--sim` unusable for anyone who exports it. Pinned by a test (env real
  embodiment set + `--sim` → sim embodiment used, no error) and documented.
- `--sim` with nothing configured → guidance `SystemExit` mirroring
  `_pick_component`'s message: list registered embodiments, show the env var
  and the `sim_embodiment = NAME` config line.
- Embodiment args under `--sim` come from `[sim_embodiment.args]` (NOT
  `[embodiment.args]` — args tuned for a real rig, e.g. a serial port, are
  wrong for a sim and vice versa). Explicit `-E k=v` still overrides.
- The run header keeps its contract: `embodiment: yam-bimanual-isaac (--sim,
  from <config path>)` or `(--sim, from $INSPECT_ROBOTS_SIM_EMBODIMENT)`.
- The operator-prompt gating is unchanged (a TTY ad-hoc run with the
  `operator` scorer still prompts under `--sim`; skip or configure a scorer).

### 3.2 Changes

`_defaults.py`:

- `ENV_SIM_EMBODIMENT = "INSPECT_ROBOTS_SIM_EMBODIMENT"`.
- `Defaults` grows `sim_embodiment`, `sim_embodiment_source`,
  `sim_embodiment_args` (same shapes as their non-sim twins).
- `_read_config` reads `[defaults] sim_embodiment` and
  `[sim_embodiment.args]`; `load_defaults` applies the env override with
  source `$INSPECT_ROBOTS_SIM_EMBODIMENT`.

`cli.py`:

- `build_parser`: add `--sim` to `run`.
- `_cmd_run`: the `--sim`/`--embodiment` conflict check joins the existing
  mode validation block; embodiment name/args selection branches on
  `args.sim` as specced above. The sim chain gets its own small
  `_pick_sim_embodiment(defaults)` helper rather than routing through
  `_pick_component` (whose reuse would be partial anyway: its guidance
  message and `f"from {source}"` format are wrong for sim): the helper
  returns `(name, f"--sim, from {source}")` on success — satisfying the
  header contract — and on the empty path raises the sim-flavored guidance
  `SystemExit` (registered embodiments + `$INSPECT_ROBOTS_SIM_EMBODIMENT` +
  the `sim_embodiment = NAME` config line). `_pick_component` stays untouched.

Docs: the config example + a `--sim` paragraph in `docs/guide/cli.md`'s
zero-config section; one line in the README block; module-map row already
covers `_defaults.py` (update its blurb to mention sim).

### 3.3 Alternatives considered

- **`--embodiment` + `--sim` = flag wins silently**: rejected — `--sim`
  becoming a no-op depending on other flags is exactly the silent-ignore
  pattern 0005 rejected for `-T`/`--max-steps`.
- **Fallback of `[embodiment.args]` onto the sim embodiment**: rejected —
  real-rig args (ports, camera serials) are wrong for a sim; empty-by-default
  is safer than wrong-by-default.
- **`--sim` flips the ad-hoc scorer to `success_at_end`**: rejected — whether
  the sim has a success oracle depends on the embodiment; guessing wrong
  silently reports 0.0 forever. Config's `scorer =` already covers it.

## 4. Tests (TDD; 100 % coverage; no vacuous tests)

`test_defaults.py`:

- config `sim_embodiment` + `[sim_embodiment.args]` parse (with `~` expansion
  and source path), and are independent of `embodiment`/`[embodiment.args]`
  (full-equality assertion on `Defaults`).
- `$INSPECT_ROBOTS_SIM_EMBODIMENT` overrides config `sim_embodiment` and sets
  the env source; does not touch `embodiment`.

`test_registry_cli.py` (e2e on the mock; `cubepick` plays the "sim").
**Prerequisite: extend the `_hermetic_defaults` autouse fixture to also
`delenv` `ENV_SIM_EMBODIMENT`** — no failing test forces this, but without it
a developer with the var exported gets flaky CLI tests, defeating the
fixture's purpose. Then:

- `--sim` swaps the embodiment: config `embodiment = missing-real-arm`,
  `sim_embodiment = cubepick` → bare instruction + `--sim` runs green
  (proving the real-arm default was NOT used), header shows
  `(--sim, from <config>)`; the same command **without** `--sim` exits with
  the unknown-embodiment `SystemExit` (proving `--sim` was load-bearing).
- `[sim_embodiment.args]` reach the constructor and `-E` overrides them.
  NOTE: the log's `embodiment_info` records only
  `control_hz`/`is_simulated`/`capabilities` (eval.py), none of which depend
  on `CubePickEmbodiment` constructor args — so assert **behaviorally**:
  `[sim_embodiment.args] max_step = 0.001` makes the run too slow to succeed
  (`success_at_end == 0.0`), and an explicit `-E max_step=0.1` restores
  success (`== 1.0`). Non-leakage: put a key in `[embodiment.args]` that
  `CubePickEmbodiment.__init__` rejects (`port = 1`) — the sim run still
  constructs fine, proving real-rig args never touch the sim path.
- `--sim --embodiment cubepick` → conflict `SystemExit` (match on "drop one").
- `--sim` with no `sim_embodiment` configured → guidance `SystemExit` naming
  `INSPECT_ROBOTS_SIM_EMBODIMENT` and `sim_embodiment`.
- `--sim` works with `--task` too (registered task + `--sim` resolves the sim
  embodiment).
- env `$INSPECT_ROBOTS_SIM_EMBODIMENT` beats config `sim_embodiment` at the
  CLI layer (header source assertion).
- asymmetry pin: `$INSPECT_ROBOTS_EMBODIMENT` set (to a bogus name) + `--sim`
  → the sim embodiment runs, no error and no use of the bogus real default.

Post-implementation: subagent mutation audit (drop the `--sim` branch, swap
sim/real chains, leak `[embodiment.args]` into sim runs — each must be killed
by a test) before the PR.

## 5. Milestones

1. `_defaults.py` sim fields + tests.
2. `cli.py` `--sim` + tests.
3. Docs + README — including reordering the README quickstart simple→complex
   per user request: zero-config one-liner first, then the CLI block, then
   the CubePick Python example.
