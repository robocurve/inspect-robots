# 0029 — public defaults API: let plugin CLIs read the wizard config

Issue: #197. Status: revised after critique rounds 1 and 2 (round 1: re-export
façade replaced by a module promotion — the docs generator cannot document
aliases; round 2: call-site inventory completed, docs steps made concrete).

## Problem

`inspect-robots setup` interviews the user and writes
`~/.config/inspect-robots/config.ini`: `[defaults]` (policy, embodiment,
scorer, max_steps, rerun, store_frames) plus `[embodiment.args]` — on a real
rig, the camera device paths and CAN channels that make anything work. The
run CLI consumes that file through `inspect_robots/_defaults.py`, an
underscore-private module.

Plugins ship their own maintenance CLIs. `inspect-robots-yam` has three
(`-health`, `-holdcheck`, `-preflight`), and each builds its config purely
from its own flags and `-E` extras, because the only reader of the wizard's
file is private to the core. The user-visible result, on a rig the wizard
configured minutes earlier:

```
$ inspect-robots-yam-health --watch
inspect-robots-yam-health: error: --watch requires configured camera devices
```

The wizard knows all three camera paths, wrote them to disk, and even prints
advice about other tools — but a plugin CLI cannot legitimately read the file
back. Copy-pasting the configparser logic into each plugin duplicates path
resolution, value parsing, `~` expansion, and the args-owner rule, and drifts
the first time the core changes any of them.

Scope limit: this plan adds the public read surface in the core and nothing
else. The yam CLIs adopting it is robocurve/inspect-robots-yam#80, planned
and reviewed in that repo. No behavior of `inspect-robots run` / `setup` /
`config` changes here.

## Design

Promote the module: `_defaults.py` is renamed to `defaults.py`, and its
public surface is pinned to exactly three names. No façade, no aliases — the
API-docs generator (`scripts/gen_api_docs.py`) documents real `Object`
members only (Griffe `Alias`es are filtered out by `_public_members` and an
all-alias module would make it raise), so the public objects must be *owned*
by the public module.

```python
from inspect_robots.defaults import Defaults, config_path, load_defaults

defaults = load_defaults(os.environ)
if defaults.embodiment_args_owner is not None:
    args = defaults.embodiment_args  # valid only for that owner — see below
```

### Public surface (`__all__`, enforced by test)

- `Defaults` — the frozen dataclass exactly as it exists today, including
  `embodiment_args` and the three `*_args_owner` fields. Owners are the
  load-bearing part of the contract for plugins: an `[embodiment.args]`
  section is only valid for the embodiment named in `[defaults]`, and a
  consumer that applies args without checking the owner will aim someone
  else's cameras.
- `load_defaults(env: Mapping[str, str]) -> Defaults` — semantics unchanged
  and now documented as stable: `env` is injected (pass `os.environ`);
  env vars override the component *names* but never the args owners; a
  missing file yields empty `Defaults()`; a malformed or type-invalid file
  raises `SystemExit` naming the file. The docstring states explicitly that
  the `SystemExit` carries a plain one-line message and is catchable — a
  diagnostic CLI that prefers to report "config malformed" as a finding and
  continue its other checks may catch it. No separate `try_load` variant:
  one reader, one error behavior, and `except SystemExit` at the one call
  site a consumer has is not enough ceremony to justify a second entry
  point.
- `config_path(env: Mapping[str, str]) -> Path | None` — the full path of
  the config *file* (`<config-home>/inspect-robots/config.ini`, where
  config-home is `$XDG_CONFIG_HOME` else `$HOME/.config`), or `None` when
  neither variable is set. Returned whether or not the file exists.
  Consumers need this for source attribution: a diagnostic CLI that
  silently absorbs devices from a file must be able to print *which* file.

### The rename, precisely

- `git mv src/inspect_robots/_defaults.py src/inspect_robots/defaults.py`.
- `_config_path` → `config_path` (public). Call sites: `load_defaults` and
  `set_default` in the module itself, **and `_setup.py:14`**, which imports
  `_config_path` by name and calls it in `run_setup` — missing that import
  breaks `inspect-robots setup` with an ImportError.
- Names that stay internal get underscore prefixes so the public surface is
  visible in the module itself and the docs generator can never pick them
  up regardless of how it filters: `parse_value` → `_parse_value`,
  `set_default` → `_set_default`, `CONFIG_KEYS` → `_CONFIG_KEYS`,
  `ENV_POLICY`/`ENV_EMBODIMENT`/`ENV_SIM_EMBODIMENT` →
  `_ENV_POLICY`/`_ENV_EMBODIMENT`/`_ENV_SIM_EMBODIMENT`, plus the constants
  `ADHOC_SCORER_FALLBACK` → `_ADHOC_SCORER_FALLBACK` and
  `ADHOC_MAX_STEPS_FALLBACK` → `_ADHOC_MAX_STEPS_FALLBACK`. Call sites to
  update — the complete inventory (re-verify with
  `grep -rn "_defaults" src/ tests/ plugins/` before committing):
  - `cli.py:47-57` — imports `parse_value`, `set_default`, `CONFIG_KEYS`,
    and the ENV/ADHOC constants (the only importer of most of them).
  - `_setup.py:14` — imports exactly `_config_path` and `parse_value`
    (bare `parse_value` calls at `_setup.py:104,109`); nothing else from
    the module.
  - `tests/test_defaults.py:10-18` — imports `ENV_*`, `Defaults`,
    `load_defaults`, `parse_value`, `set_default` directly.
  - `tests/test_registry_cli.py:15` — imports `ENV_EMBODIMENT`,
    `ENV_POLICY`, `ENV_SIM_EMBODIMENT`; the constants are used at 30+
    sites in that file (the imports change, the uses don't).
  - `tests/test_coverage_completion.py:20` — imports `parse_value`
    (aliased locally), exercised by two tests.
  In-tree `plugins/` reference none of the renamed names.
- Rationale for not exporting these: `_parse_value` is the core CLI's `-P`/
  `-E` coercion — plugins have their own rules (yam keeps camera device
  values as raw strings precisely to defeat scalar coercion), and exporting
  it invites plugins to adopt semantics the core may tune. `_set_default` /
  `_CONFIG_KEYS` keep writing core-only: the wizard and `inspect-robots
  config set` own the file's shape. The env-var constants configure
  component *selection* for the run CLI; plugins reading them directly
  would re-implement precedence `load_defaults` already applies.
- No `_defaults.py` shim is left behind. Nothing supported imports it —
  its docstring's own privacy claim is the reason this plan exists — and a
  shim would keep the temptation alive.
- Considered and rejected: an owner-gated accessor on `Defaults` (e.g.
  `embodiment_args_for(owner)` returning `{}` on mismatch). The owner rule
  is consumer-specific — yam accepts any embodiment whose *class* resolves
  to a `YAMEmbodiment` subclass through the registry (`yam_arms_omen` lives
  in a third package), which the core cannot know by name. A name-keyed
  accessor would fit no real consumer while implying it is the sanctioned
  check. The fields plus documentation are the honest interface; yam#80
  makes the class-based gate a review item on its side.

### Package contract

`src/inspect_robots/__init__.py:8-9` declares the public API to be exactly
what `__init__.py` exports via `__all__`. A public submodule must therefore
be exported there too: add `from inspect_robots import defaults` and
`"defaults"` to the package `__all__`, and extend the docstring's contract
sentence to mention public submodules. Import-cost note, stated honestly:
`defaults.py` itself imports stdlib only, but *any* `inspect_robots.*`
import already executes `__init__.py`, which pulls numpy via `types`/
`rollout`/etc. — the change adds nothing measurable on top, and plugin CLIs
import the package anyway. The plan claims no import-cost property beyond
"no new heavy imports".

### Versioning and docs

- Minor version bump on release (new public API): next tag `v0.24.0`
  (latest is `v0.23.3`; hatch-vcs derives the version from the tag).
- `CHANGELOG.md`: add an `[Unreleased]` → Added entry for
  `inspect_robots.defaults` (Keep-a-Changelog format, as the file already
  uses).
- API reference, concretely: add `"inspect_robots.defaults"` to `_SECTIONS`
  in `scripts/gen_api_docs.py` under a new **"User configuration"** heading
  (none of the existing eleven section headings fits; a wrong-section
  placement is a review nit we can decide once, here). The generator emits
  a *single* page (`docs/api/index.md`) — there are no per-module pages and
  therefore **no `website/sidebars.ts` edit**; its lone `api/index` entry
  already covers the new section. Regenerate `docs/api/` and note the
  `website/scripts/check-api-docs.mjs` check is existence-only — it cannot
  catch a stale page, so regeneration is on the implementer. Because the
  module's members are real objects (not aliases), `_public_members` finds
  them; the generator crash mode that killed the façade design does not
  apply.
- The **module docstring is rendered onto the API page** (`_render`,
  `gen_api_docs.py:133-135`) and must be rewritten for external readers:
  the current one cites "(plan 0005)" and "handled in ``cli.py``", and its
  privacy claim becomes false the moment this plan lands. New docstring:
  what the file is, the precedence contract (env names over file, owners
  never overridden), the owner rule with the one-line example, the
  `SystemExit` behavior.
- `src/inspect_robots/CLAUDE.md:33` — the module-map row naming
  `_defaults.py` and `set_default` is updated for the rename (repo
  convention: CLAUDE.md files stay current).
- Plugin-authoring docs: `docs/guide/plugins.md` ("Shipping an out-of-tree
  plugin") gains a short subsection pointing at `inspect_robots.defaults`
  with the owner-check example. (The README has no plugin-authoring
  section; it is not touched.)

## Tests

- Import updates in three test files, no assertion logic changes:
  `tests/test_defaults.py` (imports `ENV_*`, `parse_value`, `set_default`
  directly), `tests/test_registry_cli.py:15` (the three ENV constants),
  and `tests/test_coverage_completion.py:20` (`parse_value`). There is no
  existing test that imports `_config_path` —
  the `_config_path` in `tests/test_setup.py:80` is an unrelated local
  helper. `config_path` behavior is currently covered only indirectly
  through `load_defaults`.
- New `tests/test_defaults_public.py`:
  - `inspect_robots.defaults.__all__` is exactly
    `["Defaults", "config_path", "load_defaults"]` and every name resolves.
  - `"defaults"` is in `inspect_robots.__all__` and
    `inspect_robots.defaults` is the module.
  - Direct `config_path` coverage, written fresh: XDG path wins over HOME,
    HOME fallback appends `.config`, empty env returns `None`, and the
    returned path ends with `inspect-robots/config.ini` whether or not the
    file exists.
  - End-to-end smoke: write a config file under a tmp `XDG_CONFIG_HOME`,
    `load_defaults` through the public module, assert `embodiment_args`
    and `embodiment_args_owner` round-trip.
- Coverage stays at the repo's enforced 100% (`fail_under = 100`): the
  renames move covered code without adding branches, and the new tests
  cover the `__init__` export line.

## Risks

- **Contract freeze.** Publishing `Defaults` freezes its field names.
  Accepted: the fields mirror the config file format, which is already a
  user-facing contract via `setup` and the docs; renaming a field was
  already a breaking change to every written config.
- **Plugins misusing owners.** The API cannot force a consumer to check
  `embodiment_args_owner`; the docstring and the plugins-guide example lead
  with it, and the consuming plan (yam#80) makes it a review gate. The
  accessor alternative was considered and rejected above.
- **Rename fallout.** The module is renamed wholesale; any unsupported
  external import of `inspect_robots._defaults` breaks loudly (ImportError
  naming the module), which is the correct failure for an import that was
  never supported. In-repo call sites are enumerated above and verified by
  grep during implementation; CI's import of every module plus 100%
  coverage catches a missed one.
- **Import cycles.** None: `defaults.py` imports stdlib only.
