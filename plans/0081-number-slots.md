# 0081: Numeric option slots for the setup wizard (NUMBER_SLOTS)

Closes #432. Branch: `feat/number-slots`.

## Problem

The wizard's behavior-toggle protocol is boolean-only: `OptionSlot` is "one
boolean behavior toggle the setup wizard interviews" (`conformance.py:75-87`)
and `_options_section` writes only `true`/`false` (`_setup.py:984-1008`).
inspect-robots-yam's thermal guardrail is configured by a float,
`motor_temp_limit` (degrees C, `none` = off, yam#144), which operators should
be able to set during setup exactly like the other safety toggles. There is
no numeric slot protocol to declare it through.

## Design

Additive and deliberately parallel to OPTION_SLOTS, not DEVICE_SLOTS (no
probing, no grouping, no section gate).

**`conformance.py`:** a frozen `NumberSlot` dataclass after `OptionSlot`:

- `arg: str`, `label: str` (same contracts as `OptionSlot`; the unit belongs
  in the label text).
- `default: float | int | None = None` — the suggestion when the key is
  absent from an existing config; `None` is a valid suggestion only when
  `allow_none` is true.
- `minimum: float | None = None`, `maximum: float | None = None` — inclusive
  bounds; `None` means unbounded on that side.
- `allow_none: bool = False` — whether the literal `none` is an accepted
  answer (written verbatim; `_parse_value` hands the constructor `None`).

`number_slots(factory)` mirrors `option_slots` (`conformance.py:90-106`):
defensive getattr, iterable guard, per-entry isinstance filter — plus a
validity filter in the spirit of `device_slots`' `DEVICE_KINDS` check
(`conformance.py:69`): an entry is ignored when a numeric field is a bool or
non-finite, when `minimum > maximum`, when `default` is outside the bounds,
or when `default is None` without `allow_none`. The wizard never crashes on a
malformed declaration; it just does not interview it. This filter is also
what makes the Enter loop safe: `_ask` (`:123-138`) substitutes the displayed
default and re-validates it, so **the displayed suggestion must itself
validate** or Enter hard-loops — the filter guarantees every surviving
slot's default does (and `str(float)`/`str(int)` round-trip through
`_parse_value`).

**`_setup.py`:** a `_numbers_section(numbers, carried, *, input_fn, out)`
mirroring `_options_section` (`:984-1008`):

- One acceptability predicate, written once: a module-level
  `_acceptable_number(slot, parsed) -> bool` — numeric,
  `not isinstance(parsed, bool)`, `math.isfinite`, within the inclusive
  bounds; or `None` when `allow_none`. Both the suggestion rule and the
  prompt validator call it, so the two cannot drift. Because the validator is
  `_parse_value`-based, every None spelling `_parse_value` maps
  (`none`/`null`, any case, `defaults.py:69-70`) is accepted and written
  verbatim; all spellings round-trip to `None`.
- Suggestion: `slot.default`, overridden by a carried `[embodiment.args]`
  value when `_acceptable_number(slot, _parse_value(raw))` — including a
  carried `none` when `allow_none`. An unacceptable carried value (garbage,
  bool, non-finite, out of range) falls back silently, matching
  `_options_section:1001-1005` exactly (the `_prompt_defaults`
  warn-on-invalid asymmetry at `:236-240` is pre-existing and out of scope).
- Prompt: `_ask` (`:123-138`) with the suggestion stringified for display
  (`"none"` when None), a validator built by a **module-level factory**
  `_number_validator(slot) -> _Validator` (a per-slot closure inside the
  interview loop risks ruff B023 and is harder to branch-test), and a
  constraint sentence assembled from the bounds and `allow_none` covering all
  four bound arms (e.g. `minimum=1, allow_none=True` reads
  "motor_temp_limit must be a finite number >= 1, or none").
  `_setup.py` gains an `import math` for the finiteness check
  (the `cli.py:2525-2528` precedent).
- Answer: the validated input string written verbatim (it round-trips through
  `_parse_value` by construction); an accepted empty input writes the
  canonical `str(suggestion)` / `"none"`. Type round-trip note: the written
  form determines the constructor type via `_parse_args_section`
  (`defaults.py:143-155`) — `70` arrives as int, `70.5` as float; consumers
  declare `float | int | None` or coerce, which YamConfig already tolerates
  for its float fields.

**`run_setup` wiring (`:1613-1628`):** numbers are interviewed **after**
options, sharing the same collision bookkeeping: `taken` is seeded from
`managed_args` — device args on the device-slot path, else `CAMERA_KEYS`
(`:1588-1618`), never both — then grows across option args and then number
args, first declaration winning within and across protocols in that order,
so the configparser duplicate-key invariant (`:1613-1616` comment) holds.
Restated at the point of modification: **interviewed number args extend
`managed_args` exactly as option args do at `:1625`, so the interview answer
is what `_render_config:1440-1445` writes and a stale carried duplicate line
is dropped by the `not in managed_args` filter at `:1446-1449`.** Placing
numbers last keeps every existing prompt-index pin green
(`tests/test_setup.py:3954` `len(prompts) == 7`, and all `prompts[7]` option
assertions). The `except (EOFError, KeyboardInterrupt)` at `:1629-1631`
already covers the new prompts; nothing is written on abort.

## Tests (100% line+branch)

`tests/test_conformance.py` — mirror the six option-slot reader tests
(`:119-181`) for `number_slots`: absent attribute / None factory; valid
tuple in declaration order; field defaults; list accepted with offending
entries ignored; whole-value garbage parametrize; raising property and
raising `__iter__`. Plus the validity filter arms: bool field, non-finite
default/bound, `minimum > maximum`, out-of-bounds default, `default=None`
without `allow_none` — each ignored, valid siblings kept.

`tests/test_setup.py` — mirror the option-slot wizard suite (`:3694-3957`)
with module-level fixtures spanning the bound arms — `TEMP_LIMIT =
NumberSlot(arg="motor_temp_limit", ..., default=70, minimum=1,
allow_none=True)` (min-only + none), one slot with both `minimum` and
`maximum`, and one with neither bound and `allow_none=False` — plus a
`_register_*` twin:

- declared number is asked after options and written (`motor_temp_limit = 70`
  on Enter; typed `65` writes `65`);
- suggestion comes from a valid carried value; a typed answer overrides a
  carried value and the key appears exactly once in the written file (the
  `:3792-3816` options idiom);
- a carried `none` is accepted as the suggestion when `allow_none`;
- carried garbage falls back to the declared default silently: `banana`,
  `true` (bool guard), `nan` (finiteness), below `minimum`, above `maximum`;
- `none` accepted when `allow_none`, rejected with the constraint sentence
  when not; below-minimum, above-maximum, and non-numeric answers re-prompt
  with the constraint (count prompts and constraint occurrences, the
  `test_run_setup_reprompts_invalid_typed_values` idiom at `:1859-1880`);
- the constraint sentence renders correctly for all four bound arms
  (min-only, max-only, both, neither) with and without the "or none" clause;
- `None` default displays as `[none]` and Enter writes `none`;
- EOF at the number prompt aborts with "setup aborted; nothing written" and
  no file;
- combined DEVICE_SLOTS + OPTION_SLOTS + NUMBER_SLOTS factory: order is
  devices, options, numbers;
- collision: a number arg colliding with a camera key (camera-mode factory,
  the `:3877-3905` idiom — the camera-key seeding only exists on the
  no-device-slots path), with a device arg, or with an option arg is skipped
  (first declaration wins), including the option-vs-number cross-protocol
  case;
- duplicate number arg: first wins;
- no numbers declared: prompt count and texts unchanged (the existing
  `len(prompts) == 7` pin stays untouched as written).

No pre-existing assertion changes anywhere: numbers interview last, so every
existing prompt index is stable. The `# Input sequence:` comment blocks at
`:4581-4667` stay accurate for the same reason (their factories declare no
number slots).

## Docs

- `docs/guide/adapters.md`: a `## Declare number slots` section after the
  option-slot section (`:97-117`), linking
  `/api/#inspect_robots.conformance.NumberSlot`; regenerate the API reference
  (`uv run python scripts/gen_api_docs.py`) so the anchor exists for the
  strict Docusaurus build. The section must state the version floor:
  importing `NumberSlot` requires the core release that ships it — on an
  older core the ImportError at entry-point load drops the whole plugin
  (`registry.py:110-124`), so plugins adopting NUMBER_SLOTS must raise their
  inspect-robots floor. (This also corrects issue #432's claim that a
  declaration is "silently ignored" on older cores — only the reverse
  direction is free; say so in the PR.) Prose follows the repo style rules
  (root CLAUDE.md: no em dashes, bold only for term lead-ins).
- `README.md:67-70`: "behavior toggles" becomes "behavior toggles and numeric
  settings".
- `src/inspect_robots/CLAUDE.md`: extend the `conformance.py` and `_setup.py`
  module-map rows.
- `CHANGELOG.md` `## [Unreleased]` / `### Added`, current style:
  `- **Setup wizard:** ... ([plan 0081](plans/0081-number-slots.md),
  [#432](https://github.com/robocurve/inspect-robots/issues/432))` — the
  "Setup wizard" scope lead-in is new (existing entries use "Core"/"Docs")
  and deliberate.

## Out of scope

- No `__all__` / `test_api_snapshot.py` change: `NumberSlot` is imported from
  `inspect_robots.conformance` like `OptionSlot` (plan 0032 precedent).
- No runtime consumer (device slots have `claim_devices`; numbers need none).
- No section-level "Configure numbers?" gate (options have none either).
- No integer-only slot kind; `minimum`/`maximum` plus the consumer's own
  validation cover current needs.
- The `_prompt_defaults` vs `_options_section` warn/silent asymmetry stays
  as-is.
- The yam-side `NUMBER_SLOTS` declaration: separate PR in inspect-robots-yam
  after this ships in a core release (it needs the new dependency floor).
