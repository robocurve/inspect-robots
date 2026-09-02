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
malformed declaration; it just does not interview it.

**`_setup.py`:** a `_numbers_section(numbers, carried, *, input_fn, out)`
mirroring `_options_section` (`:984-1008`):

- Suggestion: `slot.default`, overridden by a carried
  `[embodiment.args]` value when `_parse_value` of it is acceptable for the
  slot — numeric, `not isinstance(parsed, bool)`, `math.isfinite`, within
  bounds; or `None` when `allow_none`. An unacceptable carried value falls
  back silently, matching `_options_section:1001-1005` exactly (the
  `_prompt_defaults` warn-on-invalid asymmetry at `:236-240` is pre-existing
  and out of scope).
- Prompt: `_ask` (`:123-139`) with the suggestion stringified for display
  (`"none"` when None), a per-slot closure validator, and a constraint
  sentence assembled from the bounds and `allow_none` (e.g.
  "motor_temp_limit must be a finite number > 0, or none"). The validator
  accepts exactly what the suggestion rule accepts; `_setup.py` gains an
  `import math` for the finiteness check (the `cli.py:2525-2528` precedent).
- Answer: the validated input string written verbatim (it round-trips through
  `_parse_value` by construction); an accepted empty input writes the
  canonical `str(suggestion)` / `"none"`. Type round-trip note: the written
  form determines the constructor type via `_parse_args_section`
  (`defaults.py:143-155`) — `70` arrives as int, `70.5` as float; consumers
  declare `float | int | None` or coerce, which YamConfig already tolerates
  for its float fields.

**`run_setup` wiring (`:1613-1628`):** numbers are interviewed **after**
options, sharing the same collision bookkeeping: `taken` grows across device
args, camera keys, option args, then number args, first declaration winning
within and across protocols in that order, so the configparser
duplicate-key invariant (`:1613-1616` comment) holds. Placing numbers last
keeps every existing prompt-index pin green (`tests/test_setup.py:3954`
`len(prompts) == 7`, and all `prompts[7]` option assertions). The
`except (EOFError, KeyboardInterrupt)` at `:1629-1631` already covers the
new prompts; nothing is written on abort.

## Tests (100% line+branch)

`tests/test_conformance.py` — mirror the five option-slot reader tests
(`:119-181`) for `number_slots`: absent attribute / None factory; valid
tuple in declaration order; field defaults; list accepted with offending
entries ignored; whole-value garbage parametrize; raising property and
raising `__iter__`. Plus the validity filter arms: bool field, non-finite
default/bound, `minimum > maximum`, out-of-bounds default, `default=None`
without `allow_none` — each ignored, valid siblings kept.

`tests/test_setup.py` — mirror the option-slot wizard suite (`:3694-3957`)
with a module-level `TEMP_LIMIT = NumberSlot(arg="motor_temp_limit", ...,
default=70, minimum=1, allow_none=True)` fixture and `_register_*` twin:

- declared number is asked after options and written (`motor_temp_limit = 70`
  on Enter; typed `65` writes `65`, appearing exactly once);
- suggestion comes from a valid carried value; carried garbage (`banana`,
  `true`, `nan`, out-of-range) falls back to the declared default silently;
- `none` accepted when `allow_none`, rejected with the constraint sentence
  when not; below-minimum and non-numeric answers re-prompt with the
  constraint (count prompts and constraint occurrences, the
  `test_run_setup_reprompts_invalid_typed_values` idiom at `:1859-1880`);
- `None` default displays as `[none]` and Enter writes `none`;
- EOF at the number prompt aborts with "setup aborted; nothing written" and
  no file;
- combined DEVICE_SLOTS + OPTION_SLOTS + NUMBER_SLOTS factory: order is
  devices, options, numbers;
- collision: a number arg colliding with a camera key, a device arg, or an
  option arg is skipped (first declaration wins), including the
  option-vs-number cross-protocol case;
- duplicate number arg: first wins;
- no numbers declared: prompt count and texts unchanged (the existing
  `len(prompts) == 7` pin stays untouched as written).

No pre-existing assertion changes anywhere: numbers interview last, so every
existing prompt index is stable. The `# Input sequence:` comment blocks at
`:4581-4667` stay accurate for the same reason (their factories declare no
number slots).

## Docs

- `docs/guide/adapters.md`: a `## Declare number slots` section after the
  option-slot section (`:98-115`), linking
  `/api/#inspect_robots.conformance.NumberSlot`; regenerate the API reference
  (`uv run python scripts/gen_api_docs.py`) so the anchor exists for the
  strict Docusaurus build.
- `README.md:67-70`: "behavior toggles" becomes "behavior toggles and numeric
  settings".
- `src/inspect_robots/CLAUDE.md`: extend the `conformance.py` and `_setup.py`
  module-map rows.
- `CHANGELOG.md` `## [Unreleased]` / `### Added`, current style:
  `- **Setup wizard:** ... ([plan 0081](plans/0081-number-slots.md),
  [#432](https://github.com/robocurve/inspect-robots/issues/432))`.

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
