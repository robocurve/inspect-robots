# 0034 — Embodiment-contributed guardrails

Issue: #232. Sibling: inspect-robots-yam#93 / its plan 0017.

## Problem

The default CLI safety chain (`_build_guardrails`, plan 0008 §3e) is fixed:
bounds clamp + per-step delta limit. Both are generic — they know the action
space, not the robot. Embodiment plugins can ship materially stronger,
embodiment-specific approvers: inspect-robots-yam's MuJoCo collision guardrail
(yam #85/#86) predicts bimanual self-, cross-arm, and table collisions and
holds at the last safe pose — complete, tested, and released. But no run path
can activate it:

- `_build_guardrails` builds only the two generic approvers; nothing consults
  the embodiment.
- The agent plugin's `pre_check` hook (#210) is programmatic-only by design
  ("-P CLI flags cannot carry callables") — and it protects only the agent
  policy, not scripted or learned policies, and sits *above* the approver
  layer rather than below it.

So a bimanual rig runs `clamp + delta-limit` while a purpose-built collision
guardrail sits unreachable in the installed wheel. The gap is a missing seam,
not a missing feature.

## Design

One optional, duck-typed embodiment method, consulted by `_build_guardrails`,
appending embodiment-contributed approvers to the default chain. Contributions
are **additive only**: the generic clamp + delta-limit stay exactly as they
are, in front, so the CLI is never less protective with a contribution than
without one. `--disable-guardrails` continues to disable the entire chain,
contributions included.

### 1. `GuardrailContribution` (approver.py)

```python
@dataclass(frozen=True)
class GuardrailContribution:
    """Approvers an embodiment adds to the CLI's default guardrail chain.

    ``approvers`` pairs a short display name (shown in the ``guardrails:``
    banner, e.g. ``"yam-collision"``) with the approver to append.
    ``warnings`` name contributions the embodiment declined to make or made
    in a degraded state, and why (mode unsupported, optional dependency
    missing, unmeasured configuration) — printed like the builder's own skip
    warnings, so neither a declined nor a degraded contribution is ever
    invisible. Warnings may accompany active approvers.
    """

    approvers: tuple[tuple[str, Approver], ...] = ()
    warnings: tuple[str, ...] = ()
```

Validation in `__post_init__`: display names must be non-empty and contain no
newline (they land in a one-line banner); approvers must satisfy the
`Approver` protocol shape (a callable `review` attribute). Exported from
`inspect_robots.approver`.

The docstring also states the **behavioral contract** a contributed approver
must honor as a chain member (today implicit in core's internals):

- *Identity preservation*: return the incoming `Action` object itself when
  approving unmodified — rollout detects modification by identity
  (`ChainApprover` docstring); a fresh equal-valued object corrupts approval
  events.
- *Reference rewind on substitution*: an approver that substitutes a
  different target (hold semantics) must rewind the delta limiter's reference
  via `DeltaLimitApprover.rewind_reference` (below), or the limiter keeps
  rating subsequent deltas against a target that never executed — permitting
  a single-step jump from the real pose far larger than `max_delta`.
- *Veto = `SafetyAbort`*: rejecting outright means raising `SafetyAbort`;
  any other exception is treated as a bug and propagates.
- *Validate your own input*: a contribution must reject non-finite values
  itself (`SafetyAbort`) and must not assume upstream clamping occurred —
  on a degraded chain (unbounded space, both generic gates skipped) the
  contribution is the first and only gate between raw policy output and
  hardware.

### 1b. Public store seam: `DeltaLimitApprover.rewind_reference`

The delta limiter's store key is module-private (`_LAST_APPROVED_KEY`), and
the first real contribution (yam's `CollisionApprover`) currently duplicates
the string literal to rewind it — a silent-desync hazard: if core renames the
key, the rewind becomes a no-op with no warning and the jump hazard above
goes live. Since this plan creates the seam that invites third-party
approvers behind the limiter, it also makes the interaction point public:

```python
@staticmethod
def rewind_reference(store: dict[str, Any], pose: npt.NDArray[np.float64]) -> None:
    """Reset the limiter's reference to ``pose`` if a reference exists."""
```

documented for exactly this substitution case — it **stores a copy** of the
pose (callers must not end up aliasing the limiter's reference to a live
array) — with a test pinning the store-key/helper coupling so a rename fails
loudly. Yam plan 0017 switches `CollisionApprover` to call it.

### 2. The embodiment method (documented protocol, duck-typed)

```python
def contribute_guardrails(self, action_space: Box) -> GuardrailContribution: ...
```

- Optional: absence means no contribution (every existing embodiment keeps
  working, zero migration).
- Instance method, not ClassVar: whether and how to contribute depends on the
  embodiment's runtime config (control interface, user opt-out flag).
- Follows the `bind_task` precedent (`embodiment.py`): documented in the
  `Embodiment` Protocol docstring, deliberately **not** a Protocol member,
  so existing embodiments remain `isinstance(x, Embodiment)`-conformant
  under `runtime_checkable`. Completing the precedent, `EmbodimentBase`
  gains a default `contribute_guardrails` returning an empty
  `GuardrailContribution()` (bind_task has a no-op base default too);
  subclasses override to contribute.
- Conformance: `check_embodiment` stays purely declarative and untouched (it
  receives an `EmbodimentInfo`, cannot see instance methods, and must never
  execute plugin code — `doctor` runs it against every installed adapter). A
  new opt-in `check_guardrail_contribution(embodiment, action_space)` joins
  `conformance.py` for adapter CI to call with an instance it constructed:
  documented as allowed to execute plugin code, it checks the attribute (if
  present) is callable and the result is a valid `GuardrailContribution`. It
  returns a `ConformanceReport` in the `check_embodiment` style, with an
  `assert_guardrail_contribution_conformant` raising wrapper mirroring the
  existing pair — so both CI idioms keep working.
- The contract mirrors `_build_guardrails`' own degrade philosophy: an
  embodiment that cannot contribute in the current mode returns a warning
  entry, never raises for "not applicable". Raising is reserved for actual
  bugs and propagates — a safety component that fails unexpectedly must be
  loud, not silently skipped.

### 3. `_build_guardrails` (cli.py)

Signature gains the embodiment:

```python
def _build_guardrails(
    space: Box, max_action_delta: float | None, embodiment: Embodiment | None = None
) -> tuple[Approver, list[str], list[str]]:
```

After the existing clamp/delta assembly:

- `hook = getattr(embodiment, "contribute_guardrails", None)`; if absent →
  done (exact current behavior).
- If present but not callable, or if calling it returns anything that is not
  a `GuardrailContribution` → **hard error** (`SystemExit` naming the
  embodiment and the actual type). There is no legitimate version skew that
  produces this: a plugin new enough to define the hook imports
  `GuardrailContribution` from the installed core (same class object), and a
  core too old to have the class never calls the hook. The branch catches
  exactly plugin bugs, and a bug in a safety component halts the run — the
  same stance as §2's "raising propagates". Degrading here would let a
  default-on guardrail vanish behind a stderr warning.
- Otherwise append each contributed approver to `parts`, its display name to
  `active`, and extend `warnings` with the contribution's warnings.

`_build_and_announce_guardrails` passes `resolved.embodiment` through from
both call sites (single-task and eval-set paths); the banner then reads e.g.
`guardrails: clamp + delta-limit + yam-collision`. The `AutoApprover`
fallback branch ("no guardrails are active…") is unreachable when a
contribution exists, and the fallback message stays accurate: contributions
append to `parts`, so `parts` non-empty ⇒ chain built.

`--disable-guardrails` already returns before `_build_guardrails` is invoked;
contributions are therefore disabled with everything else, and the
`--max-action-delta`/`--disable-guardrails` conflict check is untouched.

### 4. Ordering

Chain order is `clamp → delta-limit → contributions…` (declaration order
within a contribution). When the generic gates are active, contributed
approvers see targets already clamped and rate-limited — cheap generic gates
first, expensive embodiment-specific ones last. On a degraded chain (both
generic gates skipped for an unbounded space) the contribution sees raw
policy output, which is why the §1 contract requires contributions to
validate their own input.

## Not in scope

- Contributions replacing or reordering the generic chain (additive only).
- A CLI flag to disable a single contribution (opt-out belongs to the
  embodiment's own config, where the wizard can interview it — see yam plan
  0017).
- Plumbing contributions into programmatic `eval()` callers: `eval()` already
  accepts any `approver`; callers compose their own chain.

## Tests

In `tests/test_registry_cli.py`, where the existing `_build_guardrails`
coverage lives, plus `tests/test_approvers.py` for the store seam:

- No method → identical chain and banner to today (regression).
- Contribution with one approver → appended after delta-limit, name in
  `active`, banner string exact.
- Two contributed approvers → declaration order preserved in both chain and
  banner; a contribution with both `approvers` and `warnings` non-empty
  surfaces both.
- Contribution with warnings only → warnings printed, chain unchanged.
- Non-callable attribute / wrong return type → `SystemExit` naming the
  embodiment and type.
- Contribution while clamp+delta are both skipped (unbounded space) → chain is
  contribution-only, `AutoApprover` fallback not taken.
- `--disable-guardrails` with a contributing embodiment → no approver, no
  contribution call (method not invoked; assert via spy).
- `rewind_reference`: rewinds an existing reference; no-op without one;
  stores a copy (mutating the caller's array afterward must not move the
  reference); a test pinning it to the limiter's actual store key (rename
  fails loudly); substitution scenario — approver behind the limiter holds
  an earlier pose, rewinds, next delta is rated against the held pose.
- `EmbodimentBase` default returns an empty `GuardrailContribution()`; the
  CLI treats it identically to an absent attribute (banner unchanged).
- Conformance: `check_guardrail_contribution` passes an absent attribute,
  fails a non-callable one and a wrong return type, and the
  `assert_guardrail_contribution_conformant` wrapper raises on failure;
  `check_embodiment` behavior unchanged.

A `ValueError` escaping a contribution (e.g. a mis-measured rig model
rejected at approver construction) intentionally surfaces as a traceback
rather than the friendly `SystemExit` used for embodiment construction
errors: it is a bug-loudness path, not an operator-input path.

## Release

Minor bump (new public API `GuardrailContribution`): core `0.31.0`. The yam
plugin's floor bumps to `>=0.31` in its own PR (yam #93), which imports the
dataclass.
