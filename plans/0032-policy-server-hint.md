# 0032 — Actionable hint when a policy's action server is unreachable

Closes robocurve/inspect-robots#219.

## Problem

Server-backed policies (e.g. `molmoact2` from inspect-robots-yam) are thin
HTTP clients; the model server is a separate process the user starts
themselves. When that server is down, a run fails with raw urllib3 spew:

```
  [error] scene-0: PolicyError: HTTPConnectionPool(host='127.0.0.1', port=8202):
  Max retries exceeded with url: /act (Caused by NewConnectionError(...
  [Errno 111] Connection refused))
```

Nothing tells the user a separate server exists, which URL was expected, or
how to start it. Observed live on the yam rig 2026-07-29: the arms homed,
ran an episode, and parked before the user learned the server was never up
(the pre-homing preflight is a separate follow-up; this plan is the
error-time hint only).

## Design

Enrich the `PolicyError` message at wrap time in the rollout, where both the
original exception and the policy instance are in hand. The hint becomes part
of the recorded error string (`_record_failure` embeds `str(exc)` verbatim,
`rollout.py:139-142`), so it persists into the log JSON and reaches the `run`
summary (`cli.py:987` prints `scene.error`) and the HTML viewer (`_html.py:730`,
`pre-wrap` so the multi-line hint survives). Known non-surface: the compact
`inspect` output prints `log.error` only (the fail-on-error threshold message),
never `scene.error`, so the hint does not appear there — accepted; `inspect`'s
job is post-hoc analysis, and the hint is in the JSON it reads. The `\nhint:`
continuation lines render flush-left (not dim, not indented) in the run
summary — accepted cosmetic tradeoff to keep the log data clean.

### Detection: `_connection_failure(exc)` (`src/inspect_robots/rollout.py`)

Walk the exception's `__cause__`/`__context__` chain (cycle-safe via an
`id()` seen-set), mirroring traceback semantics: follow `__cause__` when set,
otherwise follow `__context__` only when `__suppress_context__` is false — a
`raise X from None` inside an `except ConnectionError:` block explicitly
disclaims the connection error and must not earn a hint. Return `True` when
any node is an instance of builtin `ConnectionError` (covers
`ConnectionRefusedError`, `ConnectionResetError`) **or** its type name is one
of `{"ConnectionError", "NewConnectionError", "ConnectError"}` — matching
`requests`/`urllib3`/`httpx` types by name keeps core dependency-free.
Timeouts are deliberately excluded: a slow server is a different failure from
an absent one, and a timeout hint saying "is the server running?" would
mislead. For that same reason `MaxRetryError` is NOT in the set — urllib3
raises it for retry exhaustion regardless of cause, including timeouts. The
down-server chain from a requests-based client name-matches at the top
(`requests.exceptions.ConnectionError`) anyway.

Documented scope cuts: `ExceptionGroup` members (py3.11 task groups) are not
walked — nested connect errors live in `.exceptions`, not the cause chain —
and bare DNS failures (`socket.gaierror`) do not match; requests/httpx users
are covered because their top-level wrapper name-matches, and stdlib
`urllib.error.URLError` refusals match too (urllib raises inside the
`except` handling the builtin `ConnectionRefusedError`, so the walk finds it
via implicit `__context__`). Also out of scope: a policy
that raises during the `on_trial_start` hook (`eval.py:338` formats that
into `scene_error` directly, bypassing `PolicyError` wrapping) gets no hint —
server-backed policies connect at reset/act time, which is the path covered.

### Wrapping: `_policy_error(policy, exc)` (`src/inspect_robots/rollout.py`)

Replace the two generic-exception wrap sites — `rollout.py:207` (policy
`reset`) and `rollout.py:227` (`controller.next_action`) — with a helper:

```python
def _policy_error(policy: Policy, exc: Exception) -> PolicyError:
    """Wrap a generic policy exception, appending a hint on connection failures."""
    message = str(exc)
    if _connection_failure(exc):
        try:
            url = getattr(policy, "server_url", None)
            remedy = getattr(policy, "remedy", None)
            name = policy.info.name
            if url:
                # "up and healthy", not "running": builtin ConnectionError
                # matches also cover reset/broken-pipe, where the server
                # answered and then dropped the connection.
                message += (
                    f"\nhint: policy {name!r} could not hold a connection to its "
                    f"action server at {url} — is the server up and healthy? "
                    "Start (or restart) it, then rerun."
                )
            else:
                message += (
                    f"\nhint: policy {name!r} hit a connection failure — "
                    "a backend it depends on may be down or unreachable."
                )
            if remedy:
                message += f"\nhint: {remedy}"
        except Exception:
            pass  # a raising server_url/remedy property must not mask the trial error
    return PolicyError(message)
```

The `try/except Exception` guard exists because this runs inside the
rollout's own exception handlers, *before* `_record_failure` attaches the
partial record: `server_url`/`remedy` are plugin-supplied properties, and one
that raises would otherwise escape as an untyped exception with no `.record`,
crashing the eval instead of recording a failed trial. Note `message` stays
partially extended if the raise happens after an append — acceptable; every
already-appended line is well-formed.

The dimension-mismatch `PolicyError` at `rollout.py:254` is untouched (not a
connection failure by construction).

### Policy-declared remedy (duck-typed protocol)

Policies MAY expose two optional string attributes, read via `getattr`:

- `server_url` — the endpoint the policy talks to.
- `remedy` — a one-line, policy-specific pointer, e.g.
  `"start the MolmoAct2 YAM server (serves :8202), then rerun"`.

No change to the `Policy` protocol or `PolicyConfig` — absence of the
attributes degrades to the neutral hint. Follow-up contract for the
inspect-robots-yam PR: `ActServerPolicy` keeps its config private
(`self._cfg`, yam `src/inspect_robots_yam/policy.py:61`), so instance-level
`getattr` sees nothing today — the plugin must mirror the fields onto the
instance (e.g. a `server_url` property returning `self._cfg.server_url`, plus
`remedy`), following the existing `RUNTIME_REQUIREMENTS` ClassVar precedent
on that class. The properties should be plain non-raising reads; core guards
against raising ones anyway (above).

## Tests (`tests/test_rollout_hardening.py`)

Extend the existing rollout-hardening module, reusing its conventions: local
fake policy classes (`_BoomPolicy`-style) and the `_run()` helper
(`test_rollout_hardening.py:30-48`). Direct tests of the private helpers
follow existing precedent (tests already import private names). Fakes only;
no `requests`/`httpx` import. `branch = true` coverage is on, so every new
branch needs both sides exercised:

1. Policy `reset` raises builtin `ConnectionRefusedError` → recorded
   `PolicyError` message contains the neutral hint (no URL segment).
2. Policy exposing `server_url` and `remedy`, raising a fake
   `requests`-style chain (locally defined classes *named*
   `ConnectionError`/`NewConnectionError`, not builtins, linked with
   `raise ... from ...`) at step time → message contains the URL line and the
   remedy line.
3. Policy raising `ValueError` → message contains no `hint:`.
4. `_connection_failure` on a self-referential `__context__` cycle
   terminates and returns `False`.
5. A `MaxRetryError`-named wrapper chaining to a timeout → no hint
   (regression guard for keeping `MaxRetryError` out of the name set).
6. `raise ValueError(...) from None` inside an `except ConnectionError:`
   block (`__suppress_context__` true) → no hint; same shape without
   `from None` → hint (both sides of the suppress branch).
7. `server_url` present with no `remedy`, and `remedy` present with no
   `server_url` (both sides of each getattr branch).
8. Policy whose `server_url` property raises → trial still records a
   `PolicyError` (base message intact, eval does not crash); exercises the
   guard branch for 100% branch coverage without a pragma.

## Docs / changelog

- `CHANGELOG.md` → `[Unreleased] / Added`: one entry for the hint.
- `PolicyError` docstring (`src/inspect_robots/errors.py:54`): note that
  connection-level failures carry a remediation hint in the message.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`,
`uv run pytest --cov` (must stay at 100%). Baseline before changes: 876
passed, 3 skipped.

## Out of scope

- inspect-robots-yam `molmoact2` remedy population (follow-up plugin PR).
- Probing `server_url` in the `setup` runtime-requirements checklist
  (`_setup.py:171`).
- Pre-homing liveness preflight (separate safety-focused issue, per #219).
