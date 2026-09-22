# Explicit grader effort errors

The user approved strict handling of an explicitly requested effort, including
`none`: send the requested value, surface provider rejection, and preserve the
run log. This supersedes plan 0075's warning-and-ungraded behavior for rejected
explicit-effort requests. It does not change provider defaults when effort is
omitted, or the separate documentation correction in PR #398.

## Contract

- Preserve the existing omitted-versus-explicit sentinel. Explicit Python
  `None` from CLI parsing becomes the wire string `"none"`; named and numeric
  effort values otherwise pass through unchanged.
- HTTP 400/422 on a request carrying explicit effort raises a private subtype
  of `ConfigError`. Provider error formats vary, so do not infer support by
  matching English error prose. Include the original response and requested
  effort. These statuses reject the request; they need not prove that effort
  alone caused the rejection. Never retry without effort or at another level.
- The existing token-cap compatibility retry retains the identical effort.
  Network/server failures and requests with omitted effort retain their existing
  grading behavior.
- The VLM grader propagates this typed rejection. Evaluation marks the trial
  as errored, records `grading_error` metadata, skips its scorers, and stops later
  epochs/scenes even with `fail_on_error=False`. Earlier scores, frames, actions,
  and log metadata survive. Save through `on_eval_end` before raising the error.
- `eval_set()` propagates the same error instead of starting the next task or
  replacing the completed log with an empty error row.
- No new public error class, dependencies, or log-schema fields.

## Validation

Mock provider rejections for explicit `None`, `"none"`, other named levels, and
numeric zero. Assert the exact wire value and no fallback request. Integration
tests use real mock-world rollouts and JSON log persistence to verify prior
results survive, the rejected trial is unscored, and later work never starts.
The previous degrade test changes because the user explicitly changed that
contract; omitted-effort degradation remains covered separately.
