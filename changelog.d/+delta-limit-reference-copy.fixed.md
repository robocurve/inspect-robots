**Core:** `DeltaLimitGuard` keeps its own copy of the first approved action. It
stored the caller's array, so a policy that reused one output buffer moved the
limiter's reference along with it and its second action was never limited.
Guards built from specs (`--guard`, `build_guard`) were unaffected because
`GuardChain` already copies; a `DeltaLimitGuard` passed directly to `eval()` was.
