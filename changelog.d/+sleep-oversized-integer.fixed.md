**Core:** the `sleep` tool no longer raises `OverflowError` when a model passes
`seconds` as an integer too large for a float (more than 308 digits). The call
is capped at 180 s like any other over-long duration, so one malformed tool
call does not end the trial as a policy error.
