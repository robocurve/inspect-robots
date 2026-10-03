**Core:** a run in which no scene completed cleanly now warns instead of
passing silently (survivor bias,
[#440](https://github.com/robocurve/inspect-robots/issues/440)). Its status is
unchanged, since `fail_on_error` stays the caller's tolerance control, but
`eval()` emits a `UserWarning` and the run summary and `inspect` print how
many trials the metrics rest on.
