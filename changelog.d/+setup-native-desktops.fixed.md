**Setup wizard:** local macOS and Windows sessions no longer default to
`rerun = false` or print a headless warning just because X11/Wayland display
variables are absent. SSH sessions without a display still receive the
warning; forwarded displays and saved or explicitly entered viewer settings
remain supported. Empty display variables now count as unavailable
([#437](https://github.com/robocurve/inspect-robots/pull/437)).
