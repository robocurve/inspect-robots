**Core:** `inspect-robots doctor` checks the values written for
the embodiment's declared device slots before construction. Missing camera or serial
paths and missing CAN interfaces are reported together, and the command
exits nonzero when any are found. This catches stale configured device
references before the robot is run ([plan 0082](plans/0082-doctor-device-validation.md),
[#50](https://github.com/robocurve/inspect-robots/issues/50)).
