**Core:** `DeltaLimitApprover` now keeps its own copy of each approved action of an
absolute-mode trial as its reference. A policy that wrote each target into the
same NumPy array moved the reference along with its next request, so the first
jump passed unlimited, and a clamped action mutated in
place by a later approver moved the reference the same way.
