**Core:** `DeltaLimitApprover` now copies the first approved action of an
absolute-mode trial before keeping it as its reference. A policy that wrote each
target into the same NumPy array moved the reference along with its next
request, so the first jump (and every step after it) passed unlimited.
