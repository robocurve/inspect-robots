**Core:** `DeltaLimitApprover` now validates that `max_delta` has a non-empty intersection with action space bounds in displacement mode, raising `ValueError` at construction if low exceeds high.
