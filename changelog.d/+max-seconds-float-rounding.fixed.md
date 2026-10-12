**Core:** A seconds-based task no longer gets an extra step from binary float
rounding: `max_seconds=1.1` at 50 Hz now resolves to 55 steps instead of 56, because
a product within a few ULPs of an integer counts as that integer before `ceil()`.
