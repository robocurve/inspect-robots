**Core:** Safely fall back to an empty tuple when `samples` is omitted in `EvalLog.from_dict` rather than raising `KeyError`, preserving deserialization compatibility with sample-less logs.
