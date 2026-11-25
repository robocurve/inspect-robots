**Core:** `-P`/`-E`/`-G` values and config.ini args that contain an underscore,
such as `2024_10_25`, stay strings instead of being parsed as numbers with the
underscores read as digit separators (`20241025`).
