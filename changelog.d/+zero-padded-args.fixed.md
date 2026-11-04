**Core:** `-P`/`-E`/`-G` values and config.ini args with a leading zero, such as
the RealSense serial `035622060973`, stay strings instead of being parsed as
integers that silently drop the zeros. Numeric settings such as `max_steps` and
`rerun_port` still parse `0300` as `300`.
