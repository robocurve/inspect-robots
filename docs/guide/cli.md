# Command-line interface

The `inspect_robots` CLI wraps the registry and [`eval`][inspect_robots.eval.eval].

## `inspect-robots list`

Show registered components (builtins + installed plugins):

```bash
inspect-robots list                 # all kinds
inspect-robots list policies        # just one kind
inspect-robots list embodiments
```

## `inspect-robots run`

Resolve a task/policy/embodiment from the registry and run an eval. Pass
constructor arguments with `-T` (task), `-P` (policy), and `-E` (embodiment) as
`key=value` (parsed as bool/int/float/None/str):

```bash
inspect-robots run --task cubepick-reach --policy scripted --embodiment cubepick
inspect-robots run --task cubepick-reach -T num_scenes=10 --policy scripted -P chunk_size=8 \
             --embodiment cubepick --log-dir logs --seed 0
```

`--epochs N` overrides the task's epoch count, `--fail-on-error X` halts on
`PolicyError`s (`1` = first error, `0<X<1` = proportion, `X>1` = count), and
`--store-frames` streams camera frames to `<log-dir>/frames`. When the run
finishes, the path of the written log is printed.

The exit code is `0` on a successful eval, `1` otherwise.

## `inspect-robots inspect`

Print a summary of a saved [`EvalLog`][inspect_robots.log.EvalLog]:

```bash
inspect-robots inspect logs/cubepick-reach_xxxx.json
```

```text
task:        cubepick-reach
policy:      scripted
embodiment:  cubepick
status:      success
scenes:      5   trials: 5
metrics:
  success_at_end: 1
scenes:
  [success] scene-0: success_at_end=1
  ...
```

## `inspect-robots --version`

```bash
inspect-robots --version
```
