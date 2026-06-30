# Command-line interface

The `roboinspect` CLI wraps the registry and [`eval`][roboinspect.eval.eval].

## `roboinspect list`

Show registered components (builtins + installed plugins):

```bash
roboinspect list                 # all kinds
roboinspect list policies        # just one kind
roboinspect list embodiments
```

## `roboinspect run`

Resolve a task/policy/embodiment from the registry and run an eval. Pass
constructor arguments with `-T` (task), `-P` (policy), and `-E` (embodiment) as
`key=value` (parsed as bool/int/float/None/str):

```bash
roboinspect run --task cubepick-reach --policy scripted --embodiment cubepick
roboinspect run --task cubepick-reach -T num_scenes=10 --policy scripted -P chunk_size=8 \
             --embodiment cubepick --log-dir logs --seed 0
```

The exit code is `0` on a successful eval, `1` otherwise.

## `roboinspect inspect`

Print a summary of a saved [`EvalLog`][roboinspect.log.EvalLog]:

```bash
roboinspect inspect logs/cubepick-reach_xxxx.json
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

## `roboinspect --version`

```bash
roboinspect --version
```
