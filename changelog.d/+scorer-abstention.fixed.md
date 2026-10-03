**Core:** a scorer can abstain with `Score(value=None)`. The value used to
crash `value_to_float`, so the only workaround was `0.0`, which reads as a
failed trial. Abstained epochs are now recorded as `null`, left out by the
epoch reducers and the metric mean, and a scorer that abstained everywhere
reports a `null` metric. `EvalResults.abstentions` counts abstained trials
per scorer, and `inspect` and `view` show the count beside each metric
([#436](https://github.com/robocurve/inspect-robots/issues/436)).
