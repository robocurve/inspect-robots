# Post-action frame persistence

## Problem

`FrameStore` currently writes each step's pre-action `Observation`, then stores
the unmodified `StepResult` in `StepRecord`. This has two consequences:

1. `StepResult.observation.images` retains full camera arrays in memory, so a
   stored trajectory does not satisfy the frame-store memory bound.
2. The post-action observation from the final step has no sidecar reference.
   Offline scorers therefore cannot reconstruct the terminal visual state.

For non-terminal steps, the same post-action observation becomes the next
pre-action observation and is stored one iteration later. The terminal
observation has no next iteration and is lost from the sidecar sequence.

## Contract

With a `FrameStore`, every distinct rollout observation is written exactly once:

- reset observation at frame index `0`;
- result of step `t` at frame index `t + 1`.

`StepRecord.observation` and `StepRecord.result.observation` contain no image
arrays. Their handles live in `image_refs` and `result_image_refs`,
respectively. Without a `FrameStore`, both observations and object identity
remain unchanged.

This makes a trial with `n` steps and one camera produce `n + 1` frame files.
The final `result_image_refs` always identifies the terminal post-action frame.

## Implementation

The rollout stores the reset observation before entering the control loop. Each
iteration then:

1. gives the full current observation to the policy and sinks;
2. executes the action;
3. stores and strips the resulting observation at index `t + 1`;
4. records both pre-action and post-action handles; and
5. reuses the post-action stored view as the next step's pre-action view.

Reusing the stored view prevents duplicate writes. The live observation with
images remains available for the next policy call, so policy behavior and sink
output do not change.

## Compatibility

`result_image_refs` is appended to `StepRecord` with a default of `None`, so
existing keyword and positional constructors remain valid. `StepRecord` is an
in-memory scorer interface and is not serialized in schema-versioned
`EvalLog`, so no log migration is required.

Runs without `FrameStore` preserve the original `Observation` and `StepResult`
objects. Runs with `FrameStore` intentionally replace only the recorded
`StepResult` with an image-stripped copy.

## Verification

Regression coverage must prove:

- pre-action and post-action arrays are absent from stored records;
- both reference maps load valid lossless arrays;
- frame indices are contiguous from `0` through the terminal `n`;
- one-camera trials write exactly `n + 1` files; and
- the terminal post-action frame is recoverable.

All existing lint, format, strict typing, and 100% line and branch coverage
gates remain required.
