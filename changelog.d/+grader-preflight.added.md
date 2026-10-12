**Core:** the `vlm` grader sends one preflight request (exact model, endpoint
and effort, with a small test image) before any rollout, and the CLI runs it
before the policy loads or the robot connects. An HTTP 4xx rejection stops the
run with the provider's message; an outage only warns
([plan 0085](plans/0085-grader-preflight-and-loud-ungraded.md)).
