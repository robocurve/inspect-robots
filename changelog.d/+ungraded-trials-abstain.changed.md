**Core:** a trial the grader tried and failed to judge no longer scores as a
robot failure. The grader records `grading_error` in the trial metadata, the
`operator` scorer abstains on it, and the run finishes every trial and then
ends with status `error` and an "N of M trial(s) ungraded" message
([plan 0085](plans/0085-grader-preflight-and-loud-ungraded.md)).
