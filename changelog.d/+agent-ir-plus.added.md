**Agent:** `-P toolset=plus` (IR+) offers the controller functions coding
agents write for themselves when they drive arms directly: `go` (or
`move_joints` on joint spaces) easing in and out over `seconds`, `grip`,
`park`, `state`, `look` and `wait`, and runs several tool calls per turn in
order, each motion playing out before the next call. The model sees the next
observation once the turn's calls have run.
