**Agent:** `-P toolset=plus` (IR+) offers the controller coding agents write
for themselves when they drive arms directly, with its arguments and motion:
`go` solves IK once for an `arm`'s goal pose (x, y, z with yaw, tilt, roll in
degrees, or a rotation matrix) on the arm's kinematic model and eases all its
joints there together; `move_joints`, `grip`, `park`, `state`, `look`, `wait`,
and `sequence` (several steps in one call). Several calls per turn run in
order. It needs a joint-position action space, `-P kinematics=module:factory`
naming the arms' model, and scipy (the `plus` extra).
