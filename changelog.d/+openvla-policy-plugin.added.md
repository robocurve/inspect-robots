**OpenVLA plugin:** a new in-repo plugin runs base OpenVLA-7B in the eval
process, with no model server and no second environment, behind
`--policy openvla_local`. Its `eef_delta_pose` / `euler_xyz` / binary-gripper
contract is published for up-front compatibility checks, weights load lazily on
the first `act()`, and the whole suite (39 tests, 100% coverage) runs in CI
without `torch`, a GPU, or the checkpoint.
