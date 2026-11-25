**ROS and CaP-X plugins:** the `ros` embodiment accepts `eef_pose_topic` on a
7-dimensional action space (a seven-joint arm such as a Franka, iiwa or Kinova
Gen3, or six joints plus a gripper). Core already picks `joint_pos` over a
same-shape `eef_pose` as the reference for `joint_pos` control, so the
construction-time refusal was no longer needed. The CaP-X policy now resolves
its joint-state field with the same rule (`reference_state_fields`) instead of
by shape alone, so it binds to these embodiments too.
