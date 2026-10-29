"""Newton kitchen simulation with direct MuJoCo loading, readout, and rendering."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import mujoco
import newton
import numpy as np
from inspect_robots.embodiment import (
    AUTO_RESET,
    PRIVILEGED_SUCCESS,
    RENDERABLE,
    RESETTABLE,
    SEEDABLE,
    EmbodimentInfo,
)
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult
from newton.solvers import SolverMuJoCo

from tasks.contract import (
    ACTION_HIGH,
    ACTION_LOW,
    GRIPPER_ACTUATORS,
    JOINTS,
    action_semantics,
)

from .evaluation import TargetEvaluator, load_evaluator, resolve_setup
from .fluid import Fluid


class NewtonEmbodiment:
    """Run Newton's rigid solver and optional MPM liquid on CPU.

    Newton owns the simulation model, states and controls. MuJoCo is imported
    directly to load MJCF and maintain a readout for cameras and scoring; the
    readout model is never stepped. Every supported scene uses the same named
    five-joint arm and three tendon-driven fingers; task evaluators own scoring.
    """

    def __init__(
        self,
        *,
        control_hz: float = 50.0,
        camera: str = "",
        width: int = 320,
        height: int = 240,
        debug_actions: bool = False,
    ):
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ConfigError("control_hz must be finite and positive")
        self.control_hz = float(control_hz)
        self.camera = camera
        self.debug_actions = debug_actions
        self.width, self.height = width, height
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1:
            raise ConfigError("width and height must be positive integers")
        capabilities = {AUTO_RESET, RESETTABLE, SEEDABLE, PRIVILEGED_SUCCESS}
        if camera:
            capabilities.add(RENDERABLE)
        self.info = EmbodimentInfo(
            name="newton",
            action_space=Box(
                shape=(len(ACTION_LOW),),
                low=ACTION_LOW.copy(),
                high=ACTION_HIGH.copy(),
                semantics=action_semantics(),
            ),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(camera, height, width),) if camera else (),
                state=StateSpec(
                    fields=(
                        StateField("joint_pos", (8,), "rad"),
                        StateField("eef_pos", (3,), "m"),
                        StateField("target_pos", (3,), "m"),
                        StateField("destination_pos", (3,), "m"),
                    )
                ),
            ),
            control_hz=self.control_hz,
            is_simulated=True,
            capabilities=frozenset(capabilities),
            # Setup-local evaluators are discovered only when reset receives a path.
            supported_target_kinds=frozenset(),
            docs=(
                "Absolute radians in order pan, lift, elbow, wrist, roll, thumb, "
                "finger1, finger2. Zero is upright; +pan turns toward +world-y. "
                "Scene.setup is a trusted directory containing model.xml and evaluator.py. "
                "Physics: Newton SolverMuJoCo, "
                f"native CPU backend; MuJoCo {mujoco.__version__}. "
                "Scenes with fluid_spawn use one-way coupled CPU MPM water."
            ),
        )
        self._renderer = None
        self._ready = False
        self._done = False
        self._evaluator: TargetEvaluator | None = None
        self.setup_path: str | None = None
        self.evaluator_path: str | None = None
        self.evaluator_sha256: str | None = None
        self.model_path: str | None = None
        self.model_sha256: str | None = None
        self.model = None
        self.data = None
        self.newton_model = None
        self.solver = None
        self.state = None
        self._next_state = None
        self.control = None
        self.fluid: Fluid | None = None

    def _load_model(self, model_path: str | Path) -> None:
        """Load and validate a setup's resolved MJCF on demand."""
        path = Path(model_path).resolve()
        if not path.is_file():
            raise ConfigError(f"MJCF not found: {path}")
        resolved = str(path)
        if self.model_path == resolved and self.model is not None:
            return
        self._release_model()
        try:
            self.model = mujoco.MjModel.from_xml_path(resolved)
        except ValueError as exc:
            raise ConfigError(f"Cannot load MJCF {resolved}: {exc}") from exc
        try:
            self.data = mujoco.MjData(self.model)
            self.model_path = resolved
            self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            ratio = 1.0 / (self.control_hz * self.model.opt.timestep)
            if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio)):
                raise ConfigError("control_hz must divide the model's physics frequency exactly")
            self.substeps = round(ratio)
            if self.camera and (
                self.width > self.model.vis.global_.offwidth
                or self.height > self.model.vis.global_.offheight
            ):
                raise ConfigError("Camera dimensions exceed the MJCF offscreen framebuffer")
            if self.camera and self.camera != "overview":
                self._id(mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
            self._render_camera = self.camera
            if self.camera == "overview":
                overview = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(overview)
                overview.lookat[:] = [0.2, 0.1, 0.3]
                overview.distance = 1.6
                overview.azimuth = 200
                overview.elevation = -25
                self._render_camera = overview
            self._bind_fixed_arm()
            mujoco.mj_forward(self.model, self.data)
            self._init_physics()
            self.fluid = Fluid.from_scene(self.model, self.data, self.newton_model)
        except Exception:
            self._release_model()
            raise

    def _bind_fixed_arm(self) -> None:
        """Bind the fixed named arm contract and reject incompatible MJCFs."""
        self._site = self._id(mujoco.mjtObj.mjOBJ_SITE, "pinch")
        joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
        self._arm_actuators = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINTS]
        )
        for joint, actuator in zip(joints, self._arm_actuators, strict=True):
            if not (
                self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
                and self.model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
                and self.model.actuator_trnid[actuator, 0] == joint
                and np.array_equal(self.model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])
                and self.model.actuator_dyntype[actuator] == mujoco.mjtDyn.mjDYN_NONE
                and self.model.actuator_gaintype[actuator] == mujoco.mjtGain.mjGAIN_FIXED
                and self.model.actuator_biastype[actuator] == mujoco.mjtBias.mjBIAS_AFFINE
                and self.model.actuator_gainprm[actuator, 0] > 0
                and self.model.actuator_biasprm[actuator, 0] == 0
                and self.model.actuator_biasprm[actuator, 1]
                == -self.model.actuator_gainprm[actuator, 0]
                and self.model.actuator_ctrllimited[actuator]
                and self.model.jnt_limited[joint]
            ):
                raise ConfigError(
                    f"{JOINTS[joints.index(joint)]} must use a bounded unit-gear position servo"
                )
        self._qpos = self.model.jnt_qposadr[joints]
        arm_ranges = self.model.actuator_ctrlrange[self._arm_actuators]
        arm_low = np.maximum(arm_ranges[:, 0], self.model.jnt_range[joints, 0])
        arm_high = np.minimum(arm_ranges[:, 1], self.model.jnt_range[joints, 1])
        self._gripper_actuators = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in GRIPPER_ACTUATORS]
        )
        self._tendons = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_TENDON, name) for name in GRIPPER_ACTUATORS]
        )
        if not np.all(
            self.model.actuator_trntype[self._gripper_actuators] == mujoco.mjtTrn.mjTRN_TENDON
        ):
            raise ConfigError("The three gripper actuators must drive fixed tendons")
        grip_ranges = self.model.actuator_ctrlrange[self._gripper_actuators]
        self._actuators = np.concatenate((self._arm_actuators, self._gripper_actuators))
        low = np.concatenate((arm_low, grip_ranges[:, 0]))
        high = np.concatenate((arm_high, grip_ranges[:, 1]))
        if not np.array_equal(low, ACTION_LOW) or not np.array_equal(high, ACTION_HIGH):
            raise ConfigError("MJCF arm controls do not match the fixed eight-dimensional contract")
        self._low, self._high = low, high

    def _init_physics(self) -> None:
        # MuJoCo expands <replicate> and <frame> before Newton's MJCF importer.
        with TemporaryDirectory(prefix="kitchen-newton-") as directory:
            expanded = Path(directory) / "expanded.xml"
            mujoco.mj_saveLastXML(str(expanded), self.model)
            # mj_saveLastXML abbreviates [timeconst, 1] to one number.
            # Newton 1.6's custom-attribute parser pads that with zero instead
            # of MuJoCo's default damping ratio. Restore the omitted value.
            tree = ElementTree.parse(expanded)
            for element in tree.iter():
                solref = element.get("solref", "").split()
                if len(solref) == 1:
                    element.set("solref", f"{solref[0]} 1")
            tree.write(expanded, encoding="unicode")
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(expanded), ctrl_direct=True, collapse_fixed_joints=False)
        self.newton_model = builder.finalize(device="cpu")
        self.solver = SolverMuJoCo(
            self.newton_model,
            use_mujoco_cpu=True,
            use_mujoco_contacts=True,
            skip_visual_only_geoms=False,
        )
        self.state = self.newton_model.state()
        self._next_state = self.newton_model.state()
        self.control = self.newton_model.control()
        labels = self.newton_model.mujoco.actuator_label
        original_labels = [self.model.actuator(i).name for i in range(self.model.nu)]
        if len(labels) != len(original_labels) or set(labels) != set(original_labels):
            raise ConfigError("Newton import changed the kitchen actuator contract")
        self._control_map = np.array([labels.index(name) for name in original_labels])
        self._newton_actuators = self._control_map[self._actuators]
        joint_labels = [label.rsplit("/", 1)[-1] for label in self.newton_model.joint_label]
        starts = self.newton_model.joint_q_start.numpy()
        self._newton_qpos = np.array([starts[joint_labels.index(name)] for name in JOINTS])
        self._map_readout()

    def _map_readout(self) -> None:
        """Map every joint, including unnamed food free joints, by owning body."""
        model = self.solver.mj_model
        labels = self.newton_model.body_label
        body_labels = {label.rsplit("/", 1)[-1]: label for label in labels}
        if len(body_labels) != len(labels):
            raise ConfigError("MJCF bodies must have unique names")
        qmap = np.full(self.model.nq, -1, dtype=int)
        vmap = np.full(self.model.nv, -1, dtype=int)
        for body_id in range(1, self.model.nbody):
            name = self.model.body(body_id).name
            label = body_labels.get(name)
            converted = model.body(label.replace("/", "_")) if label else None
            count = self.model.body_jntnum[body_id]
            if converted is None or model.body_jntnum[converted.id] != count:
                raise ConfigError(f"Newton import changed joints for body {name!r}")
            for offset in range(count):
                src = int(model.body_jntadr[converted.id]) + offset
                dst = int(self.model.body_jntadr[body_id]) + offset
                kind = int(self.model.jnt_type[dst])
                if model.jnt_type[src] != kind:
                    raise ConfigError(f"Newton import changed joint type for {name!r}")
                nq, nv = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}[kind]
                q = self.model.jnt_qposadr[dst]
                v = self.model.jnt_dofadr[dst]
                qmap[q : q + nq] = np.arange(model.jnt_qposadr[src], model.jnt_qposadr[src] + nq)
                vmap[v : v + nv] = np.arange(model.jnt_dofadr[src], model.jnt_dofadr[src] + nv)
        if np.any(qmap < 0) or np.any(vmap < 0):
            raise ConfigError("Newton import left unmapped MJCF coordinates")
        self._readout_qpos, self._readout_qvel = qmap, vmap

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ConfigError(f"MJCF is missing {kind.name} {name!r}")
        return index

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Lazily load a trusted setup directory and restore the simulation."""
        self._ready = False
        if scene.target is None:
            raise ConfigError("A scene target is required")
        if scene.setup is None:
            raise ConfigError("Scene setup must provide a setup directory path")
        setup = resolve_setup(scene.setup)
        try:
            self._load_model(setup.model)
            self._evaluator, self.evaluator_sha256 = load_evaluator(
                setup.evaluator, scene.target.kind, self.model
            )
        except Exception:
            self._release_model()
            raise
        self.setup_path = str(setup.directory)
        self.evaluator_path = str(setup.evaluator)
        self._instruction = scene.instruction
        actual_seed = seed if seed is not None else scene.init_seed
        rng = np.random.default_rng(0 if actual_seed is None else actual_seed)
        self._reset_physics(rng)
        self._evaluator.reset(
            self.model,
            self.data,
            scene.target.spec,
            self.control_hz,
            particles=self.fluid.positions if self.fluid else None,
        )
        self._ready, self._done = True, False
        return self._observe()

    def _reset_physics(self, rng: np.random.Generator) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos[0]] = np.clip(rng.uniform(-0.1, 0.1), self._low[0], self._high[0])
        self.data.ctrl[:] = 0
        self.data.ctrl[self._arm_actuators] = self.data.qpos[self._qpos]
        mujoco.mj_forward(self.model, self.data)
        # Reset both Newton buffers and native solver caches/time/warnings.
        mujoco.mj_resetData(self.solver.mj_model, self.solver.mj_data)
        for state in (self.state, self._next_state):
            self.solver.reset(state)
            q = state.joint_q.numpy()
            q[self._newton_qpos] = self.data.qpos[self._qpos]
            state.joint_q.assign(q)
            state.clear_forces()
            newton.eval_fk(self.newton_model, state.joint_q, state.joint_qd, state)
        controls = np.zeros(len(self.newton_model.mujoco.actuator_label), dtype=np.float32)
        controls[self._control_map] = self.data.ctrl
        self.control.mujoco.ctrl.assign(controls)
        self.solver.mj_data.qpos[self._readout_qpos] = self.data.qpos
        self.solver.mj_data.ctrl[:] = controls
        if self.fluid:
            self.fluid.reset(self.state)

    def step(self, action: Action) -> StepResult:
        """Apply bounded targets and score the model-specific task state."""
        if self.debug_actions:
            print("Actions is", action)
        if not self._ready or self._done:
            raise EmbodimentFault("Call reset() before stepping a new episode")
        values = np.asarray(action.data, dtype=np.float64)
        expected_shape = (len(self._actuators),)
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise EmbodimentFault("Expected eight finite arm-and-gripper targets")
        try:
            self._step_physics(np.clip(values, self._low, self._high))
        except Exception:
            # A partial rigid/fluid step cannot be safely continued.
            self._ready = False
            raise
        evaluation = self._evaluator.evaluate(
            self.model, self.data, particles=self.fluid.positions if self.fluid else None
        )
        self._done = evaluation.success
        return StepResult(
            observation=self._observe(),
            reward=-evaluation.distance,
            terminated=evaluation.success,
            termination_reason="success" if evaluation.success else None,
            info={
                **evaluation.info,
                "distance": evaluation.distance,
                "sim_time": float(self.data.time),
            },
        )

    def _step_physics(self, values: np.ndarray) -> None:
        controls = self.control.mujoco.ctrl.numpy()
        controls[self._newton_actuators] = values
        self.control.mujoco.ctrl.assign(controls)
        data = self.solver.mj_data
        warnings = data.warning.number.copy()
        for _ in range(self.substeps):
            self.state.clear_forces()
            self.solver.step(
                self.state, self._next_state, self.control, None, self.model.opt.timestep
            )
            self.state, self._next_state = self._next_state, self.state
            if (
                not np.all(np.isfinite(self.state.joint_q.numpy()))
                or not np.all(np.isfinite(self.state.joint_qd.numpy()))
                or np.any(data.warning.number > warnings)
            ):
                raise EmbodimentFault("Newton reported an unstable simulation or physics warning")
            if self.fluid:
                self.fluid.step(self.state, self.model.opt.timestep)
        self._sync_readout(controls)

    def _sync_readout(self, controls: np.ndarray) -> None:
        """Read Newton's rigid state without advancing a second simulation."""
        data = self.solver.mj_data
        self.data.qpos[:] = data.qpos[self._readout_qpos]
        self.data.qvel[:] = data.qvel[self._readout_qvel]
        self.data.ctrl[:] = controls[self._control_map]
        self.data.time = data.time
        # Forward kinematics only: rendering and scores describe Newton's state.
        mujoco.mj_forward(self.model, self.data)

    def _observe(self) -> Observation:
        images = {}
        if self.camera:
            images[self.camera] = self._render()
        state = {
            "joint_pos": np.concatenate(
                (self.data.qpos[self._qpos], self.data.ten_length[self._tendons])
            ),
            "eef_pos": self.data.site_xpos[self._site].copy(),
        }
        state.update(self._evaluator.state(self.model, self.data))
        return Observation(
            images=images,
            state=state,
            instruction=self._instruction,
            state_time=float(self.data.time),
            image_times={name: float(self.data.time) for name in images},
        )

    def _render(self) -> np.ndarray:
        """Render rigid readout, fluid, and optional evaluator marker together."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._renderer.update_scene(self.data, camera=self._render_camera)
        scene = self._renderer.scene
        if self.fluid:
            self.fluid.render(scene)
        marker = self._evaluator.marker()
        if self.camera == "overview" and marker is not None and scene.ngeom < scene.maxgeom:
            marker_pos, marker_radius = marker
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, marker_radius),
                marker_pos,
                np.eye(3).ravel(),
                np.array([0.1, 1.0, 0.2, 0.5], dtype=np.float32),
            )
            scene.ngeom += 1
        return self._renderer.render().copy()

    def close(self) -> None:
        """Release the lazily loaded model and renderer; repeated calls are safe."""
        self._release_model()

    def _release_model(self) -> None:
        """Drop all per-MJCF resources so another scene can load cleanly."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._ready = False
        self._done = False
        self._evaluator = None
        self.setup_path = None
        self.evaluator_path = None
        self.evaluator_sha256 = None
        self.model_path = None
        self.model_sha256 = None
        self.model = None
        self.data = None
        self.newton_model = None
        self.solver = None
        self.state = None
        self._next_state = None
        self.control = None
        self.fluid = None
