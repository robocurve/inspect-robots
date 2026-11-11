"""Small CPU MPM liquid with prescribed colliders from Newton's rigid state."""

from __future__ import annotations

import mujoco
import newton
import numpy as np
import warp as wp
from inspect_robots.errors import ConfigError, EmbodimentFault
from newton.solvers import SolverImplicitMPM


class Fluid:
    """One-way coupling: rigid shapes push liquid, without liquid reaction forces.

    A box site named ``fluid_spawn`` opts a scene into MPM. The optional MJCF
    numerics ``fluid_spacing`` and ``fluid_voxel_size`` are lengths in metres.
    Nothing here knows about task goals or success conditions.
    """

    @classmethod
    def from_scene(cls, model, data, rigid_model) -> Fluid | None:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fluid_spawn")
        if site < 0:
            return None
        if model.site_type[site] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ConfigError("fluid_spawn must be a box site")
        spacing = cls._length(model, "fluid_spacing", 0.004)
        voxel_size = cls._length(model, "fluid_voxel_size", 0.008)
        extent = 2 * model.site_size[site]
        dimensions = np.floor(extent / spacing + 1e-6).astype(int)
        if np.any(dimensions < 2) or np.prod(dimensions) > 4096:
            raise ConfigError("fluid_spawn must fit at least 2 particles per axis and at most 4096")
        cell_size = extent / dimensions
        axes = [
            (np.arange(count) + 0.5) * width - half
            for count, width, half in zip(dimensions, cell_size, extent / 2, strict=True)
        ]
        local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        positions = local @ data.site_xmat[site].reshape(3, 3).T + data.site_xpos[site]
        return cls(positions, cell_size, voxel_size, rigid_model)

    @staticmethod
    def _length(model: mujoco.MjModel, name: str, default: float) -> float:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
        if index < 0:
            return default
        values = model.numeric_data[model.numeric_adr[index] :][: model.numeric_size[index]]
        if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
            raise ConfigError(f"{name} must be a single positive finite length")
        return float(values[0])

    def __init__(self, positions, cell_size, voxel_size, rigid_model):
        self.radius = float(np.min(cell_size) / 2)
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        # Equal particle masses make counted fractions equal to mass fractions.
        builder.add_particles(
            pos=positions,
            vel=np.zeros_like(positions),
            mass=np.full(len(positions), 1000 * np.prod(cell_size)),
            radius=np.full(len(positions), self.radius),
        )
        self.model = builder.finalize(device="cpu")
        self.model.set_gravity(rigid_model.gravity.numpy()[0])
        # Newton's water-like dam-break material: no friction or tensile strength.
        self.model.mpm.friction.fill_(0.0)
        self.model.mpm.tensile_yield_ratio.fill_(1.0)
        self.model.mpm.viscosity.fill_(0.0)
        self.solver = SolverImplicitMPM(
            self.model,
            SolverImplicitMPM.Config(
                grid_type="dense",  # Sparse NanoVDB allocation requires CUDA.
                voxel_size=voxel_size,
                max_iterations=50,
                tolerance=1e-3,
                warmstart_mode="particles",
                strain_basis="P0",
                collider_basis="pic",
                collider_velocity_mode="backward",
            ),
        )
        self._setup_colliders(rigid_model)
        self.state = self.model.state()
        self._next_state = self.model.state()
        self.positions = np.asarray(positions, dtype=np.float32).copy()

    def _setup_colliders(self, rigid_model: newton.Model) -> None:
        # MPM joins a body's shapes into one triangle surface. Overlapping rigid
        # wall boxes leave internal faces that leak liquid at their seams. An
        # optional seamless fluid_collision_* mesh replaces that body's fluid
        # colliders, while MuJoCo retains the original rigid collision shapes.
        original = rigid_model.shape_flags.numpy().copy()
        selected = original.copy()
        overrides = np.array(
            [
                label.rsplit("/", 1)[-1].startswith("fluid_collision_")
                for label in rigid_model.shape_label
            ],
            dtype=bool,
        )
        if np.any(overrides):
            body = rigid_model.shape_body.numpy()
            particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
            selected[np.isin(body, body[overrides])] &= ~particle_flag
            selected[overrides] |= particle_flag
        try:
            rigid_model.shape_flags.assign(selected)
            self.solver.setup_collider(
                model=rigid_model,
                body_mass=wp.zeros_like(rigid_model.body_mass),
                body_inv_inertia=wp.zeros_like(rigid_model.body_inv_inertia),
            )
        finally:
            rigid_model.shape_flags.assign(original)

    def reset(self, rigid_state: newton.State) -> None:
        for state in (self.state, self._next_state):
            state.particle_q.assign(self.model.particle_q)
            state.particle_qd.assign(self.model.particle_qd)
            # Collider transforms are read-only aliases to the rigid state, so
            # this particle state intentionally has no body-force buffer.
            state.particle_f.zero_()
            self._colliders(state, rigid_state)
            self.solver.reset(state)
        self.positions = self.state.particle_q.numpy().copy()

    @staticmethod
    def _colliders(state: newton.State, rigid_state: newton.State) -> None:
        # Read-only aliases: MPM steps particles and never steps these bodies.
        state.body_q = rigid_state.body_q
        state.body_qd = rigid_state.body_qd

    def step(self, rigid_state: newton.State, dt: float) -> None:
        self._colliders(self.state, rigid_state)
        self._colliders(self._next_state, rigid_state)
        self.solver.step(self.state, self._next_state, None, None, dt)
        self.solver.project_outside(self._next_state, self._next_state, dt)
        self.state, self._next_state = self._next_state, self.state
        positions = self.state.particle_q.numpy()
        if not np.all(np.isfinite(positions)) or not np.all(
            np.isfinite(self.state.particle_qd.numpy())
        ):
            raise EmbodimentFault("Newton MPM reported a non-finite particle state")
        self.positions = positions.copy()

    def render(self, scene: mujoco.MjvScene) -> None:
        """Render the actual particles into the existing MuJoCo camera scene."""
        if scene.ngeom + len(self.positions) > scene.maxgeom:
            raise EmbodimentFault("Camera scene has insufficient space for fluid particles")
        for position in self.positions:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, self.radius),
                position.astype(float),
                np.eye(3).ravel(),
                np.array([0.05, 0.45, 0.95, 1], dtype=np.float32),
            )
            scene.ngeom += 1
