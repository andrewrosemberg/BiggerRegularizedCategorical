"""mjlab ShadowHand environment components.

This module contains the reusable mjlab configuration for the object rotation
task. Supports single-object (cube) and multi-object (heterogeneous mesh
variants via VariantEntityCfg) configurations.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg, VariantEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg


ASSETS_DIR = Path(__file__).resolve().parent.parent / "dex_envs" / "assets" / "hand"
CUBE_XML = ASSETS_DIR / "manipulate_cube.xml"

GYM_ACTUATOR_ORDER = (
    ("robot0:A_WRJ1", "robot0:WRJ1", -0.489, 0.14),
    ("robot0:A_WRJ0", "robot0:WRJ0", -0.698, 0.489),
    ("robot0:A_FFJ3", "robot0:FFJ3", -0.349, 0.349),
    ("robot0:A_FFJ2", "robot0:FFJ2", 0.0, 1.571),
    ("robot0:A_FFJ1", "robot0:FFJ1", 0.0, 1.571),
    ("robot0:A_MFJ3", "robot0:MFJ3", -0.349, 0.349),
    ("robot0:A_MFJ2", "robot0:MFJ2", 0.0, 1.571),
    ("robot0:A_MFJ1", "robot0:MFJ1", 0.0, 1.571),
    ("robot0:A_RFJ3", "robot0:RFJ3", -0.349, 0.349),
    ("robot0:A_RFJ2", "robot0:RFJ2", 0.0, 1.571),
    ("robot0:A_RFJ1", "robot0:RFJ1", 0.0, 1.571),
    ("robot0:A_LFJ4", "robot0:LFJ4", 0.0, 0.785),
    ("robot0:A_LFJ3", "robot0:LFJ3", -0.349, 0.349),
    ("robot0:A_LFJ2", "robot0:LFJ2", 0.0, 1.571),
    ("robot0:A_LFJ1", "robot0:LFJ1", 0.0, 1.571),
    ("robot0:A_THJ4", "robot0:THJ4", -1.047, 1.047),
    ("robot0:A_THJ3", "robot0:THJ3", 0.0, 1.222),
    ("robot0:A_THJ2", "robot0:THJ2", -0.209, 0.209),
    ("robot0:A_THJ1", "robot0:THJ1", -0.524, 0.524),
    ("robot0:A_THJ0", "robot0:THJ0", -1.571, 0.0),
)

GYM_JOINT_ORDER = tuple(row[1] for row in GYM_ACTUATOR_ORDER)
GYM_CTRL_LO = np.array([row[2] for row in GYM_ACTUATOR_ORDER])
GYM_CTRL_HI = np.array([row[3] for row in GYM_ACTUATOR_ORDER])
GYM_CTRL_CENTER = (GYM_CTRL_HI + GYM_CTRL_LO) / 2.0
GYM_CTRL_HALFWIDTH = (GYM_CTRL_HI - GYM_CTRL_LO) / 2.0

GYM_CUBE_MASS = 0.15575129663498727
GYM_OBJ_DENSITY = 567
GYM_CUBE_DENSITY = GYM_OBJ_DENSITY
GYM_OBJ_FREEJOINT_DAMPING = 0.01

STLS_DIR = (ASSETS_DIR / ".." / "stls" / "hand" / "contactdb_objects").resolve()
YCB_DIR = (ASSETS_DIR / ".." / "stls" / "hand" / "ycb_objects").resolve()


def _resolve_stl(object_name: str) -> Path:
    """Find the STL file for an object in contactdb_objects or ycb_objects."""
    p = STLS_DIR / f"{object_name}.stl"
    if p.exists():
        return p
    p2 = YCB_DIR / f"{object_name}.stl"
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"Missing mesh for '{object_name}': checked {STLS_DIR} and {YCB_DIR}"
    )


OBJ_COLORS = {
    "cube": (0.8, 0.2, 0.2, 1.0),
    "ball": (0.2, 0.2, 0.8, 1.0),
    "apple": (0.2, 0.8, 0.2, 1.0),
}

SHADOWHAND_ACTUATED_JOINTS = GYM_JOINT_ORDER
SHADOWHAND_WRIST_JOINTS = ("robot0:WRJ1", "robot0:WRJ0")
SHADOWHAND_FINGER_JOINTS = tuple(
    joint for joint in SHADOWHAND_ACTUATED_JOINTS
    if joint not in SHADOWHAND_WRIST_JOINTS
)


def _embed_assets(spec: mujoco.MjSpec) -> None:
    """Embed mesh/texture bytes so mjlab can compile the spec on workers."""
    assets = {}
    meshdir = Path(spec.modelfiledir) / spec.meshdir if spec.meshdir else Path(spec.modelfiledir)
    texdir = Path(spec.modelfiledir) / spec.texturedir if spec.texturedir else Path(spec.modelfiledir)
    for mesh in spec.meshes:
        fpath = meshdir / mesh.file
        if fpath.exists():
            assets[mesh.file] = fpath.read_bytes()
    for tex in spec.textures:
        if tex.file:
            fpath = texdir / tex.file
            if fpath.exists():
                assets[tex.file] = fpath.read_bytes()
    spec.assets = assets


def _zero_geom_margins(spec: mujoco.MjSpec) -> None:
    """Set MuJoCo geom margins to zero for MuJoCo Warp compatibility."""
    spec.default.geom.margin = 0.0
    for class_name in (
        "robot0:asset_class",
        "robot0:D_Touch",
        "robot0:DC_Hand",
        "robot0:D_Vizual",
        "robot0:D_TH_type1",
        "robot0:D_TH_type2",
    ):
        dc = spec.find_default(class_name)
        if dc:
            dc.geom.margin = 0.0

    model = spec.compile()
    for i in range(model.ngeom):
        if model.geom_margin[i] != 0.0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name:
                geom = spec.geom(name)
                if geom:
                    geom.margin = 0.0


def get_shadowhand_spec() -> mujoco.MjSpec:
    """Return a ShadowHand-only spec derived from the cube Gymnasium XML."""
    if not CUBE_XML.exists():
        raise FileNotFoundError(f"Missing XML: {CUBE_XML}")

    spec = mujoco.MjSpec.from_file(str(CUBE_XML))
    for body in list(spec.worldbody.bodies):
        if body.name in ("target", "object", "floor0"):
            spec.delete(body)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    _zero_geom_margins(spec)
    _embed_assets(spec)
    return spec


def make_shadowhand_articulation() -> EntityArticulationInfoCfg:
    """Create mjlab P-control actuators matching the Gymnasium gains."""
    return EntityArticulationInfoCfg(
        actuators=(
            IdealPdActuatorCfg(
                target_names_expr=SHADOWHAND_WRIST_JOINTS,
                stiffness=5.0,
                damping=0.0,
                effort_limit=5.0,
            ),
            IdealPdActuatorCfg(
                target_names_expr=SHADOWHAND_FINGER_JOINTS,
                stiffness=1.0,
                damping=0.0,
                effort_limit=1.0,
            ),
        ),
    )


def get_cube_spec() -> mujoco.MjSpec:
    """Return a free-body mesh cube matching the Gymnasium cube mass."""
    return get_object_spec("cube")


def get_object_spec(object_name: str) -> mujoco.MjSpec:
    """Return a free-body mesh spec for any Gymnasium ShadowHand object."""
    stl_path = _resolve_stl(object_name)

    rgba = OBJ_COLORS.get(object_name, (0.6, 0.6, 0.6, 1.0))
    mesh_name = f"{object_name}_mesh"
    stl_basename = f"{object_name}.stl"

    spec = mujoco.MjSpec()
    spec.add_mesh(name=mesh_name, file=stl_basename)
    body = spec.worldbody.add_body(name="object")
    free_joint = body.add_freejoint(name="object_joint")
    free_joint.damping = np.full(3, GYM_OBJ_FREEJOINT_DAMPING)
    body.add_geom(
        name="object_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        density=GYM_OBJ_DENSITY,
        condim=4,
        rgba=rgba,
    )
    spec.assets = {stl_basename: stl_path.read_bytes()}
    return spec


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion product for tensors using MuJoCo's ``(w, x, y, z)`` order."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for tensors using MuJoCo's ``(w, x, y, z)`` order."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def compute_rotation_distance(q_achieved: torch.Tensor, q_desired: torch.Tensor) -> torch.Tensor:
    """Gymnasium Robotics rotation distance in radians.

    This intentionally follows Gymnasium Robotics' ShadowHand formula exactly:
    it clips the scalar component to [-1, 1] and does not canonicalize
    antipodal quaternion signs with ``abs``.
    """
    quat_diff = quat_mul(q_achieved, quat_conjugate(q_desired))
    return 2.0 * torch.acos(torch.clamp(quat_diff[..., 0], -1.0, 1.0))


def compute_is_success(
    q_achieved: torch.Tensor,
    q_desired: torch.Tensor,
    rotation_threshold: float = 0.1,
) -> torch.Tensor:
    """Return sparse success flags for target_position='ignore' rotation tasks."""
    return (compute_rotation_distance(q_achieved, q_desired) < rotation_threshold).float()


def compute_sparse_reward(
    q_achieved: torch.Tensor,
    q_desired: torch.Tensor,
    rotation_threshold: float = 0.1,
) -> torch.Tensor:
    """Return Gymnasium sparse rewards, where success is 0 and failure is -1."""
    return compute_is_success(q_achieved, q_desired, rotation_threshold) - 1.0


def object_root_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_pos_w


def object_root_quat(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_quat_w


def object_root_lin_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def object_root_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_ang_vel_w


def _find_obj_entity_name(env: ManagerBasedRlEnv) -> str:
    """Discover the object entity name from the scene."""
    for name in env.scene.entities:
        if name not in ("robot", "terrain"):
            return name
    raise KeyError("No object entity found in scene")


class desired_goal_obs:
    """Random z-axis target rotation observation re-sampled on reset."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._env = env
        self._obj_entity_name = _find_obj_entity_name(env)
        env._obj_entity_name = self._obj_entity_name
        self._target = torch.zeros((env.num_envs, 7), device=env.device)
        env._desired_goal_target = self._target
        self.reset(None)

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        n = len(env_ids)
        obj = self._env.scene[self._obj_entity_name]
        obj_pos = obj.data.root_link_pos_w[env_ids]
        angles = torch.rand(n, device=self._env.device) * 2 * torch.pi - torch.pi
        zeros = torch.zeros(n, device=self._env.device)
        qw = torch.cos(angles / 2)
        target_quat = torch.stack([qw, zeros, zeros, torch.sin(angles / 2)], dim=-1)
        target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True)
        self._target[env_ids] = torch.cat([obj_pos, target_quat], dim=-1)

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return self._target


class sparse_rotation_reward:
    """Sparse ShadowHand rotation reward term for mjlab."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self._obj_entity_name = _find_obj_entity_name(env)

    def reset(self, env_ids):
        pass

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        rotation_threshold: float = 0.1,
    ) -> torch.Tensor:
        target = env._desired_goal_target
        obj_quat = env.scene[self._obj_entity_name].data.root_link_quat_w
        return compute_sparse_reward(obj_quat, target[:, 3:], rotation_threshold)


def action_scale_offset() -> tuple[dict[str, float], dict[str, float]]:
    """Per-joint action scale and offset matching Gymnasium ctrlranges."""
    scale = {}
    offset = {}
    for _, joint, ctrl_lo, ctrl_hi in GYM_ACTUATOR_ORDER:
        scale[joint] = (ctrl_hi - ctrl_lo) / 2.0
        offset[joint] = (ctrl_hi + ctrl_lo) / 2.0
    return scale, offset


def _build_env_cfg(
    *,
    obj_entity_name: str,
    obj_entity_cfg: EntityCfg,
    num_envs: int,
    auto_reset: bool,
) -> ManagerBasedRlEnvCfg:
    """Shared builder for single- and multi-object ShadowHand configs."""
    obj_pos = (1.0, 0.87, 0.2)
    act_scale, act_offset = action_scale_offset()

    observations = {
        "policy": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(
                    func=envs_mdp.joint_pos_rel,
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
                ),
                "joint_vel": ObservationTermCfg(
                    func=envs_mdp.joint_vel_rel,
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
                ),
                "object_pos": ObservationTermCfg(
                    func=object_root_pos,
                    params={"asset_cfg": SceneEntityCfg(obj_entity_name)},
                ),
                "object_quat": ObservationTermCfg(
                    func=object_root_quat,
                    params={"asset_cfg": SceneEntityCfg(obj_entity_name)},
                ),
                "object_lin_vel": ObservationTermCfg(
                    func=object_root_lin_vel,
                    params={"asset_cfg": SceneEntityCfg(obj_entity_name)},
                ),
                "object_ang_vel": ObservationTermCfg(
                    func=object_root_ang_vel,
                    params={"asset_cfg": SceneEntityCfg(obj_entity_name)},
                ),
                "desired_goal": ObservationTermCfg(func=desired_goal_obs),
            },
            enable_corruption=False,
        ),
    }

    actions = {
        "joint_pos": envs_mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=act_scale,
            offset=act_offset,
            use_default_offset=False,
        ),
    }

    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.01, 0.01),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_object_pose": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(obj_entity_name),
                "pose_range": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.005, 0.005),
                    "yaw": (-3.14159, 3.14159),
                },
                "velocity_range": {},
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    }

    rewards = {
        "sparse_rotation": RewardTermCfg(
            func=sparse_rotation_reward,
            weight=1.0,
            params={"rotation_threshold": 0.1},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                    spec_fn=get_shadowhand_spec,
                    articulation=make_shadowhand_articulation(),
                ),
                obj_entity_name: obj_entity_cfg,
            },
            num_envs=num_envs,
            env_spacing=2.5,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        sim=SimulationCfg(
            nconmax=100,
            njmax=600,
            mujoco=MujocoCfg(timestep=0.002, iterations=20),
        ),
        decimation=20,
        episode_length_s=4.0,
        auto_reset=auto_reset,
        scale_rewards_by_dt=False,
    )


def build_shadowhand_cube_env_cfg(
    *,
    num_envs: int = 4,
    auto_reset: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the batched mjlab cube-rotation environment configuration."""
    return _build_env_cfg(
        obj_entity_name="cube",
        obj_entity_cfg=EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=(1.0, 0.87, 0.2),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spec_fn=get_cube_spec,
        ),
        num_envs=num_envs,
        auto_reset=auto_reset,
    )


def _make_asset_patcher(object_names: list[str], entity_prefix: str):
    """Return a spec_fn callback that injects all object STL assets.

    VariantEntityCfg's merge keeps only variant 0's assets dict. This
    callback patches the scene spec to include all variant mesh data
    after entity attachment but before simulation compilation.
    """
    asset_data = {}
    for name in object_names:
        stl_path = _resolve_stl(name)
        basename = f"{name}.stl"
        prefixed_key = f"{entity_prefix}/{basename}"
        asset_data[prefixed_key] = stl_path.read_bytes()

    def patch_assets(spec: mujoco.MjSpec) -> None:
        if spec.assets is None:
            spec.assets = {}
        spec.assets.update(asset_data)

    return patch_assets


def build_shadowhand_multiobject_env_cfg(
    *,
    object_names: list[str],
    num_envs: int = 12,
    auto_reset: bool = False,
    variant_assignment: list[int] | tuple[int, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build a batched mjlab env with heterogeneous object meshes per world.

    Uses VariantEntityCfg to assign different object meshes to different
    env slots within one batched ManagerBasedRlEnv. All variants share
    the same kinematic structure (single free-body with mesh geom).
    """
    if variant_assignment is not None:
        assignment = tuple(int(v) for v in variant_assignment)
        if len(assignment) != num_envs:
            raise ValueError(
                f"variant_assignment length {len(assignment)} != num_envs {num_envs}."
            )
        if any(v < 0 or v >= len(object_names) for v in assignment):
            raise ValueError(
                "variant_assignment entries must index object_names; "
                f"got {assignment} for {len(object_names)} objects."
            )

        def assignment_fn(nworld: int, assignment=assignment):
            if nworld != len(assignment):
                raise ValueError(
                    f"assignment requested for {nworld} worlds, "
                    f"but fixed assignment has {len(assignment)} worlds."
                )
            return assignment
    else:
        assignment_fn = None

    variants = {
        name: (lambda n=name: get_object_spec(n))
        for name in object_names
    }
    obj_entity_cfg = VariantEntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(1.0, 0.87, 0.2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        variants=variants,
        assignment=assignment_fn,
    )
    cfg = _build_env_cfg(
        obj_entity_name="object",
        obj_entity_cfg=obj_entity_cfg,
        num_envs=num_envs,
        auto_reset=auto_reset,
    )
    cfg.scene.spec_fn = _make_asset_patcher(object_names, "object")
    return cfg
