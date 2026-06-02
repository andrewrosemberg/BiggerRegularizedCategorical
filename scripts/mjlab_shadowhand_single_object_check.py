"""Single-object ShadowHand mjlab parity prototype.

Compares the current Gymnasium ShadowHand cube-rotate-v1 environment against
an equivalent mjlab ManagerBasedRlEnv to answer:

  "Can we represent one current ShadowHand rotation task in mjlab with
   compatible observation, action, reset, reward, success, and timeout
   semantics?"

Requires the mjlab virtualenv:
  .venv_mjlab_check/bin/python scripts/mjlab_shadowhand_single_object_check.py

Uses mjlab v1.4.0 with auto_reset=False.
"""

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import IdealPdActuatorCfg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).resolve().parent.parent / "dex_envs" / "assets" / "hand"
CUBE_XML = ASSETS_DIR / "manipulate_cube.xml"
assert CUBE_XML.exists(), f"Missing XML: {CUBE_XML}"


# ---------------------------------------------------------------------------
# Gymnasium actuator data (extracted from compiled manipulate_cube.xml)
#
# Mapping: policy action in [-1, 1] -> ctrl = center + action * half_width
# where center = (ctrl_hi + ctrl_lo) / 2, half_width = (ctrl_hi - ctrl_lo) / 2
# relative_control=False
# ---------------------------------------------------------------------------

GYM_ACTUATOR_ORDER = (
    ("robot0:A_WRJ1", "robot0:WRJ1", -0.489,  0.14 ),
    ("robot0:A_WRJ0", "robot0:WRJ0", -0.698,  0.489),
    ("robot0:A_FFJ3", "robot0:FFJ3", -0.349,  0.349),
    ("robot0:A_FFJ2", "robot0:FFJ2",  0.0,    1.571),
    ("robot0:A_FFJ1", "robot0:FFJ1",  0.0,    1.571),
    ("robot0:A_MFJ3", "robot0:MFJ3", -0.349,  0.349),
    ("robot0:A_MFJ2", "robot0:MFJ2",  0.0,    1.571),
    ("robot0:A_MFJ1", "robot0:MFJ1",  0.0,    1.571),
    ("robot0:A_RFJ3", "robot0:RFJ3", -0.349,  0.349),
    ("robot0:A_RFJ2", "robot0:RFJ2",  0.0,    1.571),
    ("robot0:A_RFJ1", "robot0:RFJ1",  0.0,    1.571),
    ("robot0:A_LFJ4", "robot0:LFJ4",  0.0,    0.785),
    ("robot0:A_LFJ3", "robot0:LFJ3", -0.349,  0.349),
    ("robot0:A_LFJ2", "robot0:LFJ2",  0.0,    1.571),
    ("robot0:A_LFJ1", "robot0:LFJ1",  0.0,    1.571),
    ("robot0:A_THJ4", "robot0:THJ4", -1.047,   1.047),
    ("robot0:A_THJ3", "robot0:THJ3",  0.0,    1.222),
    ("robot0:A_THJ2", "robot0:THJ2", -0.209,  0.209),
    ("robot0:A_THJ1", "robot0:THJ1", -0.524,  0.524),
    ("robot0:A_THJ0", "robot0:THJ0", -1.571,  0.0  ),
)

GYM_JOINT_ORDER = tuple(row[1] for row in GYM_ACTUATOR_ORDER)
GYM_CTRL_LO    = np.array([row[2] for row in GYM_ACTUATOR_ORDER])
GYM_CTRL_HI    = np.array([row[3] for row in GYM_ACTUATOR_ORDER])
GYM_CTRL_CENTER    = (GYM_CTRL_HI + GYM_CTRL_LO) / 2.0
GYM_CTRL_HALFWIDTH = (GYM_CTRL_HI - GYM_CTRL_LO) / 2.0

GYM_CUBE_MASS = 0.15575129663498727
GYM_CUBE_DENSITY = 567
GYM_OBJ_FREEJOINT_DAMPING = 0.01


# ---------------------------------------------------------------------------
# Spec functions: ShadowHand robot + cube object
# ---------------------------------------------------------------------------

def _embed_assets(spec: mujoco.MjSpec) -> None:
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
    spec.default.geom.margin = 0.0
    for class_name in [
        "robot0:asset_class", "robot0:D_Touch", "robot0:DC_Hand",
        "robot0:D_Vizual", "robot0:D_TH_type1", "robot0:D_TH_type2",
    ]:
        dc = spec.find_default(class_name)
        if dc:
            dc.geom.margin = 0.0
    model = spec.compile()
    for i in range(model.ngeom):
        if model.geom_margin[i] != 0.0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name:
                g = spec.geom(name)
                if g:
                    g.margin = 0.0


SHADOWHAND_ACTUATED_JOINTS = GYM_JOINT_ORDER

SHADOWHAND_WRIST_JOINTS = ("robot0:WRJ1", "robot0:WRJ0")
SHADOWHAND_FINGER_JOINTS = tuple(
    j for j in SHADOWHAND_ACTUATED_JOINTS if j not in SHADOWHAND_WRIST_JOINTS
)


def get_shadowhand_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(CUBE_XML))
    for body in list(spec.worldbody.bodies):
        if body.name in ("target", "object", "floor0"):
            spec.delete(body)
    for act in list(spec.actuators):
        spec.delete(act)
    _zero_geom_margins(spec)
    _embed_assets(spec)
    return spec


def _make_shadowhand_articulation() -> EntityArticulationInfoCfg:
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
    cube_stl = (ASSETS_DIR / ".." / "stls" / "hand" / "contactdb_objects" / "cube.stl").resolve()
    assert cube_stl.exists(), f"Missing mesh: {cube_stl}"

    spec = mujoco.MjSpec()
    spec.add_mesh(name="cube_mesh", file=str(cube_stl))
    body = spec.worldbody.add_body(name="cube")
    fj = body.add_freejoint(name="cube_joint")
    fj.damping = np.full(3, GYM_OBJ_FREEJOINT_DAMPING)
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cube_mesh",
        density=GYM_CUBE_DENSITY,
        condim=4,
        rgba=(0.8, 0.2, 0.2, 1.0),
    )
    spec.assets = {os.path.basename(str(cube_stl)): cube_stl.read_bytes()}
    return spec


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention — matches MuJoCo and mjlab)
# ---------------------------------------------------------------------------

def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def compute_rotation_distance(q_achieved: torch.Tensor, q_desired: torch.Tensor) -> torch.Tensor:
    """Gymnasium Robotics rotation distance in radians.

    This follows the installed ShadowHand formula exactly: the scalar part of
    the quaternion difference is clipped to [-1, 1] without applying abs().
    """
    quat_diff = _quat_mul(q_achieved, _quat_conjugate(q_desired))
    return 2.0 * torch.acos(torch.clamp(quat_diff[..., 0], -1.0, 1.0))


def compute_is_success(
    q_achieved: torch.Tensor,
    q_desired: torch.Tensor,
    rotation_threshold: float = 0.1,
) -> torch.Tensor:
    """Per-env success flag matching Gymnasium ManipulateEnv._is_success.

    target_position='ignore' -> position distance not checked.
    target_rotation='z' -> full quaternion distance < rotation_threshold.
    """
    return (compute_rotation_distance(q_achieved, q_desired) < rotation_threshold).float()


def compute_sparse_reward(
    q_achieved: torch.Tensor,
    q_desired: torch.Tensor,
    rotation_threshold: float = 0.1,
) -> torch.Tensor:
    """Sparse reward matching Gymnasium: success - 1 => {0, -1}."""
    return compute_is_success(q_achieved, q_desired, rotation_threshold) - 1.0


# ---------------------------------------------------------------------------
# Custom observation terms
# ---------------------------------------------------------------------------

def object_root_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_pos_w


def object_root_quat(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_quat_w


def object_root_lin_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def object_root_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_ang_vel_w


class desired_goal_obs:
    """Random z-axis target rotation, re-sampled on reset.

    Matches Gymnasium ManipulateEnv with target_rotation='z',
    target_position='ignore'. Returns [pos(3), quat(4)] = 7 dims.
    Stores the target on env._desired_goal_target for reward/success access.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._env = env
        self._target = torch.zeros((env.num_envs, 7), device=env.device)
        env._desired_goal_target = self._target
        self.reset(None)

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        n = len(env_ids)
        cube = self._env.scene["cube"]
        obj_pos = cube.data.root_link_pos_w[env_ids]
        angles = torch.rand(n, device=self._env.device) * 2 * torch.pi - torch.pi
        zeros = torch.zeros(n, device=self._env.device)
        qw = torch.cos(angles / 2)
        target_quat = torch.stack([qw, zeros, zeros, torch.sin(angles / 2)], dim=-1)
        target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True)
        self._target[env_ids] = torch.cat([obj_pos, target_quat], dim=-1)

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return self._target


# ---------------------------------------------------------------------------
# Custom reward term
# ---------------------------------------------------------------------------

class sparse_rotation_reward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env

    def reset(self, env_ids):
        pass

    def __call__(self, env: ManagerBasedRlEnv, rotation_threshold: float = 0.1) -> torch.Tensor:
        target = env._desired_goal_target
        obj_quat = env.scene["cube"].data.root_link_quat_w
        return compute_sparse_reward(obj_quat, target[:, 3:], rotation_threshold)


# ---------------------------------------------------------------------------
# Build mjlab env config
# ---------------------------------------------------------------------------

def _action_scale_offset() -> tuple[dict[str, float], dict[str, float]]:
    """Per-joint scale and offset matching Gymnasium's ctrlrange mapping."""
    scale = {}
    offset = {}
    for _, joint, ctrl_lo, ctrl_hi in GYM_ACTUATOR_ORDER:
        scale[joint] = (ctrl_hi - ctrl_lo) / 2.0
        offset[joint] = (ctrl_hi + ctrl_lo) / 2.0
    return scale, offset


def build_shadowhand_cube_env_cfg(
    *, num_envs: int = 4, auto_reset: bool = False,
) -> ManagerBasedRlEnvCfg:
    cube_pos = (1.0, 0.87, 0.2)
    act_scale, act_offset = _action_scale_offset()

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
                    params={"asset_cfg": SceneEntityCfg("cube")},
                ),
                "object_quat": ObservationTermCfg(
                    func=object_root_quat,
                    params={"asset_cfg": SceneEntityCfg("cube")},
                ),
                "object_lin_vel": ObservationTermCfg(
                    func=object_root_lin_vel,
                    params={"asset_cfg": SceneEntityCfg("cube")},
                ),
                "object_ang_vel": ObservationTermCfg(
                    func=object_root_ang_vel,
                    params={"asset_cfg": SceneEntityCfg("cube")},
                ),
                "desired_goal": ObservationTermCfg(
                    func=desired_goal_obs,
                ),
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
        "reset_cube_pose": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube"),
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
        "cube_fell": TerminationTermCfg(
            func=envs_mdp.root_height_below_minimum,
            params={"minimum_height": 0.05, "asset_cfg": SceneEntityCfg("cube")},
        ),
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
                    articulation=_make_shadowhand_articulation(),
                ),
                "cube": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(
                        pos=cube_pos,
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                    spec_fn=get_cube_spec,
                ),
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


# ---------------------------------------------------------------------------
# Gymnasium baseline (runs in main .venv via subprocess)
# ---------------------------------------------------------------------------

def _run_in_main_venv(script: str, timeout: int = 30) -> str | None:
    import subprocess, json
    main_py = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    if not main_py.exists():
        return None
    r = subprocess.run([str(main_py), "-c", script], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"  subprocess error: {r.stderr[:300]}")
        return None
    for line in r.stdout.strip().split("\n"):
        if line.strip().startswith("{"):
            return line
    return None


def get_gymnasium_baseline():
    import json
    proj = str(Path(__file__).resolve().parent.parent)
    raw = _run_in_main_venv(f"""
import os, sys, json
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, '{proj}')
import gymnasium as gym, numpy as np, dex_envs, mujoco
from jaxrl.envs import FlattenObservationShadowhandWrapper
env = gym.make('cube-rotate-v1', reward_type='sparse')
wrapped = FlattenObservationShadowhandWrapper(env)
obs, _ = wrapped.reset()
e = env.unwrapped; m = e.model
cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'object')
print(json.dumps({{
    "obs_shape": list(obs.shape),
    "action_shape": list(env.action_space.shape),
    "action_range": [float(env.action_space.low.min()), float(env.action_space.high.max())],
    "max_episode_steps": 100,
    "dt": e.dt,
    "sim_timestep": float(m.opt.timestep),
    "n_substeps": int(e.n_substeps),
    "rotation_threshold": float(e.rotation_threshold),
    "cube_mass": float(m.body_mass[cube_bid]),
    "num_joints": 24, "num_actuators": 20,
}}))
env.close()
""")
    return json.loads(raw) if raw else None


# ---------------------------------------------------------------------------
# Parity checks
# ---------------------------------------------------------------------------

def run_checks():
    results = {}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ---------------------------------------------------------------
    # Build env
    # ---------------------------------------------------------------
    print("Building mjlab ShadowHand+cube env (num_envs=4) ...")
    cfg = build_shadowhand_cube_env_cfg(num_envs=4, auto_reset=False)
    env = ManagerBasedRlEnv(cfg, device=device)
    action_dim = env.action_manager.total_action_dim

    print(f"  max_episode_length={env.max_episode_length}  step_dt={env.step_dt}  "
          f"num_envs={env.num_envs}  decimation={cfg.decimation}")
    print()

    # ---------------------------------------------------------------
    # CHECK 1: structural — env builds, obs/action shapes
    # ---------------------------------------------------------------
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    expected_dim = 68
    c1 = (policy_obs.shape == (env.num_envs, expected_dim) and action_dim == 20)
    results["1_structural_shapes"] = c1
    print(f"[CHECK 1] obs={policy_obs.shape} action_dim={action_dim} — {'PASS' if c1 else 'FAIL'}")

    # ---------------------------------------------------------------
    # CHECK 2: step contract — shapes, finite, terminated/truncated
    # ---------------------------------------------------------------
    action = torch.zeros(env.num_envs, action_dim, device=device)
    obs_buf, rew, term, trunc, _ = env.step(action)
    step_obs = obs_buf["policy"]
    c2 = (step_obs.shape == (4, 68) and rew.shape == (4,)
           and term.shape == (4,) and trunc.shape == (4,)
           and torch.isfinite(step_obs).all())
    results["2_step_contract"] = bool(c2)
    print(f"[CHECK 2] step shapes/finite — {'PASS' if c2 else 'FAIL'}")

    # ---------------------------------------------------------------
    # CHECK 3: episode length and timeout
    # ---------------------------------------------------------------
    obs, _ = env.reset()
    timeout_hit = False
    for step_i in range(1, env.max_episode_length + 5):
        act = torch.randn(env.num_envs, action_dim, device=device) * 0.01
        _, _, term, trunc, _ = env.step(act)
        if trunc.any():
            timeout_hit = True
            terms_np = term.cpu().numpy().astype(float)
            truns_np = trunc.cpu().numpy().astype(float)
            masks = 1 - (terms_np * (1 - truns_np))
            timeout_mask_ok = all(masks[i] == 1.0 for i in range(len(truns_np)) if truns_np[i])
            break
        if (term | trunc).any():
            env.reset(env_ids=(term | trunc).nonzero(as_tuple=False).squeeze(-1))
    c3 = timeout_hit and timeout_mask_ok and step_i == env.max_episode_length
    results["3_timeout_mask"] = c3
    print(f"[CHECK 3] timeout at step {step_i}, mask=1 — {'PASS' if c3 else 'FAIL'}")

    # ---------------------------------------------------------------
    # CHECK 4: action scaling parity
    # ---------------------------------------------------------------
    env.reset()
    robot = env.scene["robot"]
    mjlab_joint_names = robot.joint_names
    mjlab_act_names = robot.actuator_names

    # Find the ordering of the 20 actuated joints within the full 24
    actuated_joint_indices = []
    for jn in GYM_JOINT_ORDER:
        idx = mjlab_joint_names.index(jn)
        actuated_joint_indices.append(idx)

    order_ok = (len(actuated_joint_indices) == 20)
    if order_ok:
        for i, (_, gj, _, _) in enumerate(GYM_ACTUATOR_ORDER):
            if mjlab_joint_names[actuated_joint_indices[i]] != gj:
                order_ok = False
                break

    max_diffs = []
    for test_val in [-1.0, 0.0, 1.0]:
        gym_targets = GYM_CTRL_CENTER + test_val * GYM_CTRL_HALFWIDTH
        act_tensor = torch.full((env.num_envs, 20), test_val, device=device)
        env.action_manager._terms["joint_pos"].process_actions(act_tensor)
        mjlab_targets = env.action_manager._terms["joint_pos"]._processed_actions[0].cpu().numpy()
        diff = np.abs(gym_targets - mjlab_targets)
        max_diffs.append(diff.max())

    rng = np.random.default_rng(42)
    rand_actions = rng.uniform(-1, 1, size=(5, 20))
    for ra in rand_actions:
        gym_targets = GYM_CTRL_CENTER + ra * GYM_CTRL_HALFWIDTH
        act_tensor = torch.tensor(ra, dtype=torch.float32, device=device).unsqueeze(0).expand(env.num_envs, -1)
        env.action_manager._terms["joint_pos"].process_actions(act_tensor)
        mjlab_targets = env.action_manager._terms["joint_pos"]._processed_actions[0].cpu().numpy()
        diff = np.abs(gym_targets - mjlab_targets)
        max_diffs.append(diff.max())

    worst_action_diff = max(max_diffs)
    c4 = order_ok and worst_action_diff < 1e-5
    results["4_action_scaling"] = c4
    print(f"[CHECK 4] action scaling: order_ok={order_ok} worst_diff={worst_action_diff:.2e} — {'PASS' if c4 else 'FAIL'}")

    # ---------------------------------------------------------------
    # CHECK 5: object physics — mass, freejoint damping
    # ---------------------------------------------------------------
    cube_entity = env.scene["cube"]
    body_ids_local, _ = cube_entity.find_bodies("cube", preserve_order=True)
    body_world_id = int(cube_entity.indexing.body_ids[body_ids_local[0]].item())
    mjlab_cube_mass = float(env.sim.model.body_mass[0, body_world_id].item())
    mass_diff = abs(mjlab_cube_mass - GYM_CUBE_MASS)
    mass_ok = mass_diff < 0.001

    # Free joints are not in entity.joint_names; query the model directly.
    njnt = env.sim.model.njnt
    damp_ok = False
    mjlab_damp = -1.0
    for ji in range(njnt):
        jname = env.sim.model.jnt_names[ji] if hasattr(env.sim.model, 'jnt_names') else ""
        jtype = int(env.sim.model.jnt_type[ji].item()) if env.sim.model.jnt_type.ndim == 1 else int(env.sim.model.jnt_type[0, ji].item())
        if jtype == 0:  # mjJNT_FREE
            dof_adr = int(env.sim.model.jnt_dofadr[ji].item())
            mjlab_damp = float(env.sim.model.dof_damping[0, dof_adr].item())
            damp_ok = abs(mjlab_damp - GYM_OBJ_FREEJOINT_DAMPING) < 1e-6
            break

    c5 = mass_ok and damp_ok
    results["5_object_physics"] = c5
    print(f"[CHECK 5] cube mass: mjlab={mjlab_cube_mass:.6f} gym={GYM_CUBE_MASS:.6f} "
          f"diff={mass_diff:.6f}  damp={mjlab_damp} — {'PASS' if c5 else 'FAIL'}")

    # ---------------------------------------------------------------
    # CHECK 6: reset randomization — position + yaw vary
    # ---------------------------------------------------------------
    positions = []
    yaws = []
    for _ in range(50):
        env.reset()
        cube = env.scene["cube"]
        pos = cube.data.root_link_pos_w[0].cpu().numpy()
        quat = cube.data.root_link_quat_w[0].cpu().numpy()
        # yaw from quat: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        w, x, y, z = quat
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        positions.append(pos)
        yaws.append(yaw)

    positions = np.array(positions)
    yaws = np.array(yaws)
    pos_range_x = positions[:, 0].max() - positions[:, 0].min()
    pos_range_y = positions[:, 1].max() - positions[:, 1].min()
    yaw_range = yaws.max() - yaws.min()

    pos_varies = pos_range_x > 0.005 and pos_range_y > 0.005
    yaw_varies = yaw_range > 3.0
    c6 = pos_varies and yaw_varies
    results["6_reset_randomization"] = c6
    print(f"[CHECK 6] reset rand: pos_range_x={pos_range_x:.4f} pos_range_y={pos_range_y:.4f} "
          f"yaw_range={yaw_range:.2f} rad — {'PASS' if c6 else 'FAIL'}")

    # ---------------------------------------------------------------
    # CHECK 7: reward/success formula — synthetic deterministic tests
    # ---------------------------------------------------------------
    print("[CHECK 7] reward/success formula tests:")
    c7_all = True

    def _check(label, q_ach, q_des, expect_rew, expect_suc):
        nonlocal c7_all
        qa = torch.tensor([q_ach], dtype=torch.float32)
        qd = torch.tensor([q_des], dtype=torch.float32)
        r = compute_sparse_reward(qa, qd, 0.1).item()
        s = compute_is_success(qa, qd, 0.1).item()
        ok = abs(r - expect_rew) < 1e-6 and abs(s - expect_suc) < 1e-6
        if not ok:
            c7_all = False
        print(f"  {label}: reward={r:.1f} success={s:.1f} — {'PASS' if ok else 'FAIL'}")

    # Exact match
    _check("exact_match",
           [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
           0.0, 1.0)

    # Gymnasium's source formula does not canonicalize antipodal signs with abs().
    _check("antipodal_quaternion_gym_formula",
           [1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0],
           -1.0, 0.0)

    # Large rotation error (pi radians)
    _check("pi_rotation_error",
           [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
           -1.0, 0.0)

    # Just below threshold: 0.09 rad z-rotation
    angle_below = 0.09
    q_below = [np.cos(angle_below/2), 0.0, 0.0, np.sin(angle_below/2)]
    _check("just_below_threshold (0.09 rad)",
           [1.0, 0.0, 0.0, 0.0], q_below,
           0.0, 1.0)

    # Just above threshold: 0.11 rad z-rotation
    angle_above = 0.11
    q_above = [np.cos(angle_above/2), 0.0, 0.0, np.sin(angle_above/2)]
    _check("just_above_threshold (0.11 rad)",
           [1.0, 0.0, 0.0, 0.0], q_above,
           -1.0, 0.0)

    results["7_reward_success_formula"] = c7_all

    # ---------------------------------------------------------------
    # CHECK 8: success helper works with live env state
    # ---------------------------------------------------------------
    env.reset()
    obj_quat = env.scene["cube"].data.root_link_quat_w
    target = env._desired_goal_target[:, 3:]
    is_suc = compute_is_success(obj_quat, target, 0.1)
    c8 = is_suc.shape == (env.num_envs,) and torch.isfinite(is_suc).all()
    results["8_success_helper_live"] = bool(c8)
    print(f"[CHECK 8] success helper live: shape={is_suc.shape} values={is_suc.cpu().tolist()} — {'PASS' if c8 else 'FAIL'}")

    # ---------------------------------------------------------------
    # CHECK 9: sparse reward values
    # ---------------------------------------------------------------
    env.reset()
    rewards_collected = []
    for _ in range(30):
        act = torch.randn(env.num_envs, action_dim, device=device) * 0.01
        _, rew, term, trunc, _ = env.step(act)
        rewards_collected.append(rew.cpu())
        if (term | trunc).any():
            env.reset(env_ids=(term | trunc).nonzero(as_tuple=False).squeeze(-1))
    all_rew = torch.cat(rewards_collected)
    unique = sorted(all_rew.unique().tolist())
    c9 = all(v in (-1.0, 0.0) for v in unique)
    results["9_reward_values"] = c9
    print(f"[CHECK 9] reward unique values: {unique} — {'PASS' if c9 else 'FAIL'}")

    # ---------------------------------------------------------------
    # CHECK 10: episode length
    # ---------------------------------------------------------------
    c10 = int(cfg.episode_length_s / env.step_dt) == 100
    results["10_episode_length"] = c10
    print(f"[CHECK 10] episode_length={int(cfg.episode_length_s / env.step_dt)} — {'PASS' if c10 else 'FAIL'}")

    env.close()
    return results


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def benchmark_gymnasium(num_steps=500):
    import json
    proj = str(Path(__file__).resolve().parent.parent)
    raw = _run_in_main_venv(f"""
import os, sys, time, json
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, '{proj}')
import gymnasium as gym, dex_envs
env = gym.make('cube-rotate-v1', reward_type='sparse')
obs, _ = env.reset()
for _ in range(50):
    obs, r, t, tr, info = env.step(env.action_space.sample())
    if t or tr: obs, _ = env.reset()
t0 = time.perf_counter()
for _ in range({num_steps}):
    obs, r, t, tr, info = env.step(env.action_space.sample())
    if t or tr: obs, _ = env.reset()
dt = time.perf_counter() - t0
env.close()
sps = {num_steps} / dt
print(json.dumps({{"steps": {num_steps}, "time_s": dt, "steps_per_sec": sps, "env_transitions_per_sec": sps}}))
""", timeout=120)
    return json.loads(raw) if raw else None


def benchmark_mjlab(num_envs=64, num_steps=500):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = build_shadowhand_cube_env_cfg(num_envs=num_envs, auto_reset=False)
    env = ManagerBasedRlEnv(cfg, device=device)
    dim = env.action_manager.total_action_dim
    env.reset()
    for _ in range(50):
        _, _, t, tr, _ = env.step(torch.randn(num_envs, dim, device=device) * 0.01)
        if (t | tr).any():
            env.reset(env_ids=(t | tr).nonzero(as_tuple=False).squeeze(-1))
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_steps):
        _, _, t, tr, _ = env.step(torch.randn(num_envs, dim, device=device) * 0.01)
        if (t | tr).any():
            env.reset(env_ids=(t | tr).nonzero(as_tuple=False).squeeze(-1))
    if device != "cpu":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    env.close()
    return {"num_envs": num_envs, "steps": num_steps, "time_s": dt,
            "steps_per_sec": num_steps / dt,
            "env_transitions_per_sec": num_steps * num_envs / dt}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import importlib.metadata
    print("=" * 70)
    print("Single-Object ShadowHand mjlab Parity Prototype")
    print("=" * 70)
    print(f"mjlab={importlib.metadata.version('mjlab')}  "
          f"torch={torch.__version__}  mujoco={mujoco.__version__}  "
          f"CUDA={'yes: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")
    print()

    # Gymnasium baseline
    print("-" * 70)
    print("Gymnasium Baseline")
    print("-" * 70)
    baseline = get_gymnasium_baseline()
    if baseline:
        for k, v in baseline.items():
            print(f"  {k}: {v}")
    else:
        print("  (could not load)")
    print()

    # Parity checks
    print("-" * 70)
    print("Parity Checks")
    print("-" * 70)
    try:
        results = run_checks()
    except Exception:
        traceback.print_exc()
        results = {}
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for key, val in results.items():
        status = "PASS" if val else "FAIL"
        if not val:
            all_pass = False
        print(f"  [{status}] {key}")
    print()
    print("ALL CHECKS PASSED" if all_pass else f"FAILED: {[k for k, v in results.items() if not v]}")
    print()

    # Throughput
    print("-" * 70)
    print("Throughput")
    print("-" * 70)
    gym_bench = mj_bench = None
    try:
        print("Gymnasium (1 env, 500 steps) ...")
        gym_bench = benchmark_gymnasium()
        if gym_bench:
            print(f"  {gym_bench['steps_per_sec']:.0f} steps/sec")
    except Exception as e:
        print(f"  failed: {e}")

    try:
        for ne in [64]:
            print(f"mjlab ({ne} envs, 500 steps) ...")
            mj_bench = benchmark_mjlab(num_envs=ne)
            print(f"  {mj_bench['steps_per_sec']:.0f} steps/sec  "
                  f"({mj_bench['env_transitions_per_sec']:.0f} transitions/sec)")
    except Exception as e:
        print(f"  failed: {e}")
        traceback.print_exc()

    if gym_bench and mj_bench:
        su = mj_bench['env_transitions_per_sec'] / gym_bench['env_transitions_per_sec']
        print(f"\nSpeedup: {su:.1f}x  (mjlab {mj_bench['num_envs']} envs vs Gymnasium 1 env)")
    print()

    # Remaining approximations
    print("-" * 70)
    print("Remaining Approximations (not claimed as parity)")
    print("-" * 70)
    print("""
  A. GEOM MARGINS: zeroed for MuJoCo Warp compatibility (was 0.0005).
  B. ACTUATOR MODEL: Gymnasium uses MuJoCo general actuators (P-control,
     Kd=0). mjlab uses IdealPd with stiffness matched but Kd=0 (matched).
     effort_limits are rounded; forcerange is not identical.
  C. JOINT COUPLING: FFJ0/MFJ0/RFJ0/LFJ0 coupled to J1 via equality
     constraints. Preserved from the loaded ShadowHand spec.
  D. PyTorch-to-JAX TRANSFER: not profiled.
  E. RESET POSITION RANGE: mjlab uses ±0.01 for x/y; Gymnasium has
     slightly different randomization via its own reset logic (wider
     effective range due to sim settling). Not semantically critical.
""")

    return all_pass


if __name__ == "__main__":
    success = False
    try:
        success = main()
    except Exception:
        traceback.print_exc()
    sys.exit(0 if success else 1)
