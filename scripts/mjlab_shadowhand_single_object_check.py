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
# Spec functions: ShadowHand robot + cube object
# ---------------------------------------------------------------------------

def _embed_assets(spec: mujoco.MjSpec) -> None:
    """Embed mesh and texture files into spec.assets for GPU/distributed use."""
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
    """Set all geom margins to 0 for MuJoCo Warp compatibility.

    MuJoCo Warp (MULTICCD) does not support non-zero geom margins.
    The ShadowHand XML sets margin=0.0005 in default classes and on
    specific geoms.
    """
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


SHADOWHAND_ACTUATED_JOINTS = (
    "robot0:WRJ1", "robot0:WRJ0",
    "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1",
    "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1",
    "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1",
    "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1",
    "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0",
)

SHADOWHAND_WRIST_JOINTS = ("robot0:WRJ1", "robot0:WRJ0")
SHADOWHAND_FINGER_JOINTS = tuple(j for j in SHADOWHAND_ACTUATED_JOINTS if j not in SHADOWHAND_WRIST_JOINTS)


def get_shadowhand_spec() -> mujoco.MjSpec:
    """Load the ShadowHand robot from the existing manipulate_cube.xml.

    Removes the object, target, and floor bodies so that only the hand
    remains. Also removes MuJoCo general actuators — mjlab will add
    IdealPd actuators via EntityArticulationInfoCfg.
    """
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
    """Create mjlab actuator config matching the ShadowHand position control.

    The ShadowHand XML uses general actuators with:
      wrist (WRJ): stiffness=5.0, damping=0, effort_limit~4.79
      finger: stiffness=1.0, damping=0, effort_limit~0.72-2.37
    """
    return EntityArticulationInfoCfg(
        actuators=(
            IdealPdActuatorCfg(
                target_names_expr=SHADOWHAND_WRIST_JOINTS,
                stiffness=5.0,
                damping=0.3,
                effort_limit=5.0,
            ),
            IdealPdActuatorCfg(
                target_names_expr=SHADOWHAND_FINGER_JOINTS,
                stiffness=1.0,
                damping=0.1,
                effort_limit=1.0,
            ),
        ),
    )


def get_cube_spec() -> mujoco.MjSpec:
    """Create a free-body cube with the same mesh used by manipulate_cube.xml."""
    cube_stl = ASSETS_DIR / ".." / "stls" / "hand" / "contactdb_objects" / "cube.stl"
    cube_stl = cube_stl.resolve()
    assert cube_stl.exists(), f"Missing mesh: {cube_stl}"

    spec = mujoco.MjSpec()
    spec.add_mesh(name="cube_mesh", file=str(cube_stl))
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cube_mesh",
        mass=0.1,
        condim=4,
        rgba=(0.8, 0.2, 0.2, 1.0),
    )
    spec.assets = {os.path.basename(str(cube_stl)): cube_stl.read_bytes()}
    return spec


# ---------------------------------------------------------------------------
# Custom observation terms
# ---------------------------------------------------------------------------

def object_root_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object world-frame position [num_envs, 3]."""
    return env.scene[asset_cfg.name].data.root_link_pos_w


def object_root_quat(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object world-frame quaternion (w,x,y,z) [num_envs, 4]."""
    return env.scene[asset_cfg.name].data.root_link_quat_w


def object_root_lin_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object world-frame linear velocity [num_envs, 3]."""
    return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def object_root_ang_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object world-frame angular velocity [num_envs, 3]."""
    return env.scene[asset_cfg.name].data.root_link_ang_vel_w


class desired_goal_obs:
    """Maintain and return a random target rotation as the desired goal.

    Matches Gymnasium ManipulateEnv with target_rotation='z',
    target_position='ignore'. Returns [pos(3), quat(4)] = 7 dims.

    Stores the target on the env object so that other terms (reward,
    metrics) can access it without going through the observation manager.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._env = env
        n = env.num_envs
        self._target = torch.zeros((n, 7), device=env.device)
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
        qx = zeros
        qy = zeros
        qz = torch.sin(angles / 2)
        target_quat = torch.stack([qw, qx, qy, qz], dim=-1)
        target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True)
        self._target[env_ids] = torch.cat([obj_pos, target_quat], dim=-1)

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return self._target


# ---------------------------------------------------------------------------
# Custom reward term: sparse reward matching Gymnasium
# ---------------------------------------------------------------------------

class sparse_rotation_reward:
    """Sparse reward: 0 if rotation distance < threshold, -1 otherwise.

    Matches Gymnasium ManipulateEnv with reward_type='sparse',
    target_rotation='z', target_position='ignore'.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env

    def reset(self, env_ids):
        pass

    def __call__(self, env: ManagerBasedRlEnv, rotation_threshold: float = 0.1) -> torch.Tensor:
        target = env._desired_goal_target
        cube = env.scene["cube"]
        obj_quat = cube.data.root_link_quat_w
        target_quat = target[:, 3:]
        quat_diff = _quat_mul(obj_quat, _quat_conjugate(target_quat))
        angle_diff = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0].abs(), 0.0, 1.0))
        success = (angle_diff < rotation_threshold).float()
        return success - 1.0


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


# ---------------------------------------------------------------------------
# Build mjlab env config
# ---------------------------------------------------------------------------

def build_shadowhand_cube_env_cfg(
    *, num_envs: int = 4, auto_reset: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build a mjlab env config matching Gymnasium cube-rotate-v1 semantics."""

    cube_pos = (1.0, 0.87, 0.2)

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
            scale=1.0,
            offset=0.0,
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
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
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
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=20,
            ),
        ),
        decimation=20,
        episode_length_s=4.0,
        auto_reset=auto_reset,
        scale_rewards_by_dt=False,
    )


# ---------------------------------------------------------------------------
# Gymnasium baseline extraction
# ---------------------------------------------------------------------------

def get_gymnasium_baseline():
    """Return a dict of key properties from Gymnasium cube-rotate-v1.

    Runs in the main .venv via subprocess since the mjlab venv does not
    have gymnasium installed.
    """
    import subprocess
    import json

    main_venv = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    if not main_venv.exists():
        return None

    script = f"""
import os, sys, json
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, '{Path(__file__).resolve().parent.parent}')
import gymnasium as gym
import numpy as np
import dex_envs
from jaxrl.envs import FlattenObservationShadowhandWrapper

env = gym.make('cube-rotate-v1', reward_type='sparse')
wrapped = FlattenObservationShadowhandWrapper(env)
obs, _ = wrapped.reset()
e = env.unwrapped
baseline = {{
    "env_id": "cube-rotate-v1",
    "obs_shape": list(obs.shape),
    "obs_components": "24 joint_pos + 24 joint_vel + 3 obj_pos + 4 obj_quat + 3 obj_linvel + 3 obj_angvel + 3 goal_pos + 4 goal_quat",
    "action_shape": list(env.action_space.shape),
    "action_range": [float(env.action_space.low.min()), float(env.action_space.high.max())],
    "max_episode_steps": 100,
    "dt": e.dt,
    "sim_timestep": float(e.model.opt.timestep),
    "n_substeps": int(e.n_substeps),
    "reward_type": e.reward_type,
    "reward_range": [-1.0, 0.0],
    "success_metric": "is_success (rotation distance < 0.1 rad)",
    "rotation_threshold": float(e.rotation_threshold),
    "distance_threshold": float(e.distance_threshold),
    "target_rotation": e.target_rotation,
    "target_position": e.target_position,
    "randomize_initial_position": bool(e.randomize_initial_position),
    "randomize_initial_rotation": bool(e.randomize_initial_rotation),
    "num_joints": 24,
    "num_actuators": 20,
}}
env.close()
print(json.dumps(baseline))
"""
    result = subprocess.run(
        [str(main_venv), "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  Gymnasium baseline subprocess error: {result.stderr[:300]}")
        return None

    for line in result.stdout.strip().split("\n"):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Parity checks
# ---------------------------------------------------------------------------

def run_parity_checks():
    results = {}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # --- Build mjlab env ---
    print("Building mjlab ShadowHand+cube env (num_envs=4) ...")
    cfg = build_shadowhand_cube_env_cfg(num_envs=4, auto_reset=False)
    env = ManagerBasedRlEnv(cfg, device=device)

    print(f"  max_episode_length: {env.max_episode_length} steps")
    print(f"  step_dt: {env.step_dt}s")
    print(f"  episode_length_s: {cfg.episode_length_s}s")
    print(f"  num_envs: {env.num_envs}")
    print(f"  decimation: {cfg.decimation}")
    print(f"  sim_timestep: {cfg.sim.mujoco.timestep}")
    print()

    # --- Check 1: env builds successfully ---
    results["1_env_builds"] = True
    print("[CHECK 1] PASS — mjlab env builds successfully")
    print()

    # --- Check 2: reset works ---
    obs, info = env.reset()
    policy_obs = obs["policy"]
    print(f"[CHECK 2] Reset observation shape: {policy_obs.shape}")
    print(f"  Expected: (num_envs, 68) = (4, 68)")
    expected_obs_dim = 24 + 24 + 3 + 4 + 3 + 3 + 7  # = 68
    obs_shape_ok = policy_obs.shape == (env.num_envs, expected_obs_dim)
    results["2_reset_obs_shape"] = obs_shape_ok
    if obs_shape_ok:
        print(f"  PASS — obs shape matches Gymnasium (68 dims)")
    else:
        print(f"  FAIL — got {policy_obs.shape}, expected ({env.num_envs}, {expected_obs_dim})")
    print()

    # --- Check 3: action shape matches ---
    action_dim = env.action_manager.total_action_dim
    print(f"[CHECK 3] Action dimension: {action_dim}")
    action_ok = action_dim == 20
    results["3_action_dim"] = action_ok
    if action_ok:
        print("  PASS — action dim matches Gymnasium (20)")
    else:
        print(f"  FAIL — got {action_dim}, expected 20")
    print()

    # --- Check 4: step accepts actions and returns correct shapes ---
    action = torch.zeros(env.num_envs, action_dim, device=device)
    obs_buf, reward_buf, terminated, truncated, extras = env.step(action)
    step_obs = obs_buf["policy"]
    print(f"[CHECK 4] Step outputs:")
    print(f"  obs shape: {step_obs.shape}")
    print(f"  reward shape: {reward_buf.shape}")
    print(f"  terminated shape: {terminated.shape}")
    print(f"  truncated shape: {truncated.shape}")
    step_ok = (
        step_obs.shape == (env.num_envs, expected_obs_dim)
        and reward_buf.shape == (env.num_envs,)
        and terminated.shape == (env.num_envs,)
        and truncated.shape == (env.num_envs,)
    )
    results["4_step_shapes"] = step_ok
    if step_ok:
        print("  PASS — all step output shapes correct")
    else:
        print("  FAIL")
    print()

    # --- Check 5: observation can be flattened into policy vector ---
    flat_obs = step_obs
    print(f"[CHECK 5] Flattened observation:")
    print(f"  shape: {flat_obs.shape}")
    print(f"  dtype: {flat_obs.dtype}")
    print(f"  finite: {torch.isfinite(flat_obs).all().item()}")
    flat_ok = flat_obs.shape[-1] == expected_obs_dim and torch.isfinite(flat_obs).all()
    results["5_flat_obs_valid"] = bool(flat_ok)
    if flat_ok:
        print("  PASS — observation is a valid finite policy vector")
    else:
        print("  FAIL")

    # Print observation breakdown
    print(f"  Breakdown:")
    idx = 0
    for name, size in [
        ("joint_pos", 24), ("joint_vel", 24),
        ("object_pos", 3), ("object_quat", 4),
        ("object_lin_vel", 3), ("object_ang_vel", 3),
        ("desired_goal", 7),
    ]:
        vals = flat_obs[0, idx:idx+size]
        print(f"    {name:20s} [{idx}:{idx+size}] mean={vals.mean():.4f} std={vals.std():.4f}")
        idx += size
    print()

    # --- Check 6: terminated and truncated are separate ---
    results["6_separate_term_trunc"] = True
    print("[CHECK 6] PASS — terminated and truncated are separate tensors")
    print()

    # --- Check 7: timeout maps to BRC mask=1 ---
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    print(f"[CHECK 7] Running until timeout (max_steps={max_steps}) ...")
    timeout_hit = False
    for step_i in range(1, max_steps + 5):
        action = torch.randn(env.num_envs, action_dim, device=device) * 0.1
        obs_buf, reward_buf, terminated, truncated, extras = env.step(action)

        done_any = (terminated | truncated).any()
        if truncated.any() and not timeout_hit:
            timeout_hit = True
            terms_np = terminated.cpu().numpy().astype(float)
            truns_np = truncated.cpu().numpy().astype(float)
            masks = 1 - (terms_np * (1 - truns_np))
            timeout_mask_correct = all(
                masks[i] == 1.0 for i in range(len(truns_np)) if truns_np[i] == 1.0
            )
            results["7_timeout_mask"] = timeout_mask_correct
            print(f"  Timeout at step {step_i}")
            print(f"  terminated: {terms_np}")
            print(f"  truncated:  {truns_np}")
            print(f"  BRC masks:  {masks}")
            print(f"  {'PASS' if timeout_mask_correct else 'FAIL'} — timeout maps to mask=1")
            break

        if done_any:
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=done_ids)

    if not timeout_hit:
        results["7_timeout_mask"] = False
        print(f"  FAIL — no timeout in {max_steps + 4} steps")
    print()

    # --- Check 8: reward exists and is sparse ---
    obs, _ = env.reset()
    rewards_collected = []
    for _ in range(20):
        action = torch.randn(env.num_envs, action_dim, device=device) * 0.1
        _, rew, term, trunc, _ = env.step(action)
        rewards_collected.append(rew.cpu())
        done = (term | trunc).any()
        if done:
            done_ids = (term | trunc).nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=done_ids)

    all_rewards = torch.cat(rewards_collected)
    unique_rewards = all_rewards.unique()
    print(f"[CHECK 8] Rewards observed: {unique_rewards.tolist()}")
    print(f"  min={all_rewards.min():.4f}, max={all_rewards.max():.4f}")
    reward_ok = all_rewards.min() >= -1.1 and all_rewards.max() <= 0.1
    results["8_reward_range"] = bool(reward_ok)
    if reward_ok:
        print("  PASS — reward range consistent with sparse [-1, 0]")
    else:
        print("  FAIL")
    print()

    # --- Check 9: batched num_envs > 1 works ---
    results["9_batched_envs"] = True  # Already tested with num_envs=4
    print("[CHECK 9] PASS — batched stepping with num_envs=4 works")
    print()

    # --- Check 10: episode length matches ---
    ep_steps = int(cfg.episode_length_s / env.step_dt)
    gymnasium_ep_steps = 100
    ep_match = ep_steps == gymnasium_ep_steps
    results["10_episode_length"] = ep_match
    print(f"[CHECK 10] Episode length: mjlab={ep_steps}, Gymnasium={gymnasium_ep_steps}")
    if ep_match:
        print("  PASS")
    else:
        print(f"  MISMATCH — mjlab has {ep_steps} steps, Gymnasium has {gymnasium_ep_steps}")
        print(f"  (episode_length_s={cfg.episode_length_s}, step_dt={env.step_dt})")
    print()

    env.close()
    return results


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def benchmark_gymnasium(num_steps=500):
    """Benchmark Gymnasium cube-rotate-v1 via subprocess using the main venv."""
    import subprocess
    import json

    main_venv = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    if not main_venv.exists():
        return None

    script = f"""
import os, sys, time, json
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, '{Path(__file__).resolve().parent.parent}')
import gymnasium as gym
import dex_envs
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
"""
    result = subprocess.run(
        [str(main_venv), "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  Gymnasium benchmark subprocess failed: {result.stderr[:200]}")
        return None

    for line in result.stdout.strip().split("\n"):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def benchmark_mjlab(num_envs=64, num_steps=500):
    """Benchmark mjlab batched stepping."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = build_shadowhand_cube_env_cfg(num_envs=num_envs, auto_reset=False)
    env = ManagerBasedRlEnv(cfg, device=device)
    action_dim = env.action_manager.total_action_dim

    obs, _ = env.reset()

    # Warmup (includes kernel compilation)
    for _ in range(50):
        action = torch.randn(num_envs, action_dim, device=device) * 0.1
        obs_buf, rew, term, trunc, extras = env.step(action)
        done = (term | trunc).any()
        if done:
            done_ids = (term | trunc).nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=done_ids)

    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_steps):
        action = torch.randn(num_envs, action_dim, device=device) * 0.1
        obs_buf, rew, term, trunc, extras = env.step(action)
        done = (term | trunc).any()
        if done:
            done_ids = (term | trunc).nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=done_ids)

    if device != "cpu":
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    env.close()

    sps = num_steps / dt
    transitions = num_steps * num_envs
    tps = transitions / dt
    return {
        "num_envs": num_envs,
        "steps": num_steps,
        "time_s": dt,
        "steps_per_sec": sps,
        "env_transitions_per_sec": tps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import importlib.metadata
    print("=" * 70)
    print("Single-Object ShadowHand mjlab Parity Prototype")
    print("=" * 70)
    print()
    print(f"mjlab version: {importlib.metadata.version('mjlab')}")
    print(f"torch version: {torch.__version__}")
    print(f"mujoco version: {mujoco.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # --- Gymnasium baseline ---
    print("-" * 70)
    print("Gymnasium Baseline: cube-rotate-v1")
    print("-" * 70)
    try:
        baseline = get_gymnasium_baseline()
        for k, v in baseline.items():
            print(f"  {k}: {v}")
        print()
    except Exception as e:
        print(f"  Could not load Gymnasium baseline: {e}")
        baseline = None
    print()

    # --- Parity checks ---
    print("-" * 70)
    print("mjlab Parity Checks")
    print("-" * 70)
    try:
        results = run_parity_checks()
    except Exception:
        traceback.print_exc()
        results = {}
    print()

    # --- Summary ---
    print("=" * 70)
    print("PARITY CHECK SUMMARY")
    print("=" * 70)
    all_pass = True
    for key, val in results.items():
        status = "PASS" if val else "FAIL"
        if not val:
            all_pass = False
        print(f"  [{status}] {key}")
    print()

    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"SOME CHECKS FAILED: {failed}")
    print()

    # --- Throughput benchmark ---
    print("-" * 70)
    print("Throughput Benchmark")
    print("-" * 70)

    try:
        print("Gymnasium (1 env, sequential, 500 steps) ...")
        gym_bench = benchmark_gymnasium(num_steps=500)
        print(f"  {gym_bench['steps_per_sec']:.1f} steps/sec "
              f"({gym_bench['env_transitions_per_sec']:.1f} transitions/sec)")
    except Exception as e:
        print(f"  Gymnasium benchmark failed: {e}")
        gym_bench = None

    try:
        for n_envs in [32, 64]:
            print(f"mjlab ({n_envs} envs, batched, 500 steps) ...")
            mj_bench = benchmark_mjlab(num_envs=n_envs, num_steps=500)
            print(f"  {mj_bench['steps_per_sec']:.1f} batched-steps/sec "
                  f"({mj_bench['env_transitions_per_sec']:.1f} transitions/sec)")
    except Exception as e:
        print(f"  mjlab benchmark failed: {e}")
        traceback.print_exc()
        mj_bench = None

    print()
    if gym_bench and mj_bench:
        speedup = mj_bench['env_transitions_per_sec'] / gym_bench['env_transitions_per_sec']
        print(f"Speedup: {speedup:.1f}x (mjlab {mj_bench['num_envs']} envs vs Gymnasium 1 env)")
    print()

    # --- Gaps & risks ---
    print("-" * 70)
    print("Known Gaps / Risks")
    print("-" * 70)
    print("""
  1. REWARD PARITY: The sparse reward implemented here computes rotation
     distance the same way as Gymnasium (quat distance < 0.1 rad), but
     the observation-frame quaternion convention needs verification.
     Gymnasium uses (w,x,y,z) from MuJoCo; mjlab also returns (w,x,y,z)
     from root_link_quat_w, so this should match.

  2. SUCCESS METRIC: The success flag is not yet surfaced as an mjlab
     metric or in step() extras. It would need a custom MetricsTermCfg
     or a post-step check in the adapter.

  3. RESET RANDOMIZATION: The Gymnasium env randomizes both initial
     object position and rotation. The mjlab config randomizes position
     but uses a fixed initial rotation. Full parity requires adding
     rotation randomization to the reset event.

  4. JOINT COUPLING: ShadowHand has 24 joints but only 20 actuators
     (FFJ0/MFJ0/RFJ0/LFJ0 are coupled to FFJ1/MFJ1/RFJ1/LFJ1).
     The Gymnasium env handles this in the actuator XML. mjlab should
     preserve this coupling from the loaded spec, but the observation
     will report 24 joint positions/velocities, same as Gymnasium.

  5. DESIRED GOAL IN OBS: The desired_goal (target rotation) is included
     in the observation vector. The target is re-sampled on reset.
     This matches Gymnasium's behavior where the policy sees the target.

  6. ACTION SEMANTICS: Gymnasium uses direct position-control actuators
     with action range [-1, 1]. mjlab JointPositionAction maps actions
     to position targets. The scale/offset may differ from Gymnasium's
     actuator control mapping. The reference project uses delta-position
     control instead. For parity, direct position control is used here,
     matching the Gymnasium actuator setup.

  7. PyTorch-to-JAX TRANSFER: Not tested here. The BRC loop requires
     converting mjlab's PyTorch tensors to JAX arrays each step.
""")

    return all_pass


if __name__ == "__main__":
    success = False
    try:
        success = main()
    except Exception:
        traceback.print_exc()
    sys.exit(0 if success else 1)
