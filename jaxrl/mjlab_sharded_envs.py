"""ShardedMjlabParallelEnv — sharded homogeneous-shard mjlab adapter.

Instead of one batched ManagerBasedRlEnv with N mesh variants (which causes
mujoco_warp SDF warp divergence), this adapter creates one single-variant
ManagerBasedRlEnv per object and steps them sequentially. Each shard runs
homogeneous physics, so GPU utilization is high within each shard.

Presents the same API as MjlabParallelEnv:
  .reset()              -> np.ndarray (total_envs, obs_dim)
  .step(actions)        -> (obs, rewards, terms, truns, goals)
  .generate_masks(t, r) -> masks
  .reset_where_done(...)-> (obs, terms, truns)
  .observation_space    -> gym.spaces.Box (total_envs, obs_dim)
  .action_space         -> gym.spaces.Box (total_envs, act_dim)
  .num_tasks            -> int
  .envs                 -> list
"""

from __future__ import annotations

import numpy as np
import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from jaxrl.mjlab_shadowhand import (
    build_shadowhand_cube_env_cfg,
    build_shadowhand_multiobject_env_cfg,
    compute_is_success,
)


def _unique_preserve_order(names: list[str]) -> list[str]:
    unique = []
    seen = set()
    for name in names:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


class _FakeGymSpace:
    def __init__(self, low: np.ndarray, high: np.ndarray, shape: tuple, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return np.random.uniform(lo, hi).astype(self.dtype)


class ShardedMjlabParallelEnv:
    """Sharded mjlab adapter with one homogeneous shard per object."""

    def __init__(
        self,
        env_names: list[str],
        seed: int = 0,
        *,
        num_envs: int | None = None,
        device: str | None = None,
        enable_raycast_sensor: bool = False,
        raycast_sensor_name: str = "pointnet_raycast",
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        names = list(env_names) if env_names else ["cube"]
        unique_objects = _unique_preserve_order(names)

        np.random.seed(seed)
        torch.manual_seed(seed)

        n = num_envs if num_envs is not None else len(names)
        if n <= 0:
            raise ValueError(f"num_envs must be positive, got {n}.")

        num_objects = len(unique_objects)
        if n < num_objects:
            raise ValueError(
                f"num_envs={n} is smaller than the number of objects "
                f"({num_objects}). Sharded mjlab needs at least one env slot "
                "per object so no object is silently omitted."
            )

        base = n // num_objects
        remainder = n % num_objects
        envs_per_object = []
        for i in range(num_objects):
            count = base + (1 if i < remainder else 0)
            envs_per_object.append(count)

        self.unique_object_names = tuple(unique_objects)
        self.num_objects = num_objects
        self._device = device
        self._raycast_sensor_name = raycast_sensor_name if enable_raycast_sensor else None
        self._shards: list[ManagerBasedRlEnv] = []
        self._shard_sizes: list[int] = []
        self._shard_obj_entity_names: list[str] = []

        object_ids = []
        object_names_by_slot = []
        for obj_idx, (obj_name, shard_size) in enumerate(
            zip(unique_objects, envs_per_object)
        ):
            if shard_size == 0:
                continue
            if obj_name == "cube" and num_objects == 1:
                cfg = build_shadowhand_cube_env_cfg(
                    num_envs=shard_size, auto_reset=False
                )
                entity_name = "cube"
            else:
                cfg = build_shadowhand_multiobject_env_cfg(
                    object_names=[obj_name],
                    num_envs=shard_size,
                    auto_reset=False,
                    variant_assignment=[0] * shard_size,
                )
                entity_name = "object"

            if enable_raycast_sensor:
                from jaxrl.mjlab_shadowhand import augment_cfg_with_raycast_sensor
                augment_cfg_with_raycast_sensor(cfg, sensor_name=raycast_sensor_name)

            env = ManagerBasedRlEnv(cfg, device=device)
            self._shards.append(env)
            self._shard_sizes.append(shard_size)
            self._shard_obj_entity_names.append(entity_name)
            object_ids.extend([obj_idx] * shard_size)
            object_names_by_slot.extend([obj_name] * shard_size)

        self.object_ids = np.asarray(object_ids, dtype=np.int32)
        self.object_names_by_slot = np.asarray(object_names_by_slot, dtype=object)
        self.object_id_by_name = {
            name: idx for idx, name in enumerate(self.unique_object_names)
        }
        self.slot_counts_by_object = {
            name: int(np.sum(self.object_names_by_slot == name))
            for name in self.unique_object_names
        }

        first_env = self._shards[0]
        self._action_dim = first_env.action_manager.total_action_dim
        obs, _ = first_env.reset()
        self._obs_dim = obs["policy"].shape[-1]

        total_envs = sum(self._shard_sizes)
        self.num_tasks = total_envs
        self.envs = [None] * total_envs

        obs_low = np.full((total_envs, self._obs_dim), -np.inf, dtype=np.float32)
        obs_high = np.full((total_envs, self._obs_dim), np.inf, dtype=np.float32)
        self.observation_space = _FakeGymSpace(
            obs_low, obs_high, (total_envs, self._obs_dim), np.float64
        )

        act_low = np.full((total_envs, self._action_dim), -1.0, dtype=np.float64)
        act_high = np.full((total_envs, self._action_dim), 1.0, dtype=np.float64)
        self.action_space = _FakeGymSpace(
            act_low, act_high, (total_envs, self._action_dim), np.float64
        )

        self.obs_dims = np.full(total_envs, self._obs_dim, dtype=np.int32)
        self.act_dims = np.full(total_envs, self._action_dim, dtype=np.int32)
        self.state_dim_differences = np.zeros(total_envs, dtype=np.int32)

        self._shard_offsets = []
        offset = 0
        for sz in self._shard_sizes:
            self._shard_offsets.append(offset)
            offset += sz

    def _obs_to_numpy(self, obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
        return obs_dict["policy"].cpu().numpy().astype(np.float64)

    def reset(self) -> np.ndarray:
        total_envs = self.num_tasks
        obs_all = np.empty((total_envs, self._obs_dim), dtype=np.float64)
        for i, env in enumerate(self._shards):
            obs, _ = env.reset()
            offset = self._shard_offsets[i]
            sz = self._shard_sizes[i]
            obs_all[offset : offset + sz] = self._obs_to_numpy(obs)
        return obs_all

    def step(self, actions: np.ndarray):
        total_envs = self.num_tasks
        obs_all = np.empty((total_envs, self._obs_dim), dtype=np.float64)
        rew_all = np.empty(total_envs, dtype=np.float64)
        term_all = np.empty(total_envs, dtype=np.float64)
        trunc_all = np.empty(total_envs, dtype=np.float64)
        goals_all = np.empty(total_envs, dtype=np.float64)

        for i, env in enumerate(self._shards):
            offset = self._shard_offsets[i]
            sz = self._shard_sizes[i]
            entity_name = self._shard_obj_entity_names[i]

            act_np = actions[offset : offset + sz]
            act = torch.tensor(act_np, dtype=torch.float32, device=self._device)
            if act.ndim == 1:
                act = act.unsqueeze(0)
            act = act[:, : self._action_dim]

            obs_buf, rew, term, trunc, extras = env.step(act)

            obs_all[offset : offset + sz] = self._obs_to_numpy(obs_buf)
            rew_all[offset : offset + sz] = rew.cpu().numpy().astype(np.float64)
            term_all[offset : offset + sz] = term.cpu().numpy().astype(np.float64)
            trunc_all[offset : offset + sz] = trunc.cpu().numpy().astype(np.float64)

            obj_quat = env.scene[entity_name].data.root_link_quat_w
            target_quat = env._desired_goal_target[:, 3:]
            goals = compute_is_success(obj_quat, target_quat, 0.1)
            goals_all[offset : offset + sz] = goals.cpu().numpy().astype(np.float64)

        return obs_all, rew_all, term_all, trunc_all, goals_all

    def generate_masks(
        self, terminals: np.ndarray, truncates: np.ndarray
    ) -> np.ndarray:
        return 1 - (terminals * (1 - truncates))

    def reset_where_done(
        self,
        states: np.ndarray,
        terminals: np.ndarray,
        truncates: np.ndarray,
    ):
        done = (terminals == True) | (truncates == True)  # noqa: E712
        if not done.any():
            return states, terminals, truncates

        for i, env in enumerate(self._shards):
            offset = self._shard_offsets[i]
            sz = self._shard_sizes[i]
            shard_done = done[offset : offset + sz]
            if not shard_done.any():
                continue

            local_ids = np.where(shard_done)[0]
            done_ids = torch.tensor(local_ids, dtype=torch.long, device=self._device)
            obs_after, _ = env.reset(env_ids=done_ids)
            new_obs = self._obs_to_numpy(obs_after)
            for local_idx in local_ids:
                global_idx = offset + local_idx
                states[global_idx] = new_obs[local_idx]
                terminals[global_idx] = False
                truncates[global_idx] = False

        return states, terminals, truncates

    def extract_object_poses(self) -> np.ndarray:
        """Extract object (pos, quat) in world frame for every slot.

        Returns
        -------
        poses : (num_slots, 7) float32
            Columns 0-2: object position, 3-6: object quaternion (w,x,y,z).
        """
        total_envs = self.num_tasks
        poses = np.empty((total_envs, 7), dtype=np.float32)
        for i, env in enumerate(self._shards):
            offset = self._shard_offsets[i]
            sz = self._shard_sizes[i]
            entity_name = self._shard_obj_entity_names[i]
            pos = env.scene[entity_name].data.root_link_pos_w.cpu().numpy()
            quat = env.scene[entity_name].data.root_link_quat_w.cpu().numpy()
            poses[offset:offset + sz, :3] = pos[:sz].astype(np.float32)
            poses[offset:offset + sz, 3:] = quat[:sz].astype(np.float32)
        return poses

    def extract_palm_poses(self) -> np.ndarray:
        """Extract palm body (pos, rot_matrix) in world frame for every slot.

        Requires the robot entity to expose body-level state via
        ``scene["robot"].data.body_link_pos_w`` and ``body_link_quat_w``.

        Returns
        -------
        palm_pos : (num_slots, 3) float32
        palm_quat : (num_slots, 4) float32  — (w,x,y,z)

        Raises
        ------
        AttributeError
            If the mjlab entity API does not expose per-body state.
        """
        total_envs = self.num_tasks
        palm_pos = np.empty((total_envs, 3), dtype=np.float32)
        palm_quat = np.empty((total_envs, 4), dtype=np.float32)
        for i, env in enumerate(self._shards):
            offset = self._shard_offsets[i]
            sz = self._shard_sizes[i]
            robot_data = env.scene["robot"].data
            bp = robot_data.body_link_pos_w.cpu().numpy()
            bq = robot_data.body_link_quat_w.cpu().numpy()
            palm_idx = self._palm_body_index(env)
            palm_pos[offset:offset + sz] = bp[:sz, palm_idx].astype(np.float32)
            palm_quat[offset:offset + sz] = bq[:sz, palm_idx].astype(np.float32)
        return palm_pos, palm_quat

    @staticmethod
    def _palm_body_index(env) -> int:
        """Find the index of robot0:palm within the robot articulation."""
        body_names = env.scene["robot"].body_names
        for idx, name in enumerate(body_names):
            if "palm" in name.lower():
                return idx
        return 0

    def extract_raycast_pointclouds(self) -> dict[str, np.ndarray]:
        """Read GPU raycast sensor data from all shards.

        Requires ``enable_raycast_sensor=True`` at construction time.

        Returns
        -------
        dict with keys:
            ``hit_pos_w``  : (num_slots, num_rays, 3) float32
            ``normals_w``  : (num_slots, num_rays, 3) float32
            ``distances``  : (num_slots, num_rays)    float32
        """
        if self._raycast_sensor_name is None:
            raise RuntimeError(
                "Raycast sensor not enabled. Construct with enable_raycast_sensor=True."
            )
        sensor_name = self._raycast_sensor_name
        total_envs = self.num_tasks

        first_sensor = self._shards[0].scene[sensor_name]
        num_rays = first_sensor.data.hit_pos_w.shape[1]

        hit_pos = np.empty((total_envs, num_rays, 3), dtype=np.float32)
        normals = np.empty((total_envs, num_rays, 3), dtype=np.float32)
        distances = np.empty((total_envs, num_rays), dtype=np.float32)

        for i, env in enumerate(self._shards):
            offset = self._shard_offsets[i]
            sz = self._shard_sizes[i]
            sd = env.scene[sensor_name].data
            hit_pos[offset:offset + sz] = sd.hit_pos_w[:sz].cpu().numpy().astype(np.float32)
            normals[offset:offset + sz] = sd.normals_w[:sz].cpu().numpy().astype(np.float32)
            distances[offset:offset + sz] = sd.distances[:sz].cpu().numpy().astype(np.float32)

        return {
            "hit_pos_w": hit_pos,
            "normals_w": normals,
            "distances": distances,
            "num_rays": num_rays,
        }

    def close(self):
        for env in self._shards:
            env.close()
