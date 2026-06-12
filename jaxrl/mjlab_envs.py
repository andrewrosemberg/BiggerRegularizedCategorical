"""MjlabParallelEnv — training-loop adapter backed by mjlab.

Presents the subset of the ParallelEnv API used by the online training loop:
  .reset()              -> np.ndarray (num_tasks, obs_dim)
  .step(actions)        -> (obs, rewards, terms, truns, goals)
  .generate_masks(t, r) -> masks
  .reset_where_done(...)-> (obs, terms, truns)
  .observation_space    -> gym.spaces.Box (num_tasks, obs_dim)
  .action_space         -> gym.spaces.Box (num_tasks, act_dim)
  .num_tasks            -> int
  .envs                 -> list  (placeholder for conditioner compat)

Supports single-object (cube, backward-compat) and multi-object modes
using VariantEntityCfg for heterogeneous mesh variants per env slot.
Requires the mjlab virtualenv (.venv_mjlab_check).
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
    """Return unique names in first-seen order."""
    unique = []
    seen = set()
    for name in names:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def _balanced_variant_assignment(num_objects: int, num_envs: int) -> np.ndarray:
    """Assign env slots as evenly as possible across object indices.

    Slots are grouped by object in the same order as the object list. For
    example, 3 objects and 8 slots gives [0, 0, 0, 1, 1, 1, 2, 2].
    """
    if num_objects <= 0:
        raise ValueError(f"num_objects must be positive, got {num_objects}.")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")

    base = num_envs // num_objects
    remainder = num_envs % num_objects
    assignment = []
    for object_id in range(num_objects):
        count = base + (1 if object_id < remainder else 0)
        assignment.extend([object_id] * count)
    return np.asarray(assignment, dtype=np.int32)


class _FakeGymSpace:
    """Minimal shim so train.py can read .shape, .sample(), .low, .high, .dtype."""

    def __init__(self, low: np.ndarray, high: np.ndarray, shape: tuple, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return np.random.uniform(lo, hi).astype(self.dtype)


class MjlabParallelEnv:
    """mjlab adapter matching ParallelEnv's API.

    Supports single-object (cube-only, backward-compat) and multi-object
    (heterogeneous mesh variants) modes. In multi-object mode a single
    batched ManagerBasedRlEnv is used with VariantEntityCfg.
    """

    def __init__(
        self,
        env_names: list[str],
        seed: int = 0,
        *,
        num_envs: int | None = None,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        names = list(env_names) if env_names else ["cube"]

        np.random.seed(seed)
        torch.manual_seed(seed)

        n = num_envs if num_envs is not None else len(names)
        if n <= 0:
            raise ValueError(f"num_envs must be positive, got {n}.")

        unique_objects = _unique_preserve_order(names)
        if unique_objects == ["cube"]:
            self._obj_entity_name = "cube"
            self.unique_object_names = tuple(unique_objects)
            self.object_ids = np.zeros(n, dtype=np.int32)
            self.object_names_by_slot = np.asarray(["cube"] * n, dtype=object)
            cfg = build_shadowhand_cube_env_cfg(num_envs=n, auto_reset=False)
        else:
            self._obj_entity_name = "object"
            variant_assignment = _balanced_variant_assignment(len(unique_objects), n)
            self.unique_object_names = tuple(unique_objects)
            self.object_ids = variant_assignment.copy()
            self.object_names_by_slot = np.asarray(
                [unique_objects[i] for i in variant_assignment],
                dtype=object,
            )
            cfg = build_shadowhand_multiobject_env_cfg(
                object_names=unique_objects,
                num_envs=n,
                auto_reset=False,
                variant_assignment=variant_assignment.tolist(),
            )
        self.object_id_by_name = {
            name: idx for idx, name in enumerate(self.unique_object_names)
        }
        self.slot_counts_by_object = {
            name: int(np.sum(self.object_names_by_slot == name))
            for name in self.unique_object_names
        }

        self._env = ManagerBasedRlEnv(cfg, device=device)
        self._device = device
        self._action_dim = self._env.action_manager.total_action_dim

        obs, _ = self._env.reset()
        self._obs_dim = obs["policy"].shape[-1]

        self.num_tasks = n
        self.envs = [None] * n

        obs_low = np.full((n, self._obs_dim), -np.inf, dtype=np.float32)
        obs_high = np.full((n, self._obs_dim), np.inf, dtype=np.float32)
        self.observation_space = _FakeGymSpace(obs_low, obs_high, (n, self._obs_dim), np.float64)

        act_low = np.full((n, self._action_dim), -1.0, dtype=np.float64)
        act_high = np.full((n, self._action_dim), 1.0, dtype=np.float64)
        self.action_space = _FakeGymSpace(act_low, act_high, (n, self._action_dim), np.float64)

        self.obs_dims = np.full(n, self._obs_dim, dtype=np.int32)
        self.act_dims = np.full(n, self._action_dim, dtype=np.int32)
        self.state_dim_differences = np.zeros(n, dtype=np.int32)

    # ------------------------------------------------------------------
    def _obs_to_numpy(self, obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
        return obs_dict["policy"].cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return self._obs_to_numpy(obs)

    # ------------------------------------------------------------------
    def step(self, actions: np.ndarray):
        act = torch.tensor(actions, dtype=torch.float32, device=self._device)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        act = act[:, :self._action_dim]

        obs_buf, rew, term, trunc, extras = self._env.step(act)

        obs_np = self._obs_to_numpy(obs_buf)
        rew_np = rew.cpu().numpy().astype(np.float64)
        term_np = term.cpu().numpy().astype(np.float64)
        trunc_np = trunc.cpu().numpy().astype(np.float64)

        obj_quat = self._env.scene[self._env._obj_entity_name].data.root_link_quat_w
        target_quat = self._env._desired_goal_target[:, 3:]
        goals_np = compute_is_success(obj_quat, target_quat, 0.1).cpu().numpy().astype(np.float64)

        return obs_np, rew_np, term_np, trunc_np, goals_np

    # ------------------------------------------------------------------
    def generate_masks(self, terminals: np.ndarray, truncates: np.ndarray) -> np.ndarray:
        return 1 - (terminals * (1 - truncates))

    # ------------------------------------------------------------------
    def reset_where_done(
        self,
        states: np.ndarray,
        terminals: np.ndarray,
        truncates: np.ndarray,
    ):
        done = ((terminals == True) | (truncates == True))  # noqa: E712
        if not done.any():
            return states, terminals, truncates

        done_ids = torch.tensor(
            np.where(done)[0], dtype=torch.long, device=self._device
        )
        obs_after, _ = self._env.reset(env_ids=done_ids)
        new_obs = self._obs_to_numpy(obs_after)
        for idx in done_ids.cpu().numpy():
            states[idx] = new_obs[idx]
            terminals[idx] = False
            truncates[idx] = False

        return states, terminals, truncates

    # ------------------------------------------------------------------
    def close(self):
        self._env.close()
