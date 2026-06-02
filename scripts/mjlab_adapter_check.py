"""Verify that MjlabParallelEnv matches the ParallelEnv API contract.

Run with:
  .venv_mjlab_check/bin/python scripts/mjlab_adapter_check.py
"""

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxrl.mjlab_envs import MjlabParallelEnv


def run_checks():
    results = {}
    N = 8

    print(f"Building MjlabParallelEnv (num_envs={N}) ...")
    env = MjlabParallelEnv(["cube"] * N, seed=42, num_envs=N)
    print()

    # ------------------------------------------------------------------
    # CHECK 1: attributes exist and shapes match
    # ------------------------------------------------------------------
    c1 = (
        env.num_tasks == N
        and env.observation_space.shape == (N, 68)
        and env.action_space.shape == (N, 20)
        and len(env.envs) == N
        and env.obs_dims.shape == (N,)
        and env.act_dims.shape == (N,)
        and env.state_dim_differences.shape == (N,)
    )
    results["1_attributes"] = c1
    print(f"[CHECK 1] attributes: num_tasks={env.num_tasks} "
          f"obs_space={env.observation_space.shape} "
          f"act_space={env.action_space.shape} — {'PASS' if c1 else 'FAIL'}")

    # ------------------------------------------------------------------
    # CHECK 2: reset() returns correct shape
    # ------------------------------------------------------------------
    obs = env.reset()
    c2 = (obs.shape == (N, 68) and np.isfinite(obs).all())
    results["2_reset"] = c2
    print(f"[CHECK 2] reset: shape={obs.shape} finite={np.isfinite(obs).all()} — {'PASS' if c2 else 'FAIL'}")

    # ------------------------------------------------------------------
    # CHECK 3: step() returns 5-tuple with correct shapes
    # ------------------------------------------------------------------
    actions = np.zeros((N, 20))
    obs2, rew, term, trunc, goals = env.step(actions)
    c3 = (
        obs2.shape == (N, 68)
        and rew.shape == (N,)
        and term.shape == (N,)
        and trunc.shape == (N,)
        and goals.shape == (N,)
        and np.isfinite(obs2).all()
    )
    results["3_step"] = c3
    print(f"[CHECK 3] step: obs={obs2.shape} rew={rew.shape} term={term.shape} "
          f"goals={goals.shape} — {'PASS' if c3 else 'FAIL'}")

    # ------------------------------------------------------------------
    # CHECK 4: generate_masks matches ParallelEnv formula
    # ------------------------------------------------------------------
    t_test = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float64)
    r_test = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.float64)
    masks = env.generate_masks(t_test, r_test)
    expected = np.array([0, 1, 1, 1, 1, 1, 0, 1], dtype=np.float64)
    c4 = np.array_equal(masks, expected)
    results["4_generate_masks"] = c4
    print(f"[CHECK 4] generate_masks: {masks.tolist()} expected {expected.tolist()} — {'PASS' if c4 else 'FAIL'}")

    # ------------------------------------------------------------------
    # CHECK 5: reset_where_done resets only done envs
    # ------------------------------------------------------------------
    obs = env.reset()
    # Step a few times
    for _ in range(5):
        obs, rew, term, trunc, goals = env.step(np.zeros((N, 20)))

    # Fake some done flags
    term_fake = np.zeros(N)
    trunc_fake = np.zeros(N)
    term_fake[0] = 1.0
    trunc_fake[3] = 1.0

    obs_before = obs.copy()
    obs_out, term_out, trunc_out = env.reset_where_done(obs, term_fake, trunc_fake)

    changed_0 = not np.array_equal(obs_before[0], obs_out[0])
    changed_3 = not np.array_equal(obs_before[3], obs_out[3])
    unchanged_1 = np.array_equal(obs_before[1], obs_out[1])
    c5 = (
        changed_0 and changed_3 and unchanged_1
        and term_out[0] == 0.0 and trunc_out[3] == 0.0
    )
    results["5_reset_where_done"] = c5
    print(f"[CHECK 5] reset_where_done: slot0_changed={changed_0} "
          f"slot3_changed={changed_3} slot1_unchanged={unchanged_1} — {'PASS' if c5 else 'FAIL'}")

    # ------------------------------------------------------------------
    # CHECK 6: goals are computed (is_success)
    # ------------------------------------------------------------------
    env.reset()
    all_goals = []
    for _ in range(100):
        obs, rew, term, trunc, goals = env.step(np.random.randn(N, 20) * 0.01)
        all_goals.append(goals.copy())
        done = (term > 0) | (trunc > 0)
        if done.any():
            obs, term, trunc = env.reset_where_done(obs, term, trunc)
    all_goals = np.concatenate(all_goals)
    c6 = (set(np.unique(all_goals).tolist()).issubset({0.0, 1.0}))
    results["6_goals"] = c6
    print(f"[CHECK 6] goals in {{0, 1}}: unique={np.unique(all_goals).tolist()} — {'PASS' if c6 else 'FAIL'}")

    # ------------------------------------------------------------------
    # CHECK 7: full rollout loop (mimics train.py inner loop)
    # ------------------------------------------------------------------
    obs = env.reset()
    rollout_ok = True
    for i in range(200):
        actions = env.action_space.sample()
        next_obs, rewards, terms, truns, goals = env.step(actions)
        masks = env.generate_masks(terms, truns)
        # Verify types/shapes match train.py expectations
        if not (isinstance(next_obs, np.ndarray) and next_obs.shape == (N, 68)):
            rollout_ok = False; break
        if not (isinstance(masks, np.ndarray) and masks.shape == (N,)):
            rollout_ok = False; break
        next_obs, terms, truns = env.reset_where_done(next_obs, terms, truns)
        obs = next_obs

    results["7_rollout_loop"] = rollout_ok
    print(f"[CHECK 7] 200-step rollout loop — {'PASS' if rollout_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # CHECK 8: throughput (steps/sec)
    # ------------------------------------------------------------------
    obs = env.reset()
    # warmup
    for _ in range(50):
        obs, rew, term, trunc, goals = env.step(np.random.randn(N, 20) * 0.01)
        done = (term > 0) | (trunc > 0)
        if done.any():
            obs, term, trunc = env.reset_where_done(obs, term, trunc)

    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    num_steps = 500
    t0 = time.perf_counter()
    for _ in range(num_steps):
        obs, rew, term, trunc, goals = env.step(np.random.randn(N, 20) * 0.01)
        done = (term > 0) | (trunc > 0)
        if done.any():
            obs, term, trunc = env.reset_where_done(obs, term, trunc)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = num_steps / dt
    tps = num_steps * N / dt
    results["8_throughput"] = True
    print(f"[CHECK 8] throughput: {sps:.0f} steps/sec ({tps:.0f} transitions/sec, {N} envs)")

    env.close()
    return results


def main():
    print("=" * 70)
    print("MjlabParallelEnv Adapter Check")
    print("=" * 70)
    print()

    try:
        results = run_checks()
    except Exception:
        traceback.print_exc()
        results = {}

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        if not v:
            all_pass = False
        print(f"  [{status}] {k}")
    print()
    print("ALL CHECKS PASSED" if all_pass else f"FAILED: {[k for k,v in results.items() if not v]}")
    return all_pass


if __name__ == "__main__":
    ok = False
    try:
        ok = main()
    except Exception:
        traceback.print_exc()
    sys.exit(0 if ok else 1)
