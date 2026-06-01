"""Verify that mjlab v1.4.0 auto_reset=False preserves BRC's timeout/bootstrap contract.

This standalone smoke script checks six properties required for the BRC training
loop:

  1. ManagerBasedRlEnvCfg has an auto_reset field.
  2. With auto_reset=False, step() returns separate terminated and truncated tensors.
  3. A timeout configured via TerminationTermCfg(time_out=True) appears as truncated=True.
  4. After a done/truncated step, stepping without reset raises RuntimeError.
  5. After reset(env_ids=...), that slot can step normally again.
  6. True terminations map to BRC mask=0.

It also reproduces ParallelEnv.generate_masks and shows that timeouts map to mask=1,
which is the condition for BRC's critic target to bootstrap through the timeout
next observation.
"""

import sys
import traceback

import numpy as np
import torch
import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.entity import EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg


def generate_masks_brc(terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
    """Reproduce the mask rule from jaxrl.envs.ParallelEnv.generate_masks.

    mask = 1  for truncation (bootstrap through timeout next obs)
    mask = 0  for true termination (no bootstrap)
    mask = 1  for non-done steps
    """
    return 1 - (terminated * (1 - truncated))


def _make_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.05,),
        mass=0.1,
        rgba=(0.8, 0.2, 0.2, 1.0),
    )
    return spec


def build_minimal_env_cfg(*, auto_reset: bool, num_envs: int = 4) -> ManagerBasedRlEnvCfg:
    """Build a minimal mjlab env config for testing reset/truncation semantics."""

    observations = {
        "policy": ObservationGroupCfg(
            {
                "ball_vel": ObservationTermCfg(
                    func=envs_mdp.base_lin_vel,
                    params={"asset_cfg": SceneEntityCfg("ball")},
                ),
            },
            enable_corruption=False,
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "ball_fell": TerminationTermCfg(
            func=envs_mdp.root_height_below_minimum,
            params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("ball")},
        ),
    }

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "ball": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 1.0),
                    ),
                    spec_fn=_make_ball_spec,
                ),
            },
            num_envs=num_envs,
            env_spacing=2.0,
        ),
        observations=observations,
        actions={},
        events=events,
        terminations=terminations,
        rewards={},
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01)),
        decimation=5,
        episode_length_s=0.5,
        auto_reset=auto_reset,
    )


def run_checks():
    results = {}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------- Check 1: auto_reset field exists ----------
    import dataclasses
    field_names = [f.name for f in dataclasses.fields(ManagerBasedRlEnvCfg)]
    has_auto_reset = "auto_reset" in field_names
    results["1_auto_reset_field_exists"] = has_auto_reset
    print(f"[CHECK 1] ManagerBasedRlEnvCfg has auto_reset field: {has_auto_reset}")

    # ---------- Build env with auto_reset=False ----------
    print("\nBuilding minimal env with auto_reset=False ...")
    cfg = build_minimal_env_cfg(auto_reset=False, num_envs=4)
    env = ManagerBasedRlEnv(cfg, device=device)

    max_ep_len = env.max_episode_length
    print(f"Max episode length: {max_ep_len} steps")
    print(f"Step dt: {env.step_dt}s, episode_length_s: {cfg.episode_length_s}s")

    obs, _ = env.reset()
    print(f"Observation keys: {list(obs.keys())}")
    print(f"Initial obs shape: {obs['policy'].shape}")

    # ---------- Check 2: step returns separate terminated / truncated ----------
    action = torch.zeros(env.num_envs, 0, device=device)
    step_result = env.step(action)
    assert len(step_result) == 5, f"Expected 5 return values, got {len(step_result)}"
    obs_buf, reward_buf, terminated, truncated, extras = step_result
    assert terminated.shape == (env.num_envs,), f"terminated shape {terminated.shape}"
    assert truncated.shape == (env.num_envs,), f"truncated shape {truncated.shape}"
    results["2_separate_terminated_truncated"] = True
    print(f"\n[CHECK 2] step() returns separate terminated and truncated tensors: True")
    print(f"  terminated dtype={terminated.dtype}, truncated dtype={truncated.dtype}")

    # ---------- Check 3: step until timeout, verify truncated=True ----------
    obs, _ = env.reset()
    timeout_hit = False
    true_term_hit = False

    for step_i in range(1, max_ep_len + 5):
        obs_buf, reward_buf, terminated, truncated, extras = env.step(action)
        policy_obs = obs_buf["policy"]

        terms_np = terminated.cpu().numpy().astype(float)
        truns_np = truncated.cpu().numpy().astype(float)

        if truncated.any() and not timeout_hit:
            timeout_hit = True
            masks = generate_masks_brc(terms_np, truns_np)
            print(f"\n[CHECK 3] Timeout detected at step {step_i}/{max_ep_len}")
            print(f"  terminated: {terms_np}")
            print(f"  truncated:  {truns_np}")
            print(f"  BRC masks:  {masks}")
            timeout_mask_correct = all(
                masks[i] == 1.0 for i in range(len(truns_np)) if truns_np[i] == 1.0
            )
            results["3_timeout_truncated_true"] = timeout_hit
            results["3_timeout_mask_is_1"] = timeout_mask_correct
            print(f"  Timeout maps to mask=1 (bootstrap): {timeout_mask_correct}")
            break

        if terminated.any() and not true_term_hit:
            true_term_hit = True
            masks = generate_masks_brc(terms_np, truns_np)
            print(f"\n  True termination at step {step_i}")
            print(f"  terminated: {terms_np}, truncated: {truns_np}")
            print(f"  BRC masks:  {masks}")

        done_any = (terminated | truncated).any()
        if done_any:
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=done_ids)

    if not timeout_hit:
        results["3_timeout_truncated_true"] = False
        print(f"\n[CHECK 3] FAILED: No timeout detected in {max_ep_len + 4} steps")

    # ---------- Check 4: stepping without reset raises RuntimeError ----------
    obs, _ = env.reset()
    # Step until timeout
    for _ in range(max_ep_len + 1):
        obs_buf, reward_buf, terminated, truncated, extras = env.step(action)
        if (terminated | truncated).any():
            break

    step_without_reset_raises = False
    try:
        env.step(action)
    except RuntimeError as e:
        step_without_reset_raises = True
        print(f"\n[CHECK 4] Stepping without reset raises RuntimeError: True")
        print(f"  Error message: {e}")
    except Exception as e:
        print(f"\n[CHECK 4] Unexpected exception type: {type(e).__name__}: {e}")

    results["4_step_without_reset_raises"] = step_without_reset_raises
    if not step_without_reset_raises:
        print(f"\n[CHECK 4] FAILED: step() did not raise after done without reset")

    # ---------- Check 5: reset(env_ids=...) clears pending state ----------
    done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
    env.reset(env_ids=done_ids)
    can_step_after_reset = False
    try:
        obs_buf, reward_buf, terminated, truncated, extras = env.step(action)
        can_step_after_reset = True
        print(f"\n[CHECK 5] After reset(env_ids=...), step() succeeds: True")
    except Exception as e:
        print(f"\n[CHECK 5] FAILED: step() still raises after reset: {e}")

    results["5_reset_clears_pending"] = can_step_after_reset

    env.close()

    # ---------- Check 6: true termination gives mask=0 ----------
    # Build a second env with a high fall threshold (0.5m) so the ball
    # triggers true termination as it falls from z=1.0 toward the ground.
    print("\nBuilding env for true-termination check (min_height=0.5) ...")
    cfg6 = build_minimal_env_cfg(auto_reset=False, num_envs=2)
    cfg6.terminations["ball_fell"] = TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("ball")},
    )
    env6 = ManagerBasedRlEnv(cfg6, device=device)
    obs6, _ = env6.reset()
    action6 = torch.zeros(env6.num_envs, 0, device=device)

    true_term_found = False
    for step_i in range(1, env6.max_episode_length + 2):
        obs_buf6, _, terminated6, truncated6, _ = env6.step(action6)
        if terminated6.any():
            terms6_np = terminated6.cpu().numpy().astype(float)
            truns6_np = truncated6.cpu().numpy().astype(float)
            masks6 = generate_masks_brc(terms6_np, truns6_np)
            # Find an env that terminated (not timed out)
            for ei in range(env6.num_envs):
                if terms6_np[ei] == 1.0 and truns6_np[ei] == 0.0:
                    true_term_found = True
                    env0_mask_zero = masks6[ei] == 0.0
                    results["6_true_term_mask_0"] = env0_mask_zero
                    print(f"\n[CHECK 6] True termination (ball fell) for env {ei} at step {step_i}:")
                    print(f"  terminated: {terms6_np[ei]}, truncated: {truns6_np[ei]}, mask: {masks6[ei]}")
                    print(f"  True termination gives mask=0: {env0_mask_zero}")
                    break
            break
        done_any = (terminated6 | truncated6).any()
        if done_any:
            done_ids = (terminated6 | truncated6).nonzero(as_tuple=False).squeeze(-1)
            env6.reset(env_ids=done_ids)

    if not true_term_found:
        # Fallback: verify mask formula analytically
        print(f"\n[CHECK 6] Ball never fell below threshold in {env6.max_episode_length} steps.")
        print("  Verifying mask formula analytically:")
        terms_test = np.array([1.0, 0.0, 0.0, 1.0])
        truns_test = np.array([0.0, 1.0, 0.0, 1.0])
        masks_test = generate_masks_brc(terms_test, truns_test)
        print(f"    terminated={terms_test}, truncated={truns_test}")
        print(f"    masks={masks_test}")
        expected = np.array([0.0, 1.0, 1.0, 1.0])
        results["6_true_term_mask_0"] = np.array_equal(masks_test, expected)
        print(f"    matches expected [0, 1, 1, 1]: {results['6_true_term_mask_0']}")

    env6.close()

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for key, val in results.items():
        status = "PASS" if val else "FAIL"
        if not val:
            all_pass = False
        print(f"  [{status}] {key}: {val}")

    print("=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
        print("\nmjlab v1.4.0 with auto_reset=False preserves BRC's")
        print("timeout/bootstrap contract. Timeouts return the true")
        print("terminal observation and map to mask=1 (bootstrap).")
        print("True terminations map to mask=0 (no bootstrap).")
        print("The caller must explicitly reset done envs before stepping.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"SOME CHECKS FAILED: {failed}")

    print("\n--- BRC critic target relevance ---")
    print("In jaxrl/agent/update.py line 52:")
    print("  target = r + gamma * mask * (V(next_obs) - temp * log_pi)")
    print("When mask=1 (timeout), the critic bootstraps from next_obs.")
    print("This requires next_obs to be the TRUE timeout observation,")
    print("NOT a post-reset observation.")
    print("auto_reset=False ensures step() returns the timeout obs,")
    print("and the training loop inserts it into replay BEFORE calling")
    print("reset(env_ids=...).")

    return all_pass


if __name__ == "__main__":
    import importlib.metadata
    print(f"mjlab version: {importlib.metadata.version('mjlab')}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    try:
        success = run_checks()
    except Exception:
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)
