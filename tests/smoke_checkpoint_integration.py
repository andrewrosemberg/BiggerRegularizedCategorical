"""Smoke tests for conditioner checkpoint integration in train.py.

Validates that train.py can initialize and run briefly with:
- wrist_raycast using a loaded raycast checkpoint;
- mesh_shape using a loaded mesh encoder checkpoint;
- mesh_pose using a loaded mesh encoder checkpoint;
- offline evaluation enabled for wrist_raycast and mesh_pose.

Tests create tiny checkpoints in temporary directories.

Run with:
    module load python/3.11.9
    source .venv/bin/activate
    MUJOCO_GL=egl JAX_PLATFORM_NAME=cpu python tests/smoke_checkpoint_integration.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

MANIFEST_PATH = os.path.join(REPO_ROOT, "manifests", "shadowhand_split_v1.json")

passed = 0
failed = 0
test_num = 0


def run_test(name: str, fn):
    global passed, failed, test_num
    test_num += 1
    label = f"TEST {test_num:2d}: {name}"
    try:
        fn()
        print(f"{label:<80s} PASS")
        passed += 1
    except Exception as e:
        print(f"{label:<80s} FAIL")
        traceback.print_exc()
        failed += 1


# -----------------------------------------------------------------------
# Helpers: create tiny checkpoints for testing
# -----------------------------------------------------------------------

def _make_tiny_raycast_checkpoint(tmpdir: str) -> str:
    """Create a minimal raycast conditioner checkpoint."""
    from jaxrl.online_conditioner import build_raycast_conditioner
    conditioner = build_raycast_conditioner(seed=0, output_dim=16,
                                             hidden_dims=(32,))
    ckpt_dir = os.path.join(tmpdir, "raycast_ckpt")
    conditioner.save_checkpoint(ckpt_dir, extra_metadata={"test": True})
    return ckpt_dir


def _make_tiny_mesh_checkpoint(tmpdir: str) -> str:
    """Create a minimal mesh encoder checkpoint."""
    import jax
    import jax.numpy as jnp
    from jaxrl.pointnet import MaskAwarePointNet
    from jaxrl.mesh_conditioner import save_mesh_encoder_checkpoint

    hidden_dims = (32,)
    output_dim = 16
    n_points = 256

    encoder_def = MaskAwarePointNet(hidden_dims=hidden_dims, output_dim=output_dim)
    rng = jax.random.PRNGKey(0)
    dummy_pts = jnp.zeros((1, n_points, 3))
    dummy_mask = jnp.ones((1, n_points), dtype=bool)
    variables = encoder_def.init(rng, dummy_pts, dummy_mask)

    ckpt_dir = os.path.join(tmpdir, "mesh_ckpt")
    save_mesh_encoder_checkpoint(
        ckpt_dir, variables["params"], hidden_dims, output_dim, n_points,
        extra_metadata={"test": True},
    )
    return ckpt_dir


# -----------------------------------------------------------------------
# 1. wrist_raycast with loaded checkpoint
# -----------------------------------------------------------------------

def test_wrist_raycast_with_checkpoint():
    """train.py wrist_raycast path should work with a loaded checkpoint."""
    from jaxrl.online_conditioner import OnlineRaycastConditioner
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.replay_buffer import ParallelReplayBuffer
    import gymnasium as gym

    tmpdir = tempfile.mkdtemp(prefix="test_rc_ckpt_")
    try:
        ckpt_dir = _make_tiny_raycast_checkpoint(tmpdir)
        conditioner = OnlineRaycastConditioner.load_checkpoint(ckpt_dir)

        env = ParallelEnv(["orange"], seed=0)
        raw_obs = env.reset()
        emb = conditioner.extract_and_encode(env.envs)
        aug_obs = np.concatenate([raw_obs, emb], axis=-1)

        agent = BRC(
            seed=0,
            observations=aug_obs[:1],
            actions=env.action_space.sample()[:1],
            num_tasks=1,
            conditioning_mode="wrist_raycast",
        )

        aug_dim = env.observation_space.shape[-1] + conditioner.embed_dim
        aug_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, aug_dim), dtype=np.float32,
        )
        replay_buffer = ParallelReplayBuffer(
            aug_obs_space, env.action_space.shape[-1], 1000, num_tasks=1,
        )

        actions = agent.sample_actions(aug_obs, temperature=1.0)
        next_raw, rewards, terms, truns, goals = env.step(actions)
        next_aug = np.concatenate(
            [next_raw, conditioner.extract_and_encode(env.envs)], axis=-1,
        )
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(aug_obs, actions, rewards, masks, next_aug)
        assert replay_buffer.size > 0
    finally:
        shutil.rmtree(tmpdir)


def test_wrist_raycast_checkpoint_eval():
    """Evaluation with loaded raycast checkpoint should produce valid metrics."""
    from jaxrl.online_conditioner import OnlineRaycastConditioner
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC

    tmpdir = tempfile.mkdtemp(prefix="test_rc_eval_")
    try:
        ckpt_dir = _make_tiny_raycast_checkpoint(tmpdir)
        conditioner = OnlineRaycastConditioner.load_checkpoint(ckpt_dir)

        eval_env = ParallelEnv(["orange"], seed=99)
        raw_obs = eval_env.reset()
        emb = conditioner.extract_and_encode(eval_env.envs)
        aug_obs = np.concatenate([raw_obs, emb], axis=-1)

        agent = BRC(
            seed=0,
            observations=aug_obs[:1],
            actions=eval_env.action_space.sample()[:1],
            num_tasks=1,
            conditioning_mode="wrist_raycast",
        )

        def augment_fn(raw_obs, envs):
            return np.concatenate(
                [raw_obs, conditioner.extract_and_encode(envs)], axis=-1,
            )

        result = eval_env.evaluate(
            agent, num_episodes=1, temperature=1.0, obs_augment_fn=augment_fn,
        )
        assert "goal" in result
        assert "return" in result
        assert np.all(np.isfinite(result["goal"]))
        assert np.all(np.isfinite(result["return"]))
    finally:
        shutil.rmtree(tmpdir)


# -----------------------------------------------------------------------
# 2. mesh_shape with loaded checkpoint
# -----------------------------------------------------------------------

def test_mesh_shape_with_checkpoint():
    """mesh_shape mode should work with a loaded mesh encoder checkpoint."""
    from jaxrl.mesh_conditioner import build_learned_mesh_features
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    import dex_envs

    tmpdir = tempfile.mkdtemp(prefix="test_ms_ckpt_")
    try:
        ckpt_dir = _make_tiny_mesh_checkpoint(tmpdir)
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")

        features, meta = build_learned_mesh_features(
            ["orange", "cube"], assets_dir, ckpt_dir, seed=0,
        )
        assert features.shape[0] == 2
        assert features.shape[1] == 16
        assert np.all(np.isfinite(features))

        env = ParallelEnv(["orange", "cube"], seed=0)
        raw_obs = env.reset()

        agent = BRC(
            seed=0,
            observations=raw_obs[:1],
            actions=env.action_space.sample()[:1],
            num_tasks=2,
            conditioning_mode="mesh_shape",
            conditioner_features=features,
        )
        actions = agent.sample_actions(raw_obs, temperature=1.0)
        assert actions.shape == (2, 20)
        assert np.all(np.isfinite(actions))
    finally:
        shutil.rmtree(tmpdir)


# -----------------------------------------------------------------------
# 3. mesh_pose with loaded checkpoint
# -----------------------------------------------------------------------

def test_mesh_pose_with_checkpoint():
    """mesh_pose mode should work with a loaded mesh encoder checkpoint."""
    from jaxrl.mesh_conditioner import (
        build_learned_mesh_features,
        MeshPoseConditioner,
        POSE_DIM,
    )
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    import dex_envs

    tmpdir = tempfile.mkdtemp(prefix="test_mp_ckpt_")
    try:
        ckpt_dir = _make_tiny_mesh_checkpoint(tmpdir)
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")

        features, meta = build_learned_mesh_features(
            ["orange", "cube"], assets_dir, ckpt_dir, seed=0,
        )
        conditioner = MeshPoseConditioner(shape_features=features)
        assert conditioner.embed_dim == 16 + POSE_DIM

        env = ParallelEnv(["orange", "cube"], seed=0)
        raw_obs = env.reset()
        emb = conditioner.extract_and_encode(env.envs)
        aug_obs = np.concatenate([raw_obs, emb], axis=-1)

        agent = BRC(
            seed=0,
            observations=aug_obs[:1],
            actions=env.action_space.sample()[:1],
            num_tasks=2,
            conditioning_mode="mesh_pose",
        )
        actions = agent.sample_actions(aug_obs, temperature=1.0)
        assert actions.shape == (2, 20)
        assert np.all(np.isfinite(actions))
    finally:
        shutil.rmtree(tmpdir)


def test_mesh_pose_checkpoint_eval():
    """Evaluation with loaded mesh encoder for mesh_pose should work."""
    from jaxrl.mesh_conditioner import (
        build_learned_mesh_features,
        MeshPoseConditioner,
    )
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    import dex_envs

    tmpdir = tempfile.mkdtemp(prefix="test_mp_eval_")
    try:
        ckpt_dir = _make_tiny_mesh_checkpoint(tmpdir)
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")

        features, _ = build_learned_mesh_features(
            ["orange", "cube"], assets_dir, ckpt_dir, seed=0,
        )
        conditioner = MeshPoseConditioner(shape_features=features)

        eval_env = ParallelEnv(["orange", "cube"], seed=99)
        raw_obs = eval_env.reset()
        emb = conditioner.extract_and_encode(eval_env.envs)
        aug_obs = np.concatenate([raw_obs, emb], axis=-1)

        agent = BRC(
            seed=0,
            observations=aug_obs[:1],
            actions=eval_env.action_space.sample()[:1],
            num_tasks=2,
            conditioning_mode="mesh_pose",
        )

        def augment_fn(raw_obs, envs):
            return np.concatenate(
                [raw_obs, conditioner.extract_and_encode(envs)], axis=-1,
            )

        result = eval_env.evaluate(
            agent, num_episodes=1, temperature=1.0, obs_augment_fn=augment_fn,
        )
        assert "goal" in result
        assert result["goal"].shape == (2,)
        assert np.all(np.isfinite(result["return"]))
    finally:
        shutil.rmtree(tmpdir)


# -----------------------------------------------------------------------
# 4. Surface sampling quality
# -----------------------------------------------------------------------

def test_triangle_area_surface_sampling():
    """Triangle-area surface sampling should produce valid points on mesh."""
    from jaxrl.mesh_conditioner import load_binary_stl_triangles, resolve_stl_path
    import dex_envs

    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    stl_path = resolve_stl_path("orange", assets_dir)
    v1, v2, v3, normals = load_binary_stl_triangles(stl_path)
    assert v1.shape == v2.shape == v3.shape
    assert v1.shape[1] == 3

    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    from train_mesh_pointnet import sample_mesh_surface_points

    rng = np.random.RandomState(42)
    pts = sample_mesh_surface_points(v1, v2, v3, 512, rng)
    assert pts.shape == (512, 3)
    assert np.all(np.isfinite(pts))
    assert np.linalg.norm(pts, axis=-1).max() <= 1.0 + 1e-5


def test_surface_sampling_deterministic():
    """Same seed should produce identical surface samples."""
    from jaxrl.mesh_conditioner import load_binary_stl_triangles, resolve_stl_path
    import dex_envs

    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    stl_path = resolve_stl_path("cube", assets_dir)
    v1, v2, v3, _ = load_binary_stl_triangles(stl_path)

    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    from train_mesh_pointnet import sample_mesh_surface_points

    pts1 = sample_mesh_surface_points(v1, v2, v3, 256, np.random.RandomState(7))
    pts2 = sample_mesh_surface_points(v1, v2, v3, 256, np.random.RandomState(7))
    assert np.allclose(pts1, pts2)


# -----------------------------------------------------------------------
# 5. Learned mesh features helper
# -----------------------------------------------------------------------

def test_build_learned_mesh_features():
    """build_learned_mesh_features should produce valid features for objects."""
    from jaxrl.mesh_conditioner import build_learned_mesh_features
    import dex_envs

    tmpdir = tempfile.mkdtemp(prefix="test_lmf_")
    try:
        ckpt_dir = _make_tiny_mesh_checkpoint(tmpdir)
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")

        features, meta = build_learned_mesh_features(
            ["orange", "cube", "hammer"], assets_dir, ckpt_dir, seed=0,
        )
        assert features.shape == (3, 16)
        assert np.all(np.isfinite(features))
        assert meta["feature_source"] == "learned_mesh_pointnet"
        assert meta["num_objects"] == 3
    finally:
        shutil.rmtree(tmpdir)


def test_learned_mesh_features_deterministic():
    """Same seed should produce identical learned features."""
    from jaxrl.mesh_conditioner import build_learned_mesh_features
    import dex_envs

    tmpdir = tempfile.mkdtemp(prefix="test_lmf_det_")
    try:
        ckpt_dir = _make_tiny_mesh_checkpoint(tmpdir)
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")

        f1, _ = build_learned_mesh_features(
            ["orange", "cube"], assets_dir, ckpt_dir, seed=42,
        )
        f2, _ = build_learned_mesh_features(
            ["orange", "cube"], assets_dir, ckpt_dir, seed=42,
        )
        assert np.allclose(f1, f2, atol=1e-5)
    finally:
        shutil.rmtree(tmpdir)


def test_learned_mesh_features_order_independent():
    """Per-object features must not depend on the surrounding object list."""
    from jaxrl.mesh_conditioner import build_learned_mesh_features
    import dex_envs

    tmpdir = tempfile.mkdtemp(prefix="test_lmf_order_")
    try:
        ckpt_dir = _make_tiny_mesh_checkpoint(tmpdir)
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")

        f_solo, _ = build_learned_mesh_features(
            ["orange"], assets_dir, ckpt_dir, seed=7,
        )
        f_pair, _ = build_learned_mesh_features(
            ["cube", "orange"], assets_dir, ckpt_dir, seed=7,
        )
        assert np.allclose(f_solo[0], f_pair[1], atol=1e-5), (
            f"orange feature differs: solo={f_solo[0][:4]}... pair={f_pair[1][:4]}..."
        )

        f_rev, _ = build_learned_mesh_features(
            ["orange", "cube"], assets_dir, ckpt_dir, seed=7,
        )
        assert np.allclose(f_pair[0], f_rev[1], atol=1e-5), (
            "cube feature differs between orderings"
        )
        assert np.allclose(f_pair[1], f_rev[0], atol=1e-5), (
            "orange feature differs between orderings"
        )
    finally:
        shutil.rmtree(tmpdir)


# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("wrist_raycast with loaded checkpoint", test_wrist_raycast_with_checkpoint),
        ("wrist_raycast checkpoint eval", test_wrist_raycast_checkpoint_eval),
        ("mesh_shape with loaded checkpoint", test_mesh_shape_with_checkpoint),
        ("mesh_pose with loaded checkpoint", test_mesh_pose_with_checkpoint),
        ("mesh_pose checkpoint eval", test_mesh_pose_checkpoint_eval),
        ("Triangle-area surface sampling", test_triangle_area_surface_sampling),
        ("Surface sampling deterministic", test_surface_sampling_deterministic),
        ("build_learned_mesh_features", test_build_learned_mesh_features),
        ("Learned mesh features deterministic", test_learned_mesh_features_deterministic),
        ("Learned mesh features order-independent", test_learned_mesh_features_order_independent),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\nRESULTS: {passed} passed, {failed} failed out of {test_num}")
    sys.exit(1 if failed > 0 else 0)
