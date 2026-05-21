"""Smoke tests for online-conditioner evaluation, encoder checkpoints,
and training script imports.

These tests validate:
- ParallelEnv.evaluate() works with obs_augment_fn for wrist_raycast;
- ParallelEnv.evaluate() works with obs_augment_fn for mesh_pose;
- OnlineRaycastConditioner checkpoint save/load round-trips;
- Mesh encoder checkpoint save/load round-trips;
- Loaded raycast conditioner produces same embeddings as original;
- Loaded mesh encoder produces same embeddings as original;
- Training script modules import cleanly.

Run with:
    module load python/3.11.9
    source .venv/bin/activate
    MUJOCO_GL=egl JAX_PLATFORM_NAME=cpu python tests/smoke_eval_and_checkpoints.py
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
        print(f"{label:<76s} PASS")
        passed += 1
    except Exception as e:
        print(f"{label:<76s} FAIL")
        traceback.print_exc()
        failed += 1


# -----------------------------------------------------------------------
# 1. Online evaluation with wrist_raycast
# -----------------------------------------------------------------------

def test_eval_wrist_raycast():
    """ParallelEnv.evaluate() with obs_augment_fn should run for wrist_raycast."""
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.online_conditioner import build_raycast_conditioner

    embed_dim = 16
    conditioner = build_raycast_conditioner(seed=0, output_dim=embed_dim)

    eval_env = ParallelEnv(["orange", "cube"], seed=99)
    raw_obs = eval_env.reset()
    emb = conditioner.extract_and_encode(eval_env.envs)
    aug_obs = np.concatenate([raw_obs, emb], axis=-1)

    agent = BRC(
        seed=0,
        observations=aug_obs[:1],
        actions=eval_env.action_space.sample()[:1],
        num_tasks=2,
        conditioning_mode="wrist_raycast",
    )

    def augment_fn(raw_obs, envs):
        embeddings = conditioner.extract_and_encode(envs)
        return np.concatenate([raw_obs, embeddings], axis=-1)

    result = eval_env.evaluate(
        agent, num_episodes=1, temperature=1.0, obs_augment_fn=augment_fn,
    )
    assert "goal" in result, f"Missing 'goal' key, got: {list(result.keys())}"
    assert "return" in result, f"Missing 'return' key"
    assert result["goal"].shape == (2,), f"goal shape {result['goal'].shape}"
    assert result["return"].shape == (2,), f"return shape {result['return'].shape}"
    assert np.all(np.isfinite(result["goal"])), "Non-finite goals"
    assert np.all(np.isfinite(result["return"])), "Non-finite returns"


def test_eval_wrist_raycast_no_augment_baseline():
    """evaluate() without obs_augment_fn should still work (backward compat)."""
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC

    eval_env = ParallelEnv(["orange"], seed=99)
    raw_obs = eval_env.reset()

    agent = BRC(
        seed=0,
        observations=raw_obs[:1],
        actions=eval_env.action_space.sample()[:1],
        num_tasks=1,
        conditioning_mode="none",
    )
    result = eval_env.evaluate(agent, num_episodes=1, temperature=1.0)
    assert "goal" in result
    assert "return" in result


# -----------------------------------------------------------------------
# 2. Online evaluation with mesh_pose
# -----------------------------------------------------------------------

def test_eval_mesh_pose():
    """ParallelEnv.evaluate() with obs_augment_fn should run for mesh_pose."""
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.mesh_conditioner import (
        MeshPoseConditioner,
        build_conditioner_features,
        MESH_FEATURE_DIM,
        POSE_DIM,
    )
    import dex_envs

    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    features, _ = build_conditioner_features(["orange", "cube"], assets_dir)
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
        embeddings = conditioner.extract_and_encode(envs)
        return np.concatenate([raw_obs, embeddings], axis=-1)

    result = eval_env.evaluate(
        agent, num_episodes=1, temperature=1.0, obs_augment_fn=augment_fn,
    )
    assert "goal" in result
    assert result["goal"].shape == (2,)
    assert np.all(np.isfinite(result["return"]))


# -----------------------------------------------------------------------
# 3. Raycast conditioner checkpoint save/load
# -----------------------------------------------------------------------

def test_raycast_checkpoint_save_load():
    """OnlineRaycastConditioner should round-trip through save/load."""
    from jaxrl.online_conditioner import (
        OnlineRaycastConditioner,
        build_raycast_conditioner,
    )

    conditioner = build_raycast_conditioner(seed=42, output_dim=16)
    tmpdir = tempfile.mkdtemp(prefix="raycast_ckpt_")
    try:
        conditioner.save_checkpoint(
            tmpdir,
            extra_metadata={"manifest_path": "test.json", "split": "train"},
        )

        assert os.path.exists(os.path.join(tmpdir, "encoder_params.bin"))
        assert os.path.exists(os.path.join(tmpdir, "metadata.json"))

        import json
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
        assert meta["encoder_type"] == "MaskAwarePointNet"
        assert meta["embed_dim"] == 16
        assert meta["manifest_path"] == "test.json"

        loaded = OnlineRaycastConditioner.load_checkpoint(tmpdir)
        assert loaded.embed_dim == 16
        assert loaded.raycast_config.grid_h == 32
        assert loaded.raycast_config.grid_w == 32
    finally:
        shutil.rmtree(tmpdir)


def test_raycast_checkpoint_produces_same_embeddings():
    """Loaded conditioner should produce the same embeddings as original."""
    import dex_envs  # noqa: F401
    import gymnasium as gym
    import mujoco
    from jaxrl.online_conditioner import (
        OnlineRaycastConditioner,
        build_raycast_conditioner,
    )

    conditioner = build_raycast_conditioner(seed=42, output_dim=16)
    env = gym.make("orange-rotate-v1", reward_type="sparse")
    env.reset(seed=42)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    emb_orig = conditioner.extract_and_encode([env])

    tmpdir = tempfile.mkdtemp(prefix="raycast_ckpt_rt_")
    try:
        conditioner.save_checkpoint(tmpdir)
        loaded = OnlineRaycastConditioner.load_checkpoint(tmpdir)

        env.reset(seed=42)
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        emb_loaded = loaded.extract_and_encode([env])

        assert np.allclose(emb_orig, emb_loaded, atol=1e-5), (
            f"Embeddings differ: max diff={np.max(np.abs(emb_orig - emb_loaded))}"
        )
    finally:
        shutil.rmtree(tmpdir)
        env.close()


# -----------------------------------------------------------------------
# 4. Mesh encoder checkpoint save/load
# -----------------------------------------------------------------------

def test_mesh_encoder_checkpoint_save_load():
    """Mesh encoder checkpoint should round-trip through save/load."""
    import jax
    import jax.numpy as jnp
    from jaxrl.pointnet import MaskAwarePointNet
    from jaxrl.mesh_conditioner import (
        save_mesh_encoder_checkpoint,
        load_mesh_encoder_checkpoint,
    )

    hidden_dims = (32, 64)
    output_dim = 16
    n_points = 256

    encoder_def = MaskAwarePointNet(hidden_dims=hidden_dims, output_dim=output_dim)
    rng = jax.random.PRNGKey(7)
    dummy_pts = jnp.zeros((1, n_points, 3))
    dummy_mask = jnp.ones((1, n_points), dtype=bool)
    variables = encoder_def.init(rng, dummy_pts, dummy_mask)
    params = variables["params"]

    tmpdir = tempfile.mkdtemp(prefix="mesh_ckpt_")
    try:
        save_mesh_encoder_checkpoint(
            tmpdir, params, hidden_dims, output_dim, n_points,
            extra_metadata={"manifest_path": "test.json"},
        )
        assert os.path.exists(os.path.join(tmpdir, "encoder_params.bin"))
        assert os.path.exists(os.path.join(tmpdir, "metadata.json"))

        loaded_def, loaded_params, meta = load_mesh_encoder_checkpoint(tmpdir)
        assert meta["output_dim"] == output_dim
        assert tuple(meta["hidden_dims"]) == hidden_dims

        test_pts = jax.random.normal(rng, (2, n_points, 3))
        test_mask = jnp.ones((2, n_points), dtype=bool)

        out_orig = encoder_def.apply({"params": params}, test_pts, test_mask)
        out_loaded = loaded_def.apply({"params": loaded_params}, test_pts, test_mask)
        assert np.allclose(np.array(out_orig), np.array(out_loaded), atol=1e-5), (
            "Loaded encoder produces different output"
        )
    finally:
        shutil.rmtree(tmpdir)


def test_mesh_checkpoint_metadata_content():
    """Mesh checkpoint metadata should contain required fields."""
    import jax
    import jax.numpy as jnp
    import json
    from jaxrl.pointnet import MaskAwarePointNet
    from jaxrl.mesh_conditioner import save_mesh_encoder_checkpoint

    encoder_def = MaskAwarePointNet(hidden_dims=(32,), output_dim=8)
    rng = jax.random.PRNGKey(0)
    variables = encoder_def.init(rng, jnp.zeros((1, 64, 3)), jnp.ones((1, 64), dtype=bool))

    tmpdir = tempfile.mkdtemp(prefix="mesh_meta_")
    try:
        save_mesh_encoder_checkpoint(
            tmpdir, variables["params"], (32,), 8, 64,
            extra_metadata={"manifest_path": "m.json", "seed": 0},
        )
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)

        required = ["encoder_type", "hidden_dims", "output_dim", "n_points",
                     "point_channels", "mask_convention", "coordinate_frame"]
        for key in required:
            assert key in meta, f"Missing metadata key: {key}"
    finally:
        shutil.rmtree(tmpdir)


# -----------------------------------------------------------------------
# 5. Training script imports
# -----------------------------------------------------------------------

def test_raycast_training_script_imports():
    """The raycast training script should import without errors."""
    import importlib.util
    name = "train_raycast_pointnet"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO_ROOT, "scripts", "train_raycast_pointnet.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.modules.pop(name, None)
    assert hasattr(mod, "collect_dataset")
    assert hasattr(mod, "build_model_and_params")
    assert hasattr(mod, "train_step")
    assert hasattr(mod, "evaluate_split")


def test_mesh_training_script_imports():
    """The mesh training script should import without errors."""
    import importlib.util
    name = "train_mesh_pointnet"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO_ROOT, "scripts", "train_mesh_pointnet.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.modules.pop(name, None)
    assert hasattr(mod, "build_mesh_dataset")
    assert hasattr(mod, "sample_mesh_points")
    assert hasattr(mod, "train_step")
    assert hasattr(mod, "evaluate_embeddings")


# -----------------------------------------------------------------------
# 6. Loaded raycast conditioner used with online wrist_raycast mode
# -----------------------------------------------------------------------

def test_loaded_raycast_conditioner_with_brc():
    """A loaded raycast conditioner should work with BRC wrist_raycast mode."""
    from jaxrl.online_conditioner import (
        OnlineRaycastConditioner,
        build_raycast_conditioner,
    )
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC

    conditioner = build_raycast_conditioner(seed=0, output_dim=16)
    tmpdir = tempfile.mkdtemp(prefix="raycast_brc_")
    try:
        conditioner.save_checkpoint(tmpdir)
        loaded = OnlineRaycastConditioner.load_checkpoint(tmpdir)

        penv = ParallelEnv(["orange"], seed=0)
        raw_obs = penv.reset()
        emb = loaded.extract_and_encode(penv.envs)
        aug_obs = np.concatenate([raw_obs, emb], axis=-1)

        agent = BRC(
            seed=0,
            observations=aug_obs[:1],
            actions=penv.action_space.sample()[:1],
            num_tasks=1,
            conditioning_mode="wrist_raycast",
        )
        actions = agent.sample_actions(aug_obs, temperature=1.0)
        assert actions.shape == (1, 20)
        assert np.all(np.isfinite(actions))
    finally:
        shutil.rmtree(tmpdir)


def test_loaded_mesh_encoder_with_mesh_pose():
    """A loaded mesh encoder should be usable for mesh_pose conditioner."""
    import jax
    import jax.numpy as jnp
    from jaxrl.pointnet import MaskAwarePointNet
    from jaxrl.mesh_conditioner import (
        save_mesh_encoder_checkpoint,
        load_mesh_encoder_checkpoint,
        MeshPoseConditioner,
        POSE_DIM,
    )

    hidden_dims = (32,)
    output_dim = 8
    n_points = 64

    encoder_def = MaskAwarePointNet(hidden_dims=hidden_dims, output_dim=output_dim)
    rng = jax.random.PRNGKey(0)
    variables = encoder_def.init(
        rng, jnp.zeros((1, n_points, 3)), jnp.ones((1, n_points), dtype=bool)
    )
    params = variables["params"]

    tmpdir = tempfile.mkdtemp(prefix="mesh_pose_ckpt_")
    try:
        save_mesh_encoder_checkpoint(tmpdir, params, hidden_dims, output_dim, n_points)
        loaded_def, loaded_params, meta = load_mesh_encoder_checkpoint(tmpdir)

        test_pts = jax.random.normal(rng, (2, n_points, 3))
        test_mask = jnp.ones((2, n_points), dtype=bool)
        features = np.array(loaded_def.apply(
            {"params": loaded_params}, test_pts, test_mask
        ))

        conditioner = MeshPoseConditioner(shape_features=features)
        assert conditioner.embed_dim == output_dim + POSE_DIM
    finally:
        shutil.rmtree(tmpdir)


# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Eval with wrist_raycast obs_augment_fn", test_eval_wrist_raycast),
        ("Eval without obs_augment_fn (backward compat)", test_eval_wrist_raycast_no_augment_baseline),
        ("Eval with mesh_pose obs_augment_fn", test_eval_mesh_pose),
        ("Raycast checkpoint save/load round-trip", test_raycast_checkpoint_save_load),
        ("Raycast checkpoint same embeddings", test_raycast_checkpoint_produces_same_embeddings),
        ("Mesh encoder checkpoint save/load", test_mesh_encoder_checkpoint_save_load),
        ("Mesh checkpoint metadata content", test_mesh_checkpoint_metadata_content),
        ("Raycast training script imports", test_raycast_training_script_imports),
        ("Mesh training script imports", test_mesh_training_script_imports),
        ("Loaded raycast conditioner with BRC", test_loaded_raycast_conditioner_with_brc),
        ("Loaded mesh encoder with mesh_pose", test_loaded_mesh_encoder_with_mesh_pose),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\nRESULTS: {passed} passed, {failed} failed out of {test_num}")
    sys.exit(1 if failed > 0 else 0)
