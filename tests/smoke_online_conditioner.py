"""Smoke tests for the online conditioner interface.

These tests validate the contracts that BRC online conditioning modes
(``wrist_raycast`` and ``mesh_pose``) depend on:

- wrist-raycast tensor construction from current environment state;
- point tensor and mask shapes for the default 32x32 ray grid;
- mask-aware PointNet forward pass with valid and all-miss inputs;
- BRC initialization with ``conditioning_mode=wrist_raycast``;
- no learned task embedding exists in ``wrist_raycast`` mode;
- online conditioner produces fresh embeddings each step;
- mesh-shape interface contract (static per-object embedding);
- mesh-pose conditioner produces shape + pose embedding.

Run with:
    module load python/3.11.9
    source .venv/bin/activate
    MUJOCO_GL=egl JAX_PLATFORM_NAME=cpu python tests/smoke_online_conditioner.py
"""

from __future__ import annotations

import os
import sys
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
        print(f"{label:<72s} PASS")
        passed += 1
    except Exception as e:
        print(f"{label:<72s} FAIL")
        traceback.print_exc()
        failed += 1


def _make_env(obj_name: str, seed: int = 42):
    import dex_envs  # noqa: F401
    import gymnasium as gym
    import mujoco

    env = gym.make(f"{obj_name}-rotate-v1", reward_type="sparse")
    env.reset(seed=seed)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    return env


# -----------------------------------------------------------------------
# 1. PointNet module
# -----------------------------------------------------------------------

def test_pointnet_imports():
    """PointNet module and constants should import without errors."""
    from jaxrl.pointnet import (  # noqa: F401
        MaskAwarePointNet,
        RAYCAST_POINT_CHANNELS,
        DEFAULT_EMBED_DIM,
        NORMALIZATION_SCALE,
    )
    assert RAYCAST_POINT_CHANNELS == 3
    assert DEFAULT_EMBED_DIM == 64
    assert NORMALIZATION_SCALE == 0.34


def test_pointnet_forward_pass():
    """PointNet should map (batch, N, 3) + mask to (batch, embed_dim)."""
    import jax
    import jax.numpy as jnp
    from jaxrl.pointnet import MaskAwarePointNet

    encoder = MaskAwarePointNet(hidden_dims=(32, 64), output_dim=16)
    rng = jax.random.PRNGKey(0)
    points = jax.random.normal(rng, (2, 1024, 3))
    mask = jnp.ones((2, 1024), dtype=bool)
    mask = mask.at[0, 500:].set(False)

    variables = encoder.init(rng, points, mask)
    out = encoder.apply(variables, points, mask)
    assert out.shape == (2, 16), f"Expected (2, 16), got {out.shape}"
    assert jnp.all(jnp.isfinite(out)), "Non-finite encoder output"


def test_pointnet_mask_affects_output():
    """Masking rays should change the pooled embedding."""
    import jax
    import jax.numpy as jnp
    from jaxrl.pointnet import MaskAwarePointNet

    encoder = MaskAwarePointNet(hidden_dims=(32,), output_dim=8)
    rng = jax.random.PRNGKey(1)
    points = jax.random.normal(rng, (1, 256, 3))

    mask_full = jnp.ones((1, 256), dtype=bool)
    mask_half = jnp.concatenate(
        [jnp.ones((1, 128), dtype=bool), jnp.zeros((1, 128), dtype=bool)],
        axis=1,
    )

    variables = encoder.init(rng, points, mask_full)
    out_full = encoder.apply(variables, points, mask_full)
    out_half = encoder.apply(variables, points, mask_half)
    assert not jnp.allclose(out_full, out_half), (
        "Full and half masks produced identical output -- masking may be broken"
    )


def test_pointnet_all_miss_returns_zero():
    """When all rays miss, the encoder should return the zero vector."""
    import jax
    import jax.numpy as jnp
    from jaxrl.pointnet import MaskAwarePointNet

    encoder = MaskAwarePointNet(hidden_dims=(32,), output_dim=8)
    rng = jax.random.PRNGKey(2)
    points = jax.random.normal(rng, (1, 64, 3))
    mask = jnp.zeros((1, 64), dtype=bool)

    variables = encoder.init(rng, points, mask)
    out = encoder.apply(variables, points, mask)
    assert jnp.allclose(out, 0.0), f"All-miss output should be zero, got {out}"


# -----------------------------------------------------------------------
# 2. Wrist-raycast tensor construction
# -----------------------------------------------------------------------

def test_raycast_tensor_shape_from_env():
    """Raycasts extracted from a live env should match the 32x32 contract."""
    from jaxrl.online_conditioner import OnlineRaycastConditioner, build_raycast_conditioner

    conditioner = build_raycast_conditioner(seed=0)
    env = _make_env("orange")
    points, masks = conditioner.extract_raycasts([env])
    assert points.shape == (1, 1024, 3), f"points shape {points.shape}"
    assert masks.shape == (1, 1024), f"masks shape {masks.shape}"
    assert masks.dtype == bool
    assert np.all(np.isfinite(points[masks])), "Non-finite valid points"
    env.close()


def test_raycast_tensor_normalized():
    """Palm-frame points should be divided by the normalization scale."""
    from jaxrl.online_conditioner import build_raycast_conditioner
    from jaxrl.pointnet import NORMALIZATION_SCALE

    conditioner = build_raycast_conditioner(seed=0)
    env = _make_env("orange")
    points, masks = conditioner.extract_raycasts([env])
    valid = points[masks]
    assert valid.shape[0] > 0, "No valid hits for orange"
    max_coord = np.abs(valid).max()
    assert max_coord < 10.0, (
        f"Max normalized coordinate {max_coord} seems too large; "
        f"normalization by {NORMALIZATION_SCALE} may not be applied"
    )
    env.close()


def test_raycast_invalid_rays_are_zero():
    """Invalid ray slots should carry zeros in the point tensor."""
    from jaxrl.online_conditioner import build_raycast_conditioner

    conditioner = build_raycast_conditioner(seed=0)
    env = _make_env("orange")
    points, masks = conditioner.extract_raycasts([env])
    invalid = points[0][~masks[0]]
    if invalid.shape[0] > 0:
        assert np.allclose(invalid, 0.0), "Invalid rays should be zero"
    env.close()


# -----------------------------------------------------------------------
# 3. Online conditioner end-to-end
# -----------------------------------------------------------------------

def test_online_conditioner_encode():
    """The full extract-and-encode path should return the right shape."""
    from jaxrl.online_conditioner import build_raycast_conditioner

    conditioner = build_raycast_conditioner(seed=0, output_dim=32)
    env = _make_env("orange")
    embeddings = conditioner.extract_and_encode([env])
    assert embeddings.shape == (1, 32), f"embeddings shape {embeddings.shape}"
    assert np.all(np.isfinite(embeddings)), "Non-finite embedding"
    env.close()


def test_online_conditioner_multi_env():
    """Online conditioner should handle multiple environments."""
    from jaxrl.online_conditioner import build_raycast_conditioner

    conditioner = build_raycast_conditioner(seed=0, output_dim=16)
    envs = [_make_env("orange", seed=1), _make_env("cube", seed=2)]
    embeddings = conditioner.extract_and_encode(envs)
    assert embeddings.shape == (2, 16), f"embeddings shape {embeddings.shape}"
    assert not np.allclose(embeddings[0], embeddings[1]), (
        "Different objects should produce different embeddings"
    )
    for e in envs:
        e.close()


def test_online_conditioner_changes_after_step():
    """Embeddings must change after an env step (online, not cached)."""
    from jaxrl.online_conditioner import build_raycast_conditioner
    import mujoco

    conditioner = build_raycast_conditioner(seed=0, output_dim=16)
    env = _make_env("orange", seed=42)
    emb1 = conditioner.extract_and_encode([env])

    action = env.action_space.sample()
    env.step(action)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    emb2 = conditioner.extract_and_encode([env])

    assert not np.allclose(emb1, emb2, atol=1e-6), (
        "Embeddings did not change after env step -- may be cached per object"
    )
    env.close()


# -----------------------------------------------------------------------
# 4. BRC wrist_raycast mode
# -----------------------------------------------------------------------

def test_brc_wrist_raycast_init():
    """BRC should initialize with conditioning_mode=wrist_raycast."""
    import jax.numpy as jnp
    from jaxrl.agent.brc_learner import BRC

    obs_dim = 68
    embed_dim = 32
    aug_dim = obs_dim + embed_dim
    obs_sample = jnp.zeros((1, aug_dim))
    act_sample = jnp.zeros((1, 20))

    agent = BRC(
        seed=0,
        observations=obs_sample,
        actions=act_sample,
        num_tasks=2,
        conditioning_mode="wrist_raycast",
    )
    assert agent.conditioning_mode == "wrist_raycast"
    assert agent.multitask is False
    assert agent.conditioner_features is None


def test_brc_wrist_raycast_no_task_embedding():
    """wrist_raycast mode must not contain learned task embedding params."""
    import jax
    import jax.numpy as jnp
    from jaxrl.agent.brc_learner import BRC

    obs_sample = jnp.zeros((1, 100))
    act_sample = jnp.zeros((1, 20))

    agent = BRC(
        seed=0, observations=obs_sample, actions=act_sample,
        num_tasks=2, conditioning_mode="wrist_raycast",
    )

    def param_paths(params):
        flat = jax.tree_util.tree_leaves_with_path(params)
        return ["/".join(str(k) for k in path) for path, _ in flat]

    critic_paths = param_paths(agent.critic.params)
    has_emb = any("task_embedding" in p for p in critic_paths)
    assert not has_emb, (
        f"wrist_raycast critic should not have task_embedding params, "
        f"found: {[p for p in critic_paths if 'task_embedding' in p]}"
    )


def test_brc_wrist_raycast_sample_actions():
    """BRC wrist_raycast should sample actions from augmented observations."""
    import jax.numpy as jnp
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.online_conditioner import build_raycast_conditioner

    obs_dim = 68
    embed_dim = 32
    conditioner = build_raycast_conditioner(seed=0, output_dim=embed_dim)

    env = _make_env("orange", seed=1)
    env2 = _make_env("cube", seed=2)

    embeddings = conditioner.extract_and_encode([env, env2])
    raw_obs = np.stack([
        np.concatenate([env.unwrapped.data.qpos[:obs_dim // 2],
                        env.unwrapped.data.qvel[:obs_dim // 2]]),
        np.concatenate([env2.unwrapped.data.qpos[:obs_dim // 2],
                        env2.unwrapped.data.qvel[:obs_dim // 2]]),
    ])

    from jaxrl.envs import ParallelEnv
    penv = ParallelEnv(["orange", "cube"], seed=0)
    raw_obs = penv.reset()
    embeddings = conditioner.extract_and_encode(penv.envs)
    aug_obs = np.concatenate([raw_obs, embeddings], axis=-1)

    agent = BRC(
        seed=0,
        observations=aug_obs[:1],
        actions=penv.action_space.sample()[:1],
        num_tasks=2,
        conditioning_mode="wrist_raycast",
    )
    actions = agent.sample_actions(aug_obs, temperature=1.0)
    assert actions.shape == (2, 20), f"actions shape {actions.shape}"
    assert np.all(np.isfinite(actions)), "Non-finite actions"

    for e in [env, env2]:
        e.close()


# -----------------------------------------------------------------------
# 5. Mesh interface: shape-only contract
# -----------------------------------------------------------------------

def test_mesh_shape_static_embedding():
    """Mesh-shape embeddings should be static (same mesh -> same features)."""
    from jaxrl.mesh_conditioner import build_conditioner_features
    import dex_envs

    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    f1, _ = build_conditioner_features(["orange", "cube"], assets_dir)
    f2, _ = build_conditioner_features(["orange", "cube"], assets_dir)
    assert np.allclose(f1, f2), "Mesh-shape features should be deterministic"


def test_mesh_shape_held_out_uses_train_stats():
    """Held-out mesh features should use train-split normalization."""
    from jaxrl.mesh_conditioner import (
        build_conditioner_features_from_manifest,
        load_manifest_objects,
    )
    import dex_envs

    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    test_objects = load_manifest_objects(MANIFEST_PATH, "test")[:2]
    features, meta = build_conditioner_features_from_manifest(
        test_objects, assets_dir, MANIFEST_PATH,
    )
    assert meta["normalize_source"] == "train_split"
    assert features.shape == (2, 8)


# -----------------------------------------------------------------------
# 6. Mesh-pose conditioner
# -----------------------------------------------------------------------

def test_mesh_pose_conditioner():
    """Mesh-pose should produce shape + pose embeddings from live envs."""
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
    assert conditioner.embed_dim == MESH_FEATURE_DIM + POSE_DIM

    envs = [_make_env("orange", seed=1), _make_env("cube", seed=2)]
    embeddings = conditioner.extract_and_encode(envs)
    assert embeddings.shape == (2, MESH_FEATURE_DIM + POSE_DIM)
    assert np.all(np.isfinite(embeddings))
    for e in envs:
        e.close()


def test_mesh_pose_changes_after_step():
    """Mesh-pose embeddings should change after env step (pose changes)."""
    from jaxrl.mesh_conditioner import MeshPoseConditioner, build_conditioner_features
    import dex_envs
    import mujoco

    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    features, _ = build_conditioner_features(["orange"], assets_dir)
    conditioner = MeshPoseConditioner(shape_features=features)

    env = _make_env("orange", seed=42)
    emb1 = conditioner.extract_and_encode([env])

    action = env.action_space.sample()
    env.step(action)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    emb2 = conditioner.extract_and_encode([env])

    shape_dim = features.shape[1]
    assert np.allclose(emb1[0, :shape_dim], emb2[0, :shape_dim]), (
        "Shape part should be static"
    )
    assert not np.allclose(emb1[0, shape_dim:], emb2[0, shape_dim:], atol=1e-6), (
        "Pose part should change after step"
    )
    env.close()


def test_mesh_pose_design_documented():
    """The mesh-pose design decision should be documented in the module."""
    import jaxrl.mesh_conditioner as mc
    doc = mc.__doc__
    assert "structured append" in doc.lower(), "Design decision not in docstring"
    assert "privileged" in doc.lower(), "Privileged label not in docstring"


# -----------------------------------------------------------------------
# 7. BRC mesh_pose mode
# -----------------------------------------------------------------------

def test_brc_mesh_pose_init():
    """BRC should initialize with conditioning_mode=mesh_pose."""
    import jax.numpy as jnp
    from jaxrl.agent.brc_learner import BRC

    obs_sample = jnp.zeros((1, 68 + 17))
    act_sample = jnp.zeros((1, 20))

    agent = BRC(
        seed=0, observations=obs_sample, actions=act_sample,
        num_tasks=2, conditioning_mode="mesh_pose",
    )
    assert agent.conditioning_mode == "mesh_pose"
    assert agent.multitask is False
    assert agent.conditioner_features is None


# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("PointNet imports and constants", test_pointnet_imports),
        ("PointNet forward pass shape", test_pointnet_forward_pass),
        ("PointNet mask affects output", test_pointnet_mask_affects_output),
        ("PointNet all-miss returns zero", test_pointnet_all_miss_returns_zero),
        ("Raycast tensor shape from live env", test_raycast_tensor_shape_from_env),
        ("Raycast points are normalized", test_raycast_tensor_normalized),
        ("Invalid ray slots are zero", test_raycast_invalid_rays_are_zero),
        ("Online conditioner encode shape", test_online_conditioner_encode),
        ("Online conditioner multi-env", test_online_conditioner_multi_env),
        ("Online embeddings change after step", test_online_conditioner_changes_after_step),
        ("BRC initializes with wrist_raycast", test_brc_wrist_raycast_init),
        ("No task embedding in wrist_raycast", test_brc_wrist_raycast_no_task_embedding),
        ("BRC wrist_raycast sample actions", test_brc_wrist_raycast_sample_actions),
        ("Mesh-shape embeddings are static", test_mesh_shape_static_embedding),
        ("Held-out mesh uses train-split stats", test_mesh_shape_held_out_uses_train_stats),
        ("Mesh-pose conditioner shape and values", test_mesh_pose_conditioner),
        ("Mesh-pose changes after step", test_mesh_pose_changes_after_step),
        ("Mesh-pose design decision documented", test_mesh_pose_design_documented),
        ("BRC initializes with mesh_pose", test_brc_mesh_pose_init),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\nRESULTS: {passed} passed, {failed} failed out of {test_num}")
    sys.exit(1 if failed > 0 else 0)
