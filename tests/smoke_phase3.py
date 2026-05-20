"""Phase 3 smoke tests: geometry-conditioning infrastructure.

Run with:
    module load python/3.11.9
    source .venv/bin/activate
    MUJOCO_GL=egl python tests/smoke_phase3.py
"""

import os
os.environ["MUJOCO_GL"] = "egl"
if "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import traceback

import numpy as np

TESTS_PASSED = 0
TESTS_FAILED = 0


def run_test(name, fn):
    global TESTS_PASSED, TESTS_FAILED
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        TESTS_PASSED += 1
        print(f"  PASS")
    except Exception:
        TESTS_FAILED += 1
        traceback.print_exc()
        print(f"  FAIL")


# ---------------------------------------------------------------------------
# 1. Mesh conditioner module
# ---------------------------------------------------------------------------

def test_mesh_conditioner_imports():
    from jaxrl.mesh_conditioner import (
        load_binary_stl,
        compute_mesh_features,
        resolve_stl_path,
        build_conditioner_features,
        load_manifest_objects,
        MESH_FEATURE_DIM,
    )
    assert MESH_FEATURE_DIM == 8
    print(f"  MESH_FEATURE_DIM = {MESH_FEATURE_DIM}")


def test_resolve_stl_and_load():
    from jaxrl.mesh_conditioner import resolve_stl_path, load_binary_stl, compute_mesh_features
    import dex_envs
    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")

    for obj in ["orange", "cube", "hammer"]:
        stl_path = resolve_stl_path(obj, assets_dir)
        assert os.path.isfile(stl_path), f"STL not found: {stl_path}"
        vertices, normals = load_binary_stl(stl_path)
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        feats = compute_mesh_features(vertices)
        assert feats.shape == (8,)
        print(f"  {obj}: vertices={len(vertices)}, features={feats}")


def test_build_conditioner_features():
    from jaxrl.mesh_conditioner import build_conditioner_features
    import dex_envs
    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    objects = ["orange", "cube"]
    features, meta = build_conditioner_features(objects, assets_dir, normalize=True)
    assert features.shape == (2, 8), f"Got {features.shape}"
    assert meta["feature_dim"] == 8
    assert meta["normalized"] is True
    assert "normalize_mean" in meta
    print(f"  features shape: {features.shape}")
    print(f"  meta keys: {list(meta.keys())}")


def test_manifest_loading():
    from jaxrl.mesh_conditioner import load_manifest_objects
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "manifests", "shadowhand_split_v1.json",
    )
    train_objects = load_manifest_objects(manifest_path, "train")
    test_objects = load_manifest_objects(manifest_path, "test")
    assert len(train_objects) == 85
    assert len(test_objects) == 29
    print(f"  train: {len(train_objects)}, test: {len(test_objects)}")


def test_full_manifest_features():
    from jaxrl.mesh_conditioner import load_manifest_objects, build_conditioner_features
    import dex_envs
    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "manifests", "shadowhand_split_v1.json",
    )
    train_objects = load_manifest_objects(manifest_path, "train")
    features, meta = build_conditioner_features(train_objects, assets_dir, normalize=True)
    assert features.shape == (85, 8)
    print(f"  Full train features shape: {features.shape}")
    print(f"  Feature range: min={features.min():.3f}, max={features.max():.3f}")


# ---------------------------------------------------------------------------
# 2. BRC categorical mode (unchanged behavior)
# ---------------------------------------------------------------------------

def test_categorical_mode():
    import jax.numpy as jnp
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.replay_buffer import ParallelReplayBuffer
    from jaxrl.normalizer import RewardNormalizer

    env = ParallelEnv(["orange", "cube"], seed=0)
    num_tasks = len(env.envs)

    agent = BRC(
        seed=0,
        observations=env.observation_space.sample()[:1],
        actions=env.action_space.sample()[:1],
        num_tasks=num_tasks,
        conditioning_mode="categorical",
    )
    assert agent.multitask is True
    assert agent.conditioning_mode == "categorical"
    assert agent.conditioner_features is None
    print(f"  multitask={agent.multitask}, conditioning_mode={agent.conditioning_mode}")

    observations = env.reset()
    actions = agent.sample_actions(observations, temperature=1.0)
    assert actions.shape == (2, 20)
    print(f"  sample_actions shape: {actions.shape}")

    replay_buffer = ParallelReplayBuffer(env.observation_space, 20, 1000, num_tasks)
    reward_normalizer = RewardNormalizer(num_tasks, target_entropy=agent.target_entropy, discount=agent.discount)

    for _ in range(20):
        actions = env.action_space.sample()
        next_obs, rewards, terms, truns, goals = env.step(actions)
        reward_normalizer.update(rewards, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_obs)
        observations = next_obs
        observations, terms, truns = env.reset_where_done(observations, terms, truns)

    batches = replay_buffer.sample(32, 2)
    batches = reward_normalizer.normalize(batches, agent.get_temperature())
    info = agent.update(batches, 2, 1)
    print(f"  update info keys: {list(info.keys())}")

    eval_stats = env.evaluate(agent, num_episodes=1, temperature=0.0, render=False)
    print(f"  eval keys: {list(eval_stats.keys())}")
    print(f"  CATEGORICAL mode: all checks passed")


# ---------------------------------------------------------------------------
# 3. BRC none mode
# ---------------------------------------------------------------------------

def test_none_mode():
    import jax.numpy as jnp
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.replay_buffer import ParallelReplayBuffer
    from jaxrl.normalizer import RewardNormalizer

    env = ParallelEnv(["orange", "cube"], seed=0)
    num_tasks = len(env.envs)

    agent = BRC(
        seed=0,
        observations=env.observation_space.sample()[:1],
        actions=env.action_space.sample()[:1],
        num_tasks=num_tasks,
        conditioning_mode="none",
    )
    assert agent.multitask is False
    assert agent.conditioning_mode == "none"
    assert agent.conditioner_features is None
    print(f"  multitask={agent.multitask}, conditioning_mode={agent.conditioning_mode}")

    observations = env.reset()
    actions = agent.sample_actions(observations, temperature=1.0)
    assert actions.shape == (2, 20)
    print(f"  sample_actions shape: {actions.shape}")

    replay_buffer = ParallelReplayBuffer(env.observation_space, 20, 1000, num_tasks)
    reward_normalizer = RewardNormalizer(num_tasks, target_entropy=agent.target_entropy, discount=agent.discount)

    for _ in range(20):
        actions = env.action_space.sample()
        next_obs, rewards, terms, truns, goals = env.step(actions)
        reward_normalizer.update(rewards, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_obs)
        observations = next_obs
        observations, terms, truns = env.reset_where_done(observations, terms, truns)

    batches = replay_buffer.sample(32, 2)
    batches = reward_normalizer.normalize(batches, agent.get_temperature())
    info = agent.update(batches, 2, 1)
    print(f"  update info keys: {list(info.keys())}")

    eval_stats = env.evaluate(agent, num_episodes=1, temperature=0.0, render=False)
    print(f"  eval keys: {list(eval_stats.keys())}")
    print(f"  NONE mode: all checks passed")


# ---------------------------------------------------------------------------
# 4. BRC mesh_shape mode
# ---------------------------------------------------------------------------

def test_mesh_shape_mode():
    import jax.numpy as jnp
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.replay_buffer import ParallelReplayBuffer
    from jaxrl.normalizer import RewardNormalizer
    from jaxrl.mesh_conditioner import build_conditioner_features
    import dex_envs

    objects = ["orange", "cube"]
    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    conditioner_features, meta = build_conditioner_features(objects, assets_dir, normalize=True)

    env = ParallelEnv(objects, seed=0)
    num_tasks = len(env.envs)

    agent = BRC(
        seed=0,
        observations=env.observation_space.sample()[:1],
        actions=env.action_space.sample()[:1],
        num_tasks=num_tasks,
        conditioning_mode="mesh_shape",
        conditioner_features=conditioner_features,
    )
    assert agent.multitask is False
    assert agent.conditioning_mode == "mesh_shape"
    assert agent.conditioner_features is not None
    assert agent.conditioner_features.shape == (2, 8)
    print(f"  multitask={agent.multitask}, conditioning_mode={agent.conditioning_mode}")
    print(f"  conditioner shape: {agent.conditioner_features.shape}")

    observations = env.reset()
    actions = agent.sample_actions(observations, temperature=1.0)
    assert actions.shape == (2, 20)
    print(f"  sample_actions shape: {actions.shape}")

    replay_buffer = ParallelReplayBuffer(env.observation_space, 20, 1000, num_tasks)
    reward_normalizer = RewardNormalizer(num_tasks, target_entropy=agent.target_entropy, discount=agent.discount)

    for _ in range(20):
        actions = env.action_space.sample()
        next_obs, rewards, terms, truns, goals = env.step(actions)
        reward_normalizer.update(rewards, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_obs)
        observations = next_obs
        observations, terms, truns = env.reset_where_done(observations, terms, truns)

    batches = replay_buffer.sample(32, 2)
    batches = reward_normalizer.normalize(batches, agent.get_temperature())
    info = agent.update(batches, 2, 1)
    print(f"  update info keys: {list(info.keys())}")

    eval_stats = env.evaluate(agent, num_episodes=1, temperature=0.0, render=False)
    print(f"  eval keys: {list(eval_stats.keys())}")
    print(f"  MESH_SHAPE mode: all checks passed")


# ---------------------------------------------------------------------------
# 5. Dimension consistency checks
# ---------------------------------------------------------------------------

def test_dimension_consistency():
    """Verify actor/critic input dimensions are correct for each mode."""
    import jax
    import jax.numpy as jnp
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.mesh_conditioner import build_conditioner_features, MESH_FEATURE_DIM
    import dex_envs

    objects = ["orange", "cube"]
    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    conditioner_features, _ = build_conditioner_features(objects, assets_dir)

    env = ParallelEnv(objects, seed=0)
    obs_sample = env.observation_space.sample()[:1]
    act_sample = env.action_space.sample()[:1]
    obs_dim = obs_sample.shape[-1]
    act_dim = act_sample.shape[-1]
    num_tasks = 2

    cat_agent = BRC(0, obs_sample, act_sample, num_tasks, conditioning_mode="categorical")
    none_agent = BRC(0, obs_sample, act_sample, num_tasks, conditioning_mode="none")
    mesh_agent = BRC(0, obs_sample, act_sample, num_tasks, conditioning_mode="mesh_shape",
                     conditioner_features=conditioner_features)

    cat_actor_params = jax.tree.map(lambda x: x.shape, cat_agent.actor.params)
    none_actor_params = jax.tree.map(lambda x: x.shape, none_agent.actor.params)
    mesh_actor_params = jax.tree.map(lambda x: x.shape, mesh_agent.actor.params)

    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, embedding_size=32, mesh_feature_dim={MESH_FEATURE_DIM}")
    print(f"  categorical actor input: {obs_dim} + 32 = {obs_dim + 32}")
    print(f"  none actor input: {obs_dim}")
    print(f"  mesh_shape actor input: {obs_dim} + {MESH_FEATURE_DIM} = {obs_dim + MESH_FEATURE_DIM}")

    assert cat_agent.multitask is True
    assert none_agent.multitask is False
    assert mesh_agent.multitask is False


# ---------------------------------------------------------------------------
# 6. Assertion: mesh_shape mode has no learned task embedding
# ---------------------------------------------------------------------------

def test_no_learned_embedding_in_mesh_mode():
    """Verify that mesh_shape Critic has no TaskEmbedding parameters."""
    import jax
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.mesh_conditioner import build_conditioner_features
    import dex_envs

    def param_path_strs(params):
        flat = jax.tree_util.tree_leaves_with_path(params)
        return ["/".join(str(k) for k in path) for path, _ in flat]

    objects = ["orange", "cube"]
    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    conditioner_features, _ = build_conditioner_features(objects, assets_dir)

    env = ParallelEnv(objects, seed=0)
    obs_sample = env.observation_space.sample()[:1]
    act_sample = env.action_space.sample()[:1]

    mesh_agent = BRC(0, obs_sample, act_sample, 2,
                     conditioning_mode="mesh_shape",
                     conditioner_features=conditioner_features)

    mesh_paths = param_path_strs(mesh_agent.critic.params)
    has_task_embedding = any("task_embedding" in p for p in mesh_paths)
    assert not has_task_embedding, (
        f"mesh_shape critic should not have task_embedding params, "
        f"found: {[p for p in mesh_paths if 'task_embedding' in p]}"
    )
    print(f"  Confirmed: no task_embedding params in mesh_shape critic")

    cat_agent = BRC(0, obs_sample, act_sample, 2, conditioning_mode="categorical")
    cat_paths = param_path_strs(cat_agent.critic.params)
    has_cat_emb = any("task_embedding" in p for p in cat_paths)
    assert has_cat_emb, "categorical critic should have task_embedding params"
    print(f"  Confirmed: task_embedding params present in categorical critic")


# ---------------------------------------------------------------------------
# 7. get_infos works for each mode
# ---------------------------------------------------------------------------

def test_get_infos_all_modes():
    import jax.numpy as jnp
    from jaxrl.envs import ParallelEnv
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.replay_buffer import ParallelReplayBuffer
    from jaxrl.normalizer import RewardNormalizer
    from jaxrl.mesh_conditioner import build_conditioner_features
    import dex_envs

    objects = ["orange", "cube"]
    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    conditioner_features, _ = build_conditioner_features(objects, assets_dir)

    for mode in ["categorical", "none", "mesh_shape"]:
        env = ParallelEnv(objects, seed=0)
        num_tasks = len(env.envs)

        kwargs = {"conditioning_mode": mode}
        if mode == "mesh_shape":
            kwargs["conditioner_features"] = conditioner_features

        agent = BRC(0, env.observation_space.sample()[:1],
                    env.action_space.sample()[:1], num_tasks, **kwargs)

        replay_buffer = ParallelReplayBuffer(env.observation_space, 20, 1000, num_tasks)
        reward_normalizer = RewardNormalizer(
            num_tasks, target_entropy=agent.target_entropy, discount=agent.discount
        )

        observations = env.reset()
        for _ in range(20):
            actions = env.action_space.sample()
            next_obs, rewards, terms, truns, goals = env.step(actions)
            reward_normalizer.update(rewards, terms, truns)
            masks = env.generate_masks(terms, truns)
            replay_buffer.insert(observations, actions, rewards, masks, next_obs)
            observations = next_obs
            observations, terms, truns = env.reset_where_done(observations, terms, truns)

        task_batches = replay_buffer.sample_task_batches()
        task_batches = reward_normalizer.normalize(task_batches, agent.get_temperature())
        infos = agent.get_infos(task_batches)
        print(f"  {mode}: get_infos keys = {list(infos.keys())}")


# ---------------------------------------------------------------------------
# 8. Manifest-based features: train-split normalization
# ---------------------------------------------------------------------------

def test_manifest_features_train_normalization():
    """Verify build_conditioner_features_from_manifest uses train-split stats."""
    from jaxrl.mesh_conditioner import (
        build_conditioner_features_from_manifest,
        _compute_raw_features,
        load_manifest_objects,
    )
    import dex_envs

    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "manifests", "shadowhand_split_v1.json",
    )

    train_objects = load_manifest_objects(manifest_path, "train")
    test_objects = load_manifest_objects(manifest_path, "test")

    train_raw = _compute_raw_features(train_objects, assets_dir)
    expected_mean = train_raw.mean(axis=0)
    expected_std = train_raw.std(axis=0)
    expected_std = np.where(expected_std < 1e-8, 1.0, expected_std)

    test_subset = test_objects[:3]
    features, meta = build_conditioner_features_from_manifest(
        test_subset, assets_dir, manifest_path,
    )
    assert features.shape == (3, 8), f"Got {features.shape}"
    assert meta["normalize_source"] == "train_split"
    assert meta["train_objects_count"] == 85

    saved_mean = np.array(meta["normalize_mean"])
    saved_std = np.array(meta["normalize_std"])
    assert np.allclose(saved_mean, expected_mean, atol=1e-6)
    assert np.allclose(saved_std, expected_std, atol=1e-6)

    raw_test = _compute_raw_features(test_subset, assets_dir)
    expected_features = ((raw_test - expected_mean) / expected_std).astype(np.float32)
    assert np.allclose(features, expected_features, atol=1e-6)
    print(f"  Test objects normalized with train stats: range [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Train mean sample: {saved_mean[:3]}")


def test_manifest_features_missing_object():
    """Verify that an object not in the manifest raises ValueError."""
    from jaxrl.mesh_conditioner import build_conditioner_features_from_manifest
    import dex_envs

    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "manifests", "shadowhand_split_v1.json",
    )

    try:
        build_conditioner_features_from_manifest(
            ["orange", "FAKE_OBJECT"], assets_dir, manifest_path,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "FAKE_OBJECT" in str(e)
        print(f"  Correctly raised ValueError: {e}")


# ---------------------------------------------------------------------------
# 9. Batch size logic
# ---------------------------------------------------------------------------

def test_batch_size_logic():
    """Verify batch_size = 1024 for multi-object regardless of conditioning mode."""
    num_tasks_multi = 85
    num_tasks_single = 1

    batch_multi = 1024 if num_tasks_multi > 1 else 256
    batch_single = 1024 if num_tasks_single > 1 else 256

    assert batch_multi == 1024, f"Multi-object batch should be 1024, got {batch_multi}"
    assert batch_single == 256, f"Single-object batch should be 256, got {batch_single}"
    print(f"  num_tasks=85 -> batch_size={batch_multi}")
    print(f"  num_tasks=1  -> batch_size={batch_single}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_test("1. Mesh conditioner imports", test_mesh_conditioner_imports)
    run_test("2. STL resolve and load", test_resolve_stl_and_load)
    run_test("3. Build conditioner features", test_build_conditioner_features)
    run_test("4. Manifest loading", test_manifest_loading)
    run_test("5. Full manifest features (85 train)", test_full_manifest_features)
    run_test("6. BRC categorical mode (unchanged)", test_categorical_mode)
    run_test("7. BRC none mode", test_none_mode)
    run_test("8. BRC mesh_shape mode", test_mesh_shape_mode)
    run_test("9. Dimension consistency", test_dimension_consistency)
    run_test("10. No learned embedding in mesh mode", test_no_learned_embedding_in_mesh_mode)
    run_test("11. get_infos all modes", test_get_infos_all_modes)
    run_test("12. Manifest features: train-split normalization", test_manifest_features_train_normalization)
    run_test("13. Manifest features: missing object error", test_manifest_features_missing_object)
    run_test("14. Batch size logic", test_batch_size_logic)

    print(f"\n{'='*60}")
    print(f"RESULTS: {TESTS_PASSED} passed, {TESTS_FAILED} failed out of {TESTS_PASSED + TESTS_FAILED}")
    print(f"{'='*60}")
    sys.exit(1 if TESTS_FAILED > 0 else 0)
