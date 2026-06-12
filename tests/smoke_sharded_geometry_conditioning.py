"""Smoke tests for geometry conditioning with sharded (slot-based) environments.

These tests exercise the shape logic of mesh_shape and mesh_pose modes
when num_slots != num_objects — the key scenario for mjlab_sharded where
e.g. 85 objects × 64 envs/object = 5440 slots.

No mjlab or MuJoCo is required.  All tests use synthetic observations,
conditioner features, and object ID mappings.

Run with:
    JAX_PLATFORM_NAME=cpu python tests/smoke_sharded_geometry_conditioning.py
"""

import os
os.environ["MUJOCO_GL"] = "egl"
if "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import traceback

import numpy as np
import jax.numpy as jnp

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
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_setup(num_objects=3, slots_per_object=4, obs_dim=42, act_dim=20, feature_dim=8):
    num_slots = num_objects * slots_per_object
    object_ids = np.repeat(np.arange(num_objects, dtype=np.int32), slots_per_object)
    observations = np.random.randn(num_slots, obs_dim).astype(np.float64)
    conditioner_features = np.random.randn(num_objects, feature_dim).astype(np.float32)
    return {
        "num_objects": num_objects,
        "num_slots": num_slots,
        "slots_per_object": slots_per_object,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "feature_dim": feature_dim,
        "object_ids": object_ids,
        "observations": observations,
        "conditioner_features": conditioner_features,
    }


# ---------------------------------------------------------------------------
# 1. BRC.sample_actions with slot-level task_ids (mesh_shape)
# ---------------------------------------------------------------------------

def test_sample_actions_slot_task_ids():
    """sample_actions with slot-level task_ids should produce (num_slots, act_dim)."""
    from jaxrl.agent.brc_learner import BRC

    s = make_synthetic_setup(num_objects=3, slots_per_object=4)
    obs_sample = np.zeros((1, s["obs_dim"]), dtype=np.float64)
    act_sample = np.zeros((1, s["act_dim"]), dtype=np.float64)

    agent = BRC(
        seed=0,
        observations=obs_sample,
        actions=act_sample,
        num_tasks=s["num_objects"],
        conditioning_mode="mesh_shape",
        conditioner_features=s["conditioner_features"],
    )

    slot_task_ids = jnp.array(s["object_ids"], dtype=jnp.int32)
    actions = agent.sample_actions(s["observations"], temperature=1.0, task_ids=slot_task_ids)

    assert actions.shape == (s["num_slots"], s["act_dim"]), \
        f"Expected ({s['num_slots']}, {s['act_dim']}), got {actions.shape}"
    assert np.all(np.isfinite(actions))
    print(f"  actions shape: {actions.shape}")


def test_sample_actions_default_task_ids():
    """sample_actions without task_ids should use self.task_ids (num_objects,)."""
    from jaxrl.agent.brc_learner import BRC

    s = make_synthetic_setup(num_objects=3, slots_per_object=1)
    obs_sample = np.zeros((1, s["obs_dim"]), dtype=np.float64)
    act_sample = np.zeros((1, s["act_dim"]), dtype=np.float64)

    agent = BRC(
        seed=0,
        observations=obs_sample,
        actions=act_sample,
        num_tasks=s["num_objects"],
        conditioning_mode="mesh_shape",
        conditioner_features=s["conditioner_features"],
    )

    actions = agent.sample_actions(s["observations"], temperature=1.0)
    assert actions.shape == (s["num_objects"], s["act_dim"]), \
        f"Expected ({s['num_objects']}, {s['act_dim']}), got {actions.shape}"
    print(f"  default task_ids: actions shape: {actions.shape}")


# ---------------------------------------------------------------------------
# 2. _augment_obs with slot-level indexing
# ---------------------------------------------------------------------------

def test_augment_obs_slot_indexing():
    """_augment_obs should index conditioner_features by slot-level task_ids."""
    from jaxrl.agent.brc_learner import BRC

    s = make_synthetic_setup(num_objects=3, slots_per_object=4, feature_dim=8)
    obs_sample = np.zeros((1, s["obs_dim"]), dtype=np.float64)
    act_sample = np.zeros((1, s["act_dim"]), dtype=np.float64)

    agent = BRC(
        seed=0,
        observations=obs_sample,
        actions=act_sample,
        num_tasks=s["num_objects"],
        conditioning_mode="mesh_shape",
        conditioner_features=s["conditioner_features"],
    )

    slot_ids = jnp.array(s["object_ids"], dtype=jnp.int32)
    obs_jnp = jnp.array(s["observations"])
    augmented = agent._augment_obs(obs_jnp, slot_ids)

    expected_shape = (s["num_slots"], s["obs_dim"] + s["feature_dim"])
    assert augmented.shape == expected_shape, \
        f"Expected {expected_shape}, got {augmented.shape}"

    for slot_idx in range(s["num_slots"]):
        obj_id = s["object_ids"][slot_idx]
        expected_feat = s["conditioner_features"][obj_id]
        actual_feat = np.asarray(augmented[slot_idx, s["obs_dim"]:])
        assert np.allclose(actual_feat, expected_feat, atol=1e-6), \
            f"Slot {slot_idx} (obj {obj_id}): feature mismatch"

    print(f"  augmented shape: {augmented.shape}, all slot features verified")


# ---------------------------------------------------------------------------
# 3. _augment_batch with replay task_ids in [0, num_objects)
# ---------------------------------------------------------------------------

def test_augment_batch_replay_task_ids():
    """_augment_batch should index features by batch.task_ids from replay."""
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.utils import Batch

    s = make_synthetic_setup(num_objects=3, slots_per_object=4, feature_dim=8)
    obs_sample = np.zeros((1, s["obs_dim"]), dtype=np.float64)
    act_sample = np.zeros((1, s["act_dim"]), dtype=np.float64)

    agent = BRC(
        seed=0,
        observations=obs_sample,
        actions=act_sample,
        num_tasks=s["num_objects"],
        conditioning_mode="mesh_shape",
        conditioner_features=s["conditioner_features"],
    )

    batch_size = 32
    replay_task_ids = np.random.randint(0, s["num_objects"], size=batch_size).astype(np.int32)
    batch = Batch(
        observations=jnp.zeros((batch_size, s["obs_dim"])),
        actions=jnp.zeros((batch_size, s["act_dim"])),
        rewards=jnp.zeros(batch_size),
        masks=jnp.ones(batch_size),
        next_observations=jnp.zeros((batch_size, s["obs_dim"])),
        task_ids=jnp.array(replay_task_ids),
    )

    aug_batch = agent._augment_batch(batch)
    assert aug_batch.observations.shape == (batch_size, s["obs_dim"] + s["feature_dim"])
    assert aug_batch.next_observations.shape == (batch_size, s["obs_dim"] + s["feature_dim"])

    for i in range(batch_size):
        obj_id = replay_task_ids[i]
        expected_feat = s["conditioner_features"][obj_id]
        actual_feat = np.asarray(aug_batch.observations[i, s["obs_dim"]:])
        assert np.allclose(actual_feat, expected_feat, atol=1e-6)

    print(f"  batch augmented: obs shape {aug_batch.observations.shape}")


# ---------------------------------------------------------------------------
# 4. Full sample + update cycle with slot-level task_ids
# ---------------------------------------------------------------------------

def test_full_sample_update_cycle():
    """BRC mesh_shape should handle sample(slot_ids) + update(replay_ids) without shape errors."""
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.replay_buffer import ObjectAwareReplayBuffer
    from jaxrl.normalizer import ObjectAwareRewardNormalizer

    num_objects = 3
    slots_per_object = 4
    num_slots = num_objects * slots_per_object
    obs_dim = 42
    act_dim = 20
    feature_dim = 8

    object_ids = np.repeat(np.arange(num_objects, dtype=np.int32), slots_per_object)
    features = np.random.randn(num_objects, feature_dim).astype(np.float32)

    obs_sample = np.zeros((1, obs_dim), dtype=np.float64)
    act_sample = np.zeros((1, act_dim), dtype=np.float64)

    agent = BRC(
        seed=0,
        observations=obs_sample,
        actions=act_sample,
        num_tasks=num_objects,
        conditioning_mode="mesh_shape",
        conditioner_features=features,
    )

    class FakeObsSpace:
        def __init__(self):
            self.shape = (num_slots, obs_dim)
            self.dtype = np.float64

    replay_buffer = ObjectAwareReplayBuffer(
        FakeObsSpace(), act_dim, capacity=200,
        num_objects=num_objects, slot_to_object=object_ids,
    )
    reward_normalizer = ObjectAwareRewardNormalizer(
        num_objects, num_slots, object_ids,
        target_entropy=agent.target_entropy, discount=agent.discount,
    )

    slot_task_ids = jnp.array(object_ids, dtype=jnp.int32)
    observations = np.random.randn(num_slots, obs_dim).astype(np.float64)

    for step in range(30):
        if step < 10:
            actions = np.random.uniform(-1, 1, (num_slots, act_dim)).astype(np.float64)
        else:
            actions = agent.sample_actions(observations, temperature=1.0, task_ids=slot_task_ids)

        next_obs = np.random.randn(num_slots, obs_dim).astype(np.float64)
        rewards = np.random.randn(num_slots).astype(np.float64)
        terms = np.zeros(num_slots, dtype=np.float64)
        truns = np.zeros(num_slots, dtype=np.float64)

        reward_normalizer.update(rewards, terms, truns)
        masks = 1 - (terms * (1 - truns))
        replay_buffer.insert(observations, actions, rewards, masks, next_obs)
        observations = next_obs

        if step >= 10:
            batches = replay_buffer.sample(32, 2)
            assert batches.task_ids.max() < num_objects, \
                f"Replay task_ids should be in [0, {num_objects}), max={batches.task_ids.max()}"
            batches = reward_normalizer.normalize(batches, agent.get_temperature())
            info = agent.update(batches, 2, step)

    print(f"  30-step cycle completed, last update info keys: {list(info.keys())}")


# ---------------------------------------------------------------------------
# 5. 85-object shape check (no actual training, just construction + sample)
# ---------------------------------------------------------------------------

def test_85_object_shapes():
    """85 objects × 64 slots/object should produce correct shapes throughout."""
    from jaxrl.agent.brc_learner import BRC

    num_objects = 85
    slots_per_object = 64
    num_slots = num_objects * slots_per_object
    obs_dim = 42
    act_dim = 20
    feature_dim = 8

    object_ids = np.repeat(np.arange(num_objects, dtype=np.int32), slots_per_object)
    features = np.random.randn(num_objects, feature_dim).astype(np.float32)

    obs_sample = np.zeros((1, obs_dim), dtype=np.float64)
    act_sample = np.zeros((1, act_dim), dtype=np.float64)

    agent = BRC(
        seed=0,
        observations=obs_sample,
        actions=act_sample,
        num_tasks=num_objects,
        conditioning_mode="mesh_shape",
        conditioner_features=features,
    )

    assert agent.conditioner_features.shape == (85, 8)
    assert agent.task_ids.shape == (85,)

    observations = np.random.randn(num_slots, obs_dim).astype(np.float64)
    slot_task_ids = jnp.array(object_ids, dtype=jnp.int32)

    actions = agent.sample_actions(observations, temperature=1.0, task_ids=slot_task_ids)
    assert actions.shape == (num_slots, act_dim), \
        f"Expected ({num_slots}, {act_dim}), got {actions.shape}"

    print(f"  85×64 = {num_slots} slots: sample_actions → {actions.shape}")


# ---------------------------------------------------------------------------
# 6. get_infos with ObjectAwareReplayBuffer task_ids
# ---------------------------------------------------------------------------

def test_get_infos_mesh_shape_sharded():
    """get_infos should work with object-aware replay task_ids."""
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.replay_buffer import ObjectAwareReplayBuffer
    from jaxrl.normalizer import ObjectAwareRewardNormalizer

    num_objects = 3
    slots_per_object = 4
    num_slots = num_objects * slots_per_object
    obs_dim = 42
    act_dim = 20
    feature_dim = 8

    object_ids = np.repeat(np.arange(num_objects, dtype=np.int32), slots_per_object)
    features = np.random.randn(num_objects, feature_dim).astype(np.float32)

    agent = BRC(
        seed=0,
        observations=np.zeros((1, obs_dim), dtype=np.float64),
        actions=np.zeros((1, act_dim), dtype=np.float64),
        num_tasks=num_objects,
        conditioning_mode="mesh_shape",
        conditioner_features=features,
    )

    class FakeObsSpace:
        def __init__(self):
            self.shape = (num_slots, obs_dim)
            self.dtype = np.float64

    replay = ObjectAwareReplayBuffer(
        FakeObsSpace(), act_dim, 200, num_objects, object_ids,
    )
    normalizer = ObjectAwareRewardNormalizer(
        num_objects, num_slots, object_ids,
        target_entropy=agent.target_entropy, discount=agent.discount,
    )

    observations = np.random.randn(num_slots, obs_dim).astype(np.float64)
    for _ in range(20):
        actions = np.random.uniform(-1, 1, (num_slots, act_dim)).astype(np.float64)
        next_obs = np.random.randn(num_slots, obs_dim).astype(np.float64)
        rewards = np.random.randn(num_slots).astype(np.float64)
        terms = np.zeros(num_slots, dtype=np.float64)
        truns = np.zeros(num_slots, dtype=np.float64)
        normalizer.update(rewards, terms, truns)
        masks = 1 - (terms * (1 - truns))
        replay.insert(observations, actions, rewards, masks, next_obs)
        observations = next_obs

    task_batches = replay.sample_task_batches()
    task_batches = normalizer.normalize(task_batches, agent.get_temperature())
    infos = agent.get_infos(task_batches)
    print(f"  get_infos keys: {list(infos.keys())}")


# ---------------------------------------------------------------------------
# 7. No categorical embedding reintroduction
# ---------------------------------------------------------------------------

def test_no_categorical_with_mesh_shape():
    """mesh_shape mode must have multitask=False (no categorical embedding)."""
    from jaxrl.agent.brc_learner import BRC
    import jax

    features = np.random.randn(3, 8).astype(np.float32)
    agent = BRC(
        seed=0,
        observations=np.zeros((1, 42), dtype=np.float64),
        actions=np.zeros((1, 20), dtype=np.float64),
        num_tasks=3,
        conditioning_mode="mesh_shape",
        conditioner_features=features,
    )

    assert agent.multitask is False
    assert agent.conditioner_features is not None
    assert agent.conditioning_mode == "mesh_shape"

    critic_paths = [
        "/".join(str(k) for k in path)
        for path, _ in jax.tree_util.tree_leaves_with_path(agent.critic.params)
    ]
    has_task_emb = any("task_embedding" in p for p in critic_paths)
    assert not has_task_emb, f"Found task_embedding in mesh_shape critic: {[p for p in critic_paths if 'task_embedding' in p]}"
    print(f"  multitask={agent.multitask}, no task_embedding in critic")


# ---------------------------------------------------------------------------
# 8. ShardedMeshPoseConditioner shape logic (synthetic)
# ---------------------------------------------------------------------------

def test_sharded_mesh_pose_conditioner_shapes():
    """ShardedMeshPoseConditioner should produce (num_slots, embed_dim) embeddings."""
    from jaxrl.mesh_conditioner import ShardedMeshPoseConditioner, POSE_DIM

    num_objects = 3
    slots_per_object = 4
    num_slots = num_objects * slots_per_object
    shape_dim = 8

    object_ids = np.repeat(np.arange(num_objects, dtype=np.int32), slots_per_object)
    shape_features = np.random.randn(num_objects, shape_dim).astype(np.float32)

    cond = ShardedMeshPoseConditioner(
        shape_features=shape_features,
        object_ids=object_ids,
    )
    assert cond.embed_dim == shape_dim + POSE_DIM
    print(f"  embed_dim = {cond.embed_dim} (shape={shape_dim} + pose={POSE_DIM})")

    # Verify the shape feature indexing
    expected_shape_feats = shape_features[object_ids]  # (num_slots, shape_dim)
    assert expected_shape_feats.shape == (num_slots, shape_dim)
    for slot in range(num_slots):
        obj_id = object_ids[slot]
        assert np.allclose(expected_shape_feats[slot], shape_features[obj_id])

    print(f"  Shape feature indexing verified for {num_slots} slots → {num_objects} objects")


# ---------------------------------------------------------------------------
# 9. _quat_to_rot6d correctness
# ---------------------------------------------------------------------------

def test_quat_to_rot6d():
    """_quat_to_rot6d should produce correct 6D rotation from quaternion."""
    from jaxrl.mesh_conditioner import _quat_to_rot6d

    # Identity quaternion (w,x,y,z) = (1,0,0,0) → identity rotation
    q_identity = np.array([[1, 0, 0, 0]], dtype=np.float32)
    rot6d = _quat_to_rot6d(q_identity)
    assert rot6d.shape == (1, 6)
    expected = np.array([[1, 0, 0, 0, 1, 0]], dtype=np.float32)
    assert np.allclose(rot6d, expected, atol=1e-6), f"Identity: got {rot6d}"

    # 90° rotation around z: (w,x,y,z) = (cos(45°), 0, 0, sin(45°))
    c = np.cos(np.pi / 4)
    s = np.sin(np.pi / 4)
    q_z90 = np.array([[c, 0, 0, s]], dtype=np.float32)
    rot6d_z90 = _quat_to_rot6d(q_z90)
    expected_z90 = np.array([[0, 1, 0, -1, 0, 0]], dtype=np.float32)
    assert np.allclose(rot6d_z90, expected_z90, atol=1e-5), f"z90: got {rot6d_z90}"

    # Batch test
    q_batch = np.tile(q_identity, (5, 1))
    rot6d_batch = _quat_to_rot6d(q_batch)
    assert rot6d_batch.shape == (5, 6)

    print(f"  Identity and z-90° rotation verified, batch shape: {rot6d_batch.shape}")


# ---------------------------------------------------------------------------
# 10. BRC mesh_shape + none mode equivalence (no features = none behavior)
# ---------------------------------------------------------------------------

def test_none_mode_unchanged():
    """Passing task_ids=None to sample_actions in none mode should still work."""
    from jaxrl.agent.brc_learner import BRC

    agent = BRC(
        seed=0,
        observations=np.zeros((1, 42), dtype=np.float64),
        actions=np.zeros((1, 20), dtype=np.float64),
        num_tasks=3,
        conditioning_mode="none",
    )

    observations = np.random.randn(3, 42).astype(np.float64)
    actions = agent.sample_actions(observations, temperature=1.0)
    assert actions.shape == (3, 20)

    # With explicit task_ids
    slot_ids = jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32)
    obs_5 = np.random.randn(5, 42).astype(np.float64)
    actions_5 = agent.sample_actions(obs_5, temperature=1.0, task_ids=slot_ids)
    assert actions_5.shape == (5, 20)

    print(f"  none mode: default={actions.shape}, explicit={actions_5.shape}")


# ---------------------------------------------------------------------------
# 11. Online conditioner + sharded BRC update (mesh_pose-like flow)
# ---------------------------------------------------------------------------

def test_online_conditioner_sharded_update():
    """Simulated mesh_pose flow: obs+embedding in replay, BRC update works."""
    from jaxrl.agent.brc_learner import BRC
    from jaxrl.replay_buffer import ObjectAwareReplayBuffer
    from jaxrl.normalizer import ObjectAwareRewardNormalizer

    num_objects = 3
    slots_per_object = 4
    num_slots = num_objects * slots_per_object
    obs_dim = 42
    act_dim = 20
    embed_dim = 17  # shape_dim(8) + pose_dim(9)
    aug_dim = obs_dim + embed_dim

    object_ids = np.repeat(np.arange(num_objects, dtype=np.int32), slots_per_object)

    obs_sample = np.zeros((1, aug_dim), dtype=np.float64)
    act_sample = np.zeros((1, act_dim), dtype=np.float64)

    agent = BRC(
        seed=0,
        observations=obs_sample,
        actions=act_sample,
        num_tasks=num_objects,
        conditioning_mode="none",
    )

    class FakeObsSpace:
        def __init__(self):
            self.shape = (num_slots, aug_dim)
            self.dtype = np.float64

    replay = ObjectAwareReplayBuffer(
        FakeObsSpace(), act_dim, 200, num_objects, object_ids,
    )
    normalizer = ObjectAwareRewardNormalizer(
        num_objects, num_slots, object_ids,
        target_entropy=agent.target_entropy, discount=agent.discount,
    )

    slot_task_ids = jnp.array(object_ids, dtype=jnp.int32)

    for step in range(25):
        aug_obs = np.random.randn(num_slots, aug_dim).astype(np.float64)
        if step < 10:
            actions = np.random.uniform(-1, 1, (num_slots, act_dim)).astype(np.float64)
        else:
            actions = agent.sample_actions(aug_obs, temperature=1.0, task_ids=slot_task_ids)
            assert actions.shape == (num_slots, act_dim)

        next_aug_obs = np.random.randn(num_slots, aug_dim).astype(np.float64)
        rewards = np.random.randn(num_slots).astype(np.float64)
        terms = np.zeros(num_slots, dtype=np.float64)
        truns = np.zeros(num_slots, dtype=np.float64)
        normalizer.update(rewards, terms, truns)
        masks = np.ones(num_slots, dtype=np.float64)
        replay.insert(aug_obs, actions, rewards, masks, next_aug_obs)

        if step >= 10:
            batches = replay.sample(32, 2)
            batches = normalizer.normalize(batches, agent.get_temperature())
            info = agent.update(batches, 2, step)

    print(f"  mesh_pose-like online flow: 25 steps completed, info keys: {list(info.keys())}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_test("1. sample_actions with slot-level task_ids", test_sample_actions_slot_task_ids)
    run_test("2. sample_actions default task_ids unchanged", test_sample_actions_default_task_ids)
    run_test("3. _augment_obs slot-level indexing", test_augment_obs_slot_indexing)
    run_test("4. _augment_batch with replay task_ids", test_augment_batch_replay_task_ids)
    run_test("5. Full sample + update cycle", test_full_sample_update_cycle)
    run_test("6. 85×64 shape check", test_85_object_shapes)
    run_test("7. get_infos with ObjectAwareReplayBuffer", test_get_infos_mesh_shape_sharded)
    run_test("8. No categorical in mesh_shape", test_no_categorical_with_mesh_shape)
    run_test("9. ShardedMeshPoseConditioner shapes", test_sharded_mesh_pose_conditioner_shapes)
    run_test("10. _quat_to_rot6d correctness", test_quat_to_rot6d)
    run_test("11. none mode unchanged with task_ids", test_none_mode_unchanged)
    run_test("12. Online conditioner sharded update flow", test_online_conditioner_sharded_update)

    print(f"\n{'='*60}")
    print(f"RESULTS: {TESTS_PASSED} passed, {TESTS_FAILED} failed out of {TESTS_PASSED + TESTS_FAILED}")
    print(f"{'='*60}")
    sys.exit(1 if TESTS_FAILED > 0 else 0)
