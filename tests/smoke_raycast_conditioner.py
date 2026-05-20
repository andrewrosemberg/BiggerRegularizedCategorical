"""Smoke tests for the wrist-raycast pointcloud conditioner.

The wrist-raycast path is the deployable geometry sensor path for ShadowHand
experiments.  It casts a deterministic pinhole grid from a camera/site attached
to the palm, returns hit points in world and palm frames, and keeps hit masks so
downstream encoders can distinguish real geometry from missed rays.

These tests check the contracts that matter before any learned encoder is added:
ray-grid geometry, pointcloud tensor shapes, finite valid hits, deterministic
reset behavior, train/test split hygiene for diagnostics, XML camera placement,
and preservation of the existing environment reset/step path.

Run with:
    module load python/3.11.9
    source .venv/bin/activate
    MUJOCO_GL=egl JAX_PLATFORM_NAME=cpu python tests/smoke_raycast_conditioner.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MANIFEST_PATH = os.path.join(REPO_ROOT, "manifests", "shadowhand_split_v1.json")

passed = 0
failed = 0
test_num = 0


def run_test(name: str, fn):
    global passed, failed, test_num
    test_num += 1
    label = f"TEST: {test_num}. {name}"
    try:
        fn()
        print(f"{label:<60s} PASS")
        passed += 1
    except Exception as e:
        print(f"{label:<60s} FAIL")
        traceback.print_exc()
        failed += 1


# Helpers

def _make_env(obj_name: str, seed: int = 42):
    """Create a ShadowHand object-rotation environment and advance kinematics."""
    import dex_envs  # noqa: F401
    import gymnasium as gym
    import mujoco

    env = gym.make(f"{obj_name}-rotate-v1", reward_type="sparse")
    env.reset(seed=seed)
    uw = env.unwrapped
    mujoco.mj_forward(uw.model, uw.data)
    return env


# Tests

def test_01_imports():
    """The public raycast-conditioner API should import without side effects."""
    from jaxrl.raycast_conditioner import (  # noqa: F401
        RaycastConfig,
        generate_ray_grid,
        get_raycast_pointcloud,
    )


def test_02_ray_grid_shape():
    """The default 16x16 pinhole grid should produce 256 camera-frame rays."""
    from jaxrl.raycast_conditioner import generate_ray_grid

    dirs = generate_ray_grid(16, 16, 60.0)
    assert dirs.shape == (256, 3), f"Expected (256, 3), got {dirs.shape}"


def test_03_ray_grid_unit_vectors():
    """Ray directions should be normalized before passing them to MuJoCo."""
    from jaxrl.raycast_conditioner import generate_ray_grid

    dirs = generate_ray_grid(16, 16, 60.0)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10), f"Norms not unit: {norms.min()}, {norms.max()}"


def test_04_ray_grid_center_points_neg_z():
    """MuJoCo cameras look along the negative camera z axis."""
    from jaxrl.raycast_conditioner import generate_ray_grid

    dirs = generate_ray_grid(16, 16, 60.0)
    center = dirs[16 * 8 + 8]
    assert center[2] < -0.9, f"Center ray z={center[2]}, expected < -0.9 (MuJoCo camera looks along -z)"


def test_05_ray_grid_custom_shape():
    """Non-default grid resolutions should preserve row-major ray layout."""
    from jaxrl.raycast_conditioner import generate_ray_grid

    for h, w in [(8, 8), (32, 32), (4, 16)]:
        dirs = generate_ray_grid(h, w, 60.0)
        assert dirs.shape == (h * w, 3), f"Expected ({h * w}, 3), got {dirs.shape}"


def test_06_pointcloud_output_shape():
    """A raycast result should expose all tensors needed by downstream encoders."""
    from jaxrl.raycast_conditioner import get_raycast_pointcloud

    env = _make_env("orange")
    uw = env.unwrapped
    result = get_raycast_pointcloud(uw.model, uw.data)
    n = 16 * 16
    assert result["points_world"].shape == (n, 3)
    assert result["points_palm"].shape == (n, 3)
    assert result["hit_distances"].shape == (n,)
    assert result["hit_mask"].shape == (n,)
    assert result["hit_geom_ids"].shape == (n,)
    assert result["hit_normals"].shape == (n, 3)
    assert result["ray_origins"].shape == (n, 3)
    assert result["ray_dirs_world"].shape == (n, 3)
    assert result["grid_shape"] == (16, 16)
    assert result["total_rays"] == n
    env.close()


def test_07_hit_mask_matches_distances():
    """A valid hit is represented exactly by a positive hit distance."""
    from jaxrl.raycast_conditioner import get_raycast_pointcloud

    env = _make_env("orange")
    uw = env.unwrapped
    result = get_raycast_pointcloud(uw.model, uw.data)
    mask = result["hit_mask"]
    dists = result["hit_distances"]
    assert np.array_equal(mask, dists > 0), "hit_mask should be (distances > 0)"
    env.close()


def test_08_finite_valid_points():
    """Valid hits should be finite, positive-distance, and within max range."""
    from jaxrl.raycast_conditioner import get_raycast_pointcloud

    env = _make_env("orange")
    uw = env.unwrapped
    result = get_raycast_pointcloud(uw.model, uw.data)
    mask = result["hit_mask"]
    assert mask.sum() > 0, "No hits for orange - camera may be misaligned"
    pw = result["points_world"][mask]
    pp = result["points_palm"][mask]
    dists = result["hit_distances"][mask]
    normals = result["hit_normals"][mask]
    assert np.all(np.isfinite(pw)), "Non-finite world points"
    assert np.all(np.isfinite(pp)), "Non-finite palm points"
    assert np.all(np.isfinite(dists)), "Non-finite distances"
    assert np.all(np.isfinite(normals)), "Non-finite normals"
    assert np.all(dists > 0), "Valid distances should be positive"
    assert np.all(dists <= 0.34), "Distances should not exceed max_dist"
    env.close()


def test_09_deterministic_reset():
    """Resetting the same object with the same seed should reproduce the raycast."""
    from jaxrl.raycast_conditioner import get_raycast_pointcloud
    import mujoco

    env = _make_env("orange", seed=123)
    uw = env.unwrapped
    r1 = get_raycast_pointcloud(uw.model, uw.data)
    env.close()

    env = _make_env("orange", seed=123)
    uw = env.unwrapped
    r2 = get_raycast_pointcloud(uw.model, uw.data)
    env.close()

    assert np.array_equal(r1["hit_mask"], r2["hit_mask"]), "Hit masks differ across identical resets"
    assert np.allclose(r1["points_world"], r2["points_world"]), "World points differ"
    assert np.allclose(r1["hit_distances"], r2["hit_distances"]), "Distances differ"


def test_10_three_train_objects():
    """The raycast path should work across multiple randomly sampled train objects."""
    from jaxrl.raycast_conditioner import get_raycast_pointcloud

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    train_objects = manifest["train"]

    rng = np.random.RandomState(2026)
    sampled = rng.choice(train_objects, size=3, replace=False).tolist()

    for obj_name in sampled:
        env = _make_env(obj_name, seed=99)
        uw = env.unwrapped
        result = get_raycast_pointcloud(uw.model, uw.data)
        n = result["total_rays"]
        assert result["points_world"].shape == (n, 3), f"{obj_name}: wrong shape"
        assert result["hit_mask"].shape == (n,), f"{obj_name}: wrong mask shape"
        mask = result["hit_mask"]
        if mask.sum() > 0:
            assert np.all(np.isfinite(result["points_world"][mask])), f"{obj_name}: non-finite"
        env.close()


def test_11_no_heldout_in_train_diagnostics():
    """Diagnostic object sampling should respect the canonical train/test split."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    train_set = set(manifest["train"])
    test_set = set(manifest["test"])
    assert len(train_set & test_set) == 0, "Train/test overlap"
    assert len(train_set) == 85, f"Expected 85 train, got {len(train_set)}"
    assert len(test_set) == 29, f"Expected 29 test, got {len(test_set)}"


def test_12_xml_site_present():
    """The ShadowHand model should contain the palm-mounted raycast site and camera."""
    import mujoco

    env = _make_env("orange")
    uw = env.unwrapped
    site_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_SITE, "pointnet_camera_site")
    cam_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_CAMERA, "pointnet_camera")
    assert site_id >= 0, "pointnet_camera_site not found"
    assert cam_id >= 0, "pointnet_camera not found"
    palm_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:palm")
    assert uw.model.site_bodyid[site_id] == palm_id, "Site not on palm body"
    assert uw.model.cam_fovy[cam_id] == 60.0, f"Camera fovy={uw.model.cam_fovy[cam_id]}, expected 60"
    env.close()


def test_13_palm_frame_consistency():
    """Verify palm-frame transform is self-consistent: converting back to world
    frame should match the original world points."""
    from jaxrl.raycast_conditioner import get_raycast_pointcloud

    env = _make_env("orange")
    uw = env.unwrapped
    result = get_raycast_pointcloud(uw.model, uw.data)
    mask = result["hit_mask"]
    if mask.sum() == 0:
        env.close()
        return
    pw = result["points_world"][mask]
    pp = result["points_palm"][mask]
    palm_pos = result["palm_pos_world"]
    palm_xmat = result["palm_xmat"]
    reconstructed = (palm_xmat @ pp.T).T + palm_pos
    assert np.allclose(pw, reconstructed, atol=1e-10), (
        f"Palm frame round-trip error: max diff = {np.abs(pw - reconstructed).max()}"
    )
    env.close()


def test_14_env_step_still_works():
    """Verify that adding the site/camera to robot.xml does not break env step."""
    from jaxrl.envs import make_env

    env = make_env("orange", seed=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (68,), f"Obs shape {obs.shape}, expected (68,)"
    action = env.action_space.sample()
    obs2, reward, term, trunc, info = env.step(action)
    assert obs2.shape == (68,), f"Step obs shape {obs2.shape}, expected (68,)"
    env.close()


def test_15_different_seeds_different_hits():
    """Different seeds should generally produce different hit patterns."""
    from jaxrl.raycast_conditioner import get_raycast_pointcloud

    env1 = _make_env("orange", seed=1)
    r1 = get_raycast_pointcloud(env1.unwrapped.model, env1.unwrapped.data)
    env1.close()

    env2 = _make_env("orange", seed=999)
    r2 = get_raycast_pointcloud(env2.unwrapped.model, env2.unwrapped.data)
    env2.close()

    same = np.array_equal(r1["hit_distances"], r2["hit_distances"])
    assert not same, "Different seeds produced identical raycasts - suspicious"


# Main

if __name__ == "__main__":
    tests = [
        ("Raycast conditioner imports", test_01_imports),
        ("Ray grid shape (16x16 = 256 rays)", test_02_ray_grid_shape),
        ("Ray grid vectors are unit length", test_03_ray_grid_unit_vectors),
        ("Ray grid center points along -z", test_04_ray_grid_center_points_neg_z),
        ("Ray grid custom shapes", test_05_ray_grid_custom_shape),
        ("Pointcloud output shapes", test_06_pointcloud_output_shape),
        ("Hit mask matches distances", test_07_hit_mask_matches_distances),
        ("Valid points are finite and bounded", test_08_finite_valid_points),
        ("Deterministic reset gives deterministic raycast", test_09_deterministic_reset),
        ("Three random train-split objects", test_10_three_train_objects),
        ("No held-out data in train diagnostics", test_11_no_heldout_in_train_diagnostics),
        ("XML site and camera present on palm", test_12_xml_site_present),
        ("Palm frame round-trip consistency", test_13_palm_frame_consistency),
        ("Env reset/step still works after XML change", test_14_env_step_still_works),
        ("Different seeds produce different raycasts", test_15_different_seeds_different_hits),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\nRESULTS: {passed} passed, {failed} failed out of {test_num}")
    sys.exit(1 if failed > 0 else 0)
