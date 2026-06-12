"""Visual sanity check for the wrist-raycast pointcloud path.

Produces one figure per object with three panels:
  1. Bird's-eye rendered scene.
  2. Wrist-camera RGB.
  3. Raycast pointcloud from the same bird's-eye viewpoint.

The figures are qualitative diagnostics.  They make it easy to catch wrong
camera placement, flipped coordinate frames, empty raycasts, and accidental hits
on unrelated geometry before the pointcloud path is used by a learned encoder.

Run:
    module load python/3.11.9
    source .venv/bin/activate
    MUJOCO_GL=egl JAX_PLATFORM_NAME=cpu python scripts/raycast_visual_check.py
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from jaxrl.raycast_conditioner import get_raycast_pointcloud

MANIFEST_PATH = os.path.join(REPO_ROOT, "manifests", "shadowhand_split_v1.json")
OUTPUT_DIR = os.path.join(REPO_ROOT, "agent_reports", "raycast_visuals")
RANDOM_SEED = 20260520
RENDER_SIZE = 480
BIRDSEYE_LOOKAT = (1.0, 0.93, 0.16)
BIRDSEYE_DIST = 0.55
BIRDSEYE_AZIM = 160.0
BIRDSEYE_ELEV = -30.0


def _make_env(obj_name: str, seed: int):
    """Create a deterministic ShadowHand object-rotation scene for rendering."""
    import dex_envs  # noqa: F401
    import gymnasium as gym

    env = gym.make(f"{obj_name}-rotate-v1", reward_type="sparse")
    env.reset(seed=seed)
    uw = env.unwrapped
    mujoco.mj_forward(uw.model, uw.data)
    return env


def _render_scene(model, data, width=RENDER_SIZE, height=RENDER_SIZE) -> np.ndarray:
    """Render the full hand-object scene from the fixed bird's-eye viewpoint."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = BIRDSEYE_LOOKAT
    cam.distance = BIRDSEYE_DIST
    cam.azimuth = BIRDSEYE_AZIM
    cam.elevation = BIRDSEYE_ELEV
    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, cam)
    img = renderer.render()
    renderer.close()
    return img


def _render_wrist_camera(model, data, width=RENDER_SIZE, height=RENDER_SIZE) -> np.ndarray:
    """Render the RGB image from the palm-mounted MuJoCo camera."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "pointnet_camera")
    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=cam_id)
    img = renderer.render()
    renderer.close()
    return img


def _set_axes_equal(ax) -> None:
    """Use equal 3D axis scales so the pointcloud shape is not distorted."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    if radius <= 0:
        radius = 0.1
    for setter, c in zip(
        [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d], centers
    ):
        setter([c - radius, c + radius])


def make_figure(obj_name: str, seed: int, output_dir: str) -> dict:
    """Render the scene, wrist image, and corresponding raycast pointcloud."""
    env = _make_env(obj_name, seed)
    uw = env.unwrapped
    model, data = uw.model, uw.data

    scene_img = _render_scene(model, data)
    wrist_img = _render_wrist_camera(model, data)

    result = get_raycast_pointcloud(model, data)
    mask = result["hit_mask"]
    hit_count = result["hit_count"]
    total_rays = result["total_rays"]

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(scene_img)
    ax1.set_title(f"{obj_name} - Bird's-eye scene")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(wrist_img)
    ax2.set_title("Wrist-camera RGB (pointnet_camera)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    if hit_count > 0:
        pw = result["points_world"][mask]
        cam_pos = result["cam_pos_world"]
        dists = np.linalg.norm(pw - cam_pos, axis=1)
        sc = ax3.scatter(
            pw[:, 0], pw[:, 1], pw[:, 2],
            c=dists, cmap="viridis", s=30, alpha=0.9, edgecolors="k", linewidths=0.3,
        )
        cb = fig.colorbar(sc, ax=ax3, shrink=0.6, pad=0.1)
        cb.set_label("Dist to wrist cam (m)")
        ax3.scatter(*cam_pos, c="red", marker="^", s=80, label="Camera")
        ax3.legend(loc="upper right", fontsize=7)

    ax3.set_title(f"Raycast pointcloud ({hit_count}/{total_rays} hits)\nWorld frame")
    ax3.set_xlabel("world x (m)")
    ax3.set_ylabel("world y (m)")
    ax3.set_zlabel("world z (m)")
    ax3.view_init(elev=-BIRDSEYE_ELEV, azim=180.0 - BIRDSEYE_AZIM)
    if hit_count > 0:
        _set_axes_equal(ax3)

    out_path = os.path.join(output_dir, f"{obj_name}_raycast_check.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    env.close()

    return {
        "object": obj_name,
        "seed": seed,
        "hit_count": hit_count,
        "total_rays": total_rays,
        "hit_fraction": hit_count / max(total_rays, 1),
        "figure_path": out_path,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    train_objects = manifest["train"]

    rng = np.random.RandomState(RANDOM_SEED)
    sampled = rng.choice(train_objects, size=3, replace=False).tolist()

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Sampled train objects: {sampled}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    results = []
    for obj_name in sampled:
        print(f"Generating figure for {obj_name}...")
        info = make_figure(obj_name, seed=RANDOM_SEED, output_dir=OUTPUT_DIR)
        results.append(info)
        print(f"  Hits: {info['hit_count']}/{info['total_rays']} "
              f"({info['hit_fraction']:.1%})")
        print(f"  Saved: {info['figure_path']}")

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"seed": RANDOM_SEED, "objects": sampled, "results": results}, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
