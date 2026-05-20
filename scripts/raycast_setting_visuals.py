"""Generate visual comparison figures for candidate raycast sensor settings.

Produces three-panel figures (bird's-eye scene, wrist-camera RGB, raycast
pointcloud) for specified settings and objects, plus an optional fourth
panel with object-vs-hand hit coloring (diagnostic only).

Run:
    module load python/3.11.9
    source .venv/bin/activate
    MUJOCO_GL=egl JAX_PLATFORM_NAME=cpu python scripts/raycast_setting_visuals.py
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

from jaxrl.raycast_conditioner import RaycastConfig, get_raycast_pointcloud

MANIFEST_PATH = os.path.join(REPO_ROOT, "manifests", "shadowhand_split_v1.json")
OUTPUT_DIR = os.path.join(REPO_ROOT, "agent_reports", "raycast_sensor_sweep", "figures")

RENDER_SIZE = 480
BIRDSEYE_LOOKAT = (1.0, 0.93, 0.16)
BIRDSEYE_DIST = 0.55
BIRDSEYE_AZIM = 160.0
BIRDSEYE_ELEV = -30.0

CANDIDATE_SETTINGS = [
    {"label": "baseline_16x16_fov60", "grid": 16, "fov": 60.0},
    {"label": "default_32x32_fov40", "grid": 32, "fov": 40.0},
    {"label": "hq_64x64_fov40", "grid": 64, "fov": 40.0},
]

VISUAL_OBJECTS = ["mug", "hammer", "knife", "cracker_box", "wine_glass"]
SEED = 20260520


def _make_env(obj_name: str, seed: int):
    import dex_envs  # noqa: F401
    import gymnasium as gym

    env = gym.make(f"{obj_name}-rotate-v1", reward_type="sparse")
    env.reset(seed=seed)
    uw = env.unwrapped
    mujoco.mj_forward(uw.model, uw.data)
    return env


def _render_scene(model, data, width=RENDER_SIZE, height=RENDER_SIZE):
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


def _render_wrist_camera(model, data, width=RENDER_SIZE, height=RENDER_SIZE):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "pointnet_camera")
    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=cam_id)
    img = renderer.render()
    renderer.close()
    return img


def _find_object_geom_ids(model):
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id < 0:
        return set()
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == obj_body_id}


def _set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    if radius <= 0:
        radius = 0.1
    for setter, c in zip([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d], centers):
        setter([c - radius, c + radius])


def make_comparison_figure(obj_name: str, seed: int, output_dir: str):
    """Generate a multi-row figure comparing all candidate settings for one object."""
    env = _make_env(obj_name, seed)
    uw = env.unwrapped
    model, data = uw.model, uw.data
    object_geom_ids = _find_object_geom_ids(model)

    scene_img = _render_scene(model, data)
    wrist_img = _render_wrist_camera(model, data)

    n_settings = len(CANDIDATE_SETTINGS)
    fig = plt.figure(figsize=(20, 5 * n_settings), constrained_layout=True)
    fig.suptitle(f"{obj_name} - Raycast Setting Comparison (seed={seed})",
                 fontsize=14, fontweight="bold")

    for row_idx, setting in enumerate(CANDIDATE_SETTINGS):
        config = RaycastConfig(
            grid_h=setting["grid"],
            grid_w=setting["grid"],
            fovy_deg=setting["fov"],
            max_dist=0.34,
        )
        result = get_raycast_pointcloud(model, data, config=config)
        mask = result["hit_mask"]
        hit_count = result["hit_count"]
        total_rays = result["total_rays"]
        geom_ids = result["hit_geom_ids"]

        obj_mask = np.array(
            [mask[i] and int(geom_ids[i]) in object_geom_ids for i in range(total_rays)],
            dtype=bool,
        )
        hand_mask = mask & ~obj_mask
        obj_hits = int(obj_mask.sum())
        hand_hits = int(hand_mask.sum())

        ax1 = fig.add_subplot(n_settings, 4, row_idx * 4 + 1)
        ax1.imshow(scene_img)
        ax1.set_title(f"Bird's-eye scene")
        ax1.axis("off")
        if row_idx == 0:
            ax1.text(0.02, 0.98, f"seed={seed}", transform=ax1.transAxes,
                     fontsize=8, va="top", color="white",
                     bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

        ax2 = fig.add_subplot(n_settings, 4, row_idx * 4 + 2)
        ax2.imshow(wrist_img)
        ax2.set_title("Wrist-camera RGB")
        ax2.axis("off")

        ax3 = fig.add_subplot(n_settings, 4, row_idx * 4 + 3, projection="3d")
        if hit_count > 0:
            pw = result["points_world"][mask]
            cam_pos = result["cam_pos_world"]
            dists = np.linalg.norm(pw - cam_pos, axis=1)
            ax3.scatter(pw[:, 0], pw[:, 1], pw[:, 2],
                        c=dists, cmap="viridis", s=max(1, 30 - setting["grid"] // 4),
                        alpha=0.8, edgecolors="none")
            ax3.scatter(*cam_pos, c="red", marker="^", s=60, zorder=10)
        ax3.set_title(f"{setting['label']}\n{hit_count}/{total_rays} hits "
                      f"({hit_count/total_rays:.1%})")
        ax3.set_xlabel("x (m)", fontsize=7)
        ax3.set_ylabel("y (m)", fontsize=7)
        ax3.set_zlabel("z (m)", fontsize=7)
        ax3.view_init(elev=-BIRDSEYE_ELEV, azim=180.0 - BIRDSEYE_AZIM)
        ax3.tick_params(labelsize=6)
        if hit_count > 0:
            _set_axes_equal(ax3)

        ax4 = fig.add_subplot(n_settings, 4, row_idx * 4 + 4, projection="3d")
        if hit_count > 0:
            if obj_hits > 0:
                pw_obj = result["points_world"][obj_mask]
                ax4.scatter(pw_obj[:, 0], pw_obj[:, 1], pw_obj[:, 2],
                            c="tab:blue", s=max(1, 30 - setting["grid"] // 4),
                            alpha=0.8, label=f"Object ({obj_hits})",
                            edgecolors="none")
            if hand_hits > 0:
                pw_hand = result["points_world"][hand_mask]
                ax4.scatter(pw_hand[:, 0], pw_hand[:, 1], pw_hand[:, 2],
                            c="tab:orange", s=max(1, 30 - setting["grid"] // 4),
                            alpha=0.5, label=f"Hand ({hand_hits})",
                            edgecolors="none")
            ax4.scatter(*result["cam_pos_world"], c="red", marker="^", s=60, zorder=10)
            ax4.legend(fontsize=7, loc="upper right")
        ax4.set_title(f"DIAGNOSTIC: obj vs hand\nObj={obj_hits} Hand={hand_hits}")
        ax4.set_xlabel("x (m)", fontsize=7)
        ax4.set_ylabel("y (m)", fontsize=7)
        ax4.set_zlabel("z (m)", fontsize=7)
        ax4.view_init(elev=-BIRDSEYE_ELEV, azim=180.0 - BIRDSEYE_AZIM)
        ax4.tick_params(labelsize=6)
        if hit_count > 0:
            _set_axes_equal(ax4)

    out_path = os.path.join(output_dir, f"{obj_name}_setting_comparison.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    env.close()
    print(f"  Saved: {out_path}")
    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    train_set = set(manifest["train"])
    for obj in VISUAL_OBJECTS:
        assert obj in train_set, f"{obj} not in train split"

    print(f"Candidate settings: {[s['label'] for s in CANDIDATE_SETTINGS]}")
    print(f"Visual objects: {VISUAL_OBJECTS}")
    print(f"Seed: {SEED}")
    print(f"Output: {OUTPUT_DIR}\n")

    figure_paths = []
    for obj_name in VISUAL_OBJECTS:
        print(f"Generating comparison for {obj_name}...")
        path = make_comparison_figure(obj_name, SEED, OUTPUT_DIR)
        figure_paths.append(path)

    summary = {
        "seed": SEED,
        "settings": CANDIDATE_SETTINGS,
        "objects": VISUAL_OBJECTS,
        "figure_paths": figure_paths,
    }
    summary_path = os.path.join(OUTPUT_DIR, "visual_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
