"""Sweep raycast sensor settings (grid size, FOV, max distance) across objects.

Evaluates hit statistics for each (grid_size, fov, max_dist, object) combination.
Outputs a JSON results file and prints a summary table.

Run:
    module load python/3.11.9
    source .venv/bin/activate
    MUJOCO_GL=egl JAX_PLATFORM_NAME=cpu python scripts/raycast_sensor_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from jaxrl.raycast_conditioner import RaycastConfig, get_raycast_pointcloud

MANIFEST_PATH = os.path.join(REPO_ROOT, "manifests", "shadowhand_split_v1.json")
OUTPUT_DIR = os.path.join(REPO_ROOT, "agent_reports", "raycast_sensor_sweep")

GRID_SIZES = [16, 24, 32, 48, 64]
FOV_VALUES = [60.0, 50.0, 45.0, 40.0, 35.0, 30.0]
MAX_DIST_VALUES = [0.34]

EVAL_OBJECTS = [
    "mug",
    "cracker_box",
    "sugar_box",
    "flat_screwdriver",
    "knife",
    "hammer",
    "power_drill",
    "orange",
    "wine_glass",
    "banana",
    "light_bulb",
    "scissors",
    "chain",
]

NUM_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2026]


def _make_env(obj_name: str, seed: int):
    import dex_envs  # noqa: F401
    import gymnasium as gym

    env = gym.make(f"{obj_name}-rotate-v1", reward_type="sparse")
    env.reset(seed=seed)
    uw = env.unwrapped
    mujoco.mj_forward(uw.model, uw.data)
    return env


def _find_object_geom_ids(model):
    """Find geom IDs belonging to the 'object' body."""
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id < 0:
        return set()
    geom_ids = set()
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == obj_body_id:
            geom_ids.add(g)
    return geom_ids


def evaluate_setting(model, data, grid_size, fov, max_dist, object_geom_ids):
    """Run raycast with given settings and return hit statistics."""
    config = RaycastConfig(
        grid_h=grid_size,
        grid_w=grid_size,
        fovy_deg=fov,
        max_dist=max_dist,
    )

    t0 = time.perf_counter()
    result = get_raycast_pointcloud(model, data, config=config)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    total_rays = result["total_rays"]
    hit_mask = result["hit_mask"]
    hit_count = int(hit_mask.sum())
    hit_geom_ids = result["hit_geom_ids"]

    obj_hit_mask = np.array(
        [hit_mask[i] and int(hit_geom_ids[i]) in object_geom_ids
         for i in range(total_rays)],
        dtype=bool,
    )
    obj_hit_count = int(obj_hit_mask.sum())
    hand_hit_count = hit_count - obj_hit_count

    return {
        "total_rays": total_rays,
        "hit_count": hit_count,
        "hit_fraction": hit_count / max(total_rays, 1),
        "obj_hit_count": obj_hit_count,
        "obj_hit_frac_of_rays": obj_hit_count / max(total_rays, 1),
        "obj_hit_frac_of_hits": obj_hit_count / max(hit_count, 1),
        "hand_hit_count": hand_hit_count,
        "runtime_ms": elapsed_ms,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    train_set = set(manifest["train"])
    for obj in EVAL_OBJECTS:
        assert obj in train_set, f"{obj} not in train split"

    all_results = []
    total_combos = len(GRID_SIZES) * len(FOV_VALUES) * len(MAX_DIST_VALUES) * len(EVAL_OBJECTS) * NUM_SEEDS
    combo_idx = 0

    for obj_name in EVAL_OBJECTS:
        for seed in SEEDS:
            env = _make_env(obj_name, seed)
            uw = env.unwrapped
            model, data = uw.model, uw.data
            object_geom_ids = _find_object_geom_ids(model)

            for grid_size in GRID_SIZES:
                for fov in FOV_VALUES:
                    for max_dist in MAX_DIST_VALUES:
                        combo_idx += 1
                        stats = evaluate_setting(
                            model, data, grid_size, fov, max_dist, object_geom_ids
                        )
                        row = {
                            "object": obj_name,
                            "seed": seed,
                            "grid_size": grid_size,
                            "fov_deg": fov,
                            "max_dist": max_dist,
                            **stats,
                        }
                        all_results.append(row)

                        if combo_idx % 100 == 0 or combo_idx == total_combos:
                            print(f"  [{combo_idx}/{total_combos}] "
                                  f"{obj_name} seed={seed} grid={grid_size} "
                                  f"fov={fov} => hits={stats['hit_count']}/{stats['total_rays']} "
                                  f"obj={stats['obj_hit_count']} "
                                  f"t={stats['runtime_ms']:.1f}ms")

            env.close()

    results_path = os.path.join(OUTPUT_DIR, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "eval_objects": EVAL_OBJECTS,
            "grid_sizes": GRID_SIZES,
            "fov_values": FOV_VALUES,
            "max_dist_values": MAX_DIST_VALUES,
            "seeds": SEEDS,
            "num_results": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print("\n=== AGGREGATE TABLE (mean over seeds) ===")
    print(f"{'Object':<20s} {'Grid':>5s} {'FOV':>5s} {'MaxD':>5s} "
          f"{'Rays':>6s} {'Hits':>6s} {'HitF':>6s} "
          f"{'ObjH':>6s} {'ObjF':>6s} {'OHoH':>6s} "
          f"{'HandH':>6s} {'ms':>7s}")
    print("-" * 110)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        key = (r["object"], r["grid_size"], r["fov_deg"], r["max_dist"])
        agg[key].append(r)

    for key in sorted(agg.keys()):
        rows = agg[key]
        obj, grid, fov, md = key
        n = len(rows)
        mean_hits = sum(r["hit_count"] for r in rows) / n
        mean_hf = sum(r["hit_fraction"] for r in rows) / n
        mean_oh = sum(r["obj_hit_count"] for r in rows) / n
        mean_of = sum(r["obj_hit_frac_of_rays"] for r in rows) / n
        mean_ohoh = sum(r["obj_hit_frac_of_hits"] for r in rows) / n
        mean_hh = sum(r["hand_hit_count"] for r in rows) / n
        mean_ms = sum(r["runtime_ms"] for r in rows) / n
        total_rays = rows[0]["total_rays"]
        print(f"{obj:<20s} {grid:>5d} {fov:>5.0f} {md:>5.2f} "
              f"{total_rays:>6d} {mean_hits:>6.1f} {mean_hf:>6.3f} "
              f"{mean_oh:>6.1f} {mean_of:>6.3f} {mean_ohoh:>6.3f} "
              f"{mean_hh:>6.1f} {mean_ms:>7.1f}")

    summary_path = os.path.join(OUTPUT_DIR, "sweep_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== SETTING-LEVEL AGGREGATES (mean over all objects and seeds) ===\n\n")
        f.write(f"{'Grid':>5s} {'FOV':>5s} {'Rays':>6s} "
                f"{'MeanHits':>9s} {'MeanHitF':>9s} "
                f"{'MeanObjH':>9s} {'MeanObjF':>9s} "
                f"{'MeanOHoH':>9s} {'MeanHandH':>10s} "
                f"{'MeanMs':>8s}\n")
        f.write("-" * 100 + "\n")

        setting_agg = defaultdict(list)
        for r in all_results:
            skey = (r["grid_size"], r["fov_deg"])
            setting_agg[skey].append(r)

        for skey in sorted(setting_agg.keys()):
            rows = setting_agg[skey]
            grid, fov = skey
            n = len(rows)
            total_rays = rows[0]["total_rays"]
            mean_hits = sum(r["hit_count"] for r in rows) / n
            mean_hf = sum(r["hit_fraction"] for r in rows) / n
            mean_oh = sum(r["obj_hit_count"] for r in rows) / n
            mean_of = sum(r["obj_hit_frac_of_rays"] for r in rows) / n
            mean_ohoh = sum(r["obj_hit_frac_of_hits"] for r in rows) / n
            mean_hh = sum(r["hand_hit_count"] for r in rows) / n
            mean_ms = sum(r["runtime_ms"] for r in rows) / n
            f.write(f"{grid:>5d} {fov:>5.0f} {total_rays:>6d} "
                    f"{mean_hits:>9.1f} {mean_hf:>9.3f} "
                    f"{mean_oh:>9.1f} {mean_of:>9.3f} "
                    f"{mean_ohoh:>9.3f} {mean_hh:>10.1f} "
                    f"{mean_ms:>8.1f}\n")

    print(f"\nSetting-level summary: {summary_path}")


if __name__ == "__main__":
    main()
