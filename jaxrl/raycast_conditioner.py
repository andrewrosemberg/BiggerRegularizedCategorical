"""Wrist-mounted raycast pointcloud extraction for geometry-conditioned BRC.

Generates a deterministic pinhole-camera ray grid from the palm-mounted
``pointnet_camera_site`` and uses ``mujoco.mj_ray`` to produce a partial
pointcloud.  The pointcloud is returned in both world frame and palm frame.

Frame conventions
-----------------
- MuJoCo camera convention: camera looks along **-z** of the site frame,
  with **+x** right and **+y** up.
- Ray grid pixel (u, v): u increases rightward, v increases downward.
  Pixel centers are at (u + 0.5, v + 0.5) for u in [0, W) and v in [0, H).
- Palm frame: obtained from ``data.xpos`` / ``data.xmat`` of the
  ``robot0:palm`` body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

SITE_NAME = "pointnet_camera_site"
CAMERA_NAME = "pointnet_camera"
PALM_BODY_NAME = "robot0:palm"

DEFAULT_GRID_H = 32
DEFAULT_GRID_W = 32
DEFAULT_FOVY_DEG = 40.0
DEFAULT_MAX_DIST = 0.34


@dataclass
class RaycastConfig:
    grid_h: int = DEFAULT_GRID_H
    grid_w: int = DEFAULT_GRID_W
    fovy_deg: float = DEFAULT_FOVY_DEG
    max_dist: float = DEFAULT_MAX_DIST
    site_name: str = SITE_NAME
    palm_body_name: str = PALM_BODY_NAME
    exclude_parent_body: bool = True
    flg_static: int = 0
    geomgroup: tuple[int, ...] | None = (0, 1, 2)


def generate_ray_grid(
    height: int = DEFAULT_GRID_H,
    width: int = DEFAULT_GRID_W,
    fovy_deg: float = DEFAULT_FOVY_DEG,
) -> np.ndarray:
    """Return unit-length ray directions in **camera frame** for a pinhole grid.

    Parameters
    ----------
    height, width : int
        Grid resolution.
    fovy_deg : float
        Vertical field of view in degrees.

    Returns
    -------
    dirs : ndarray, shape (height * width, 3)
        Unit ray directions in MuJoCo camera frame (-z forward, +x right,
        +y up).  Ordered row-major (v changes slowest).
    """
    fovy_rad = math.radians(fovy_deg)
    f = 0.5 * height / math.tan(0.5 * fovy_rad)

    cx = 0.5 * width
    cy = 0.5 * height

    dirs = np.empty((height * width, 3), dtype=np.float64)
    idx = 0
    for v in range(height):
        for u in range(width):
            x = (u + 0.5 - cx) / f
            y = -(v + 0.5 - cy) / f
            z = -1.0
            norm = math.sqrt(x * x + y * y + z * z)
            dirs[idx, 0] = x / norm
            dirs[idx, 1] = y / norm
            dirs[idx, 2] = z / norm
            idx += 1
    return dirs


def _make_geomgroup_array(groups: tuple[int, ...] | None) -> np.ndarray | None:
    if groups is None:
        return None
    arr = np.zeros(6, dtype=np.uint8)
    for g in groups:
        arr[g] = 1
    return arr


def get_raycast_pointcloud(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: RaycastConfig | None = None,
) -> dict[str, Any]:
    """Cast rays from the wrist camera site and return the pointcloud.

    Returns
    -------
    result : dict with keys
        ``points_world``  : (N, 3) float64 - hit positions in world frame
        ``points_palm``   : (N, 3) float64 - hit positions in palm frame
        ``hit_distances`` : (N,)   float64 - ray distances (>0 where valid)
        ``hit_mask``      : (N,)   bool    - True where a ray hit geometry
        ``hit_geom_ids``  : (N,)   int32   - geom id per ray (-1 if miss)
        ``hit_normals``   : (N, 3) float64 - surface normals at hit points (world frame)
        ``ray_origins``   : (N, 3) float64 - ray origins in world frame (all identical)
        ``ray_dirs_world``: (N, 3) float64 - ray directions in world frame
        ``cam_pos_world`` : (3,)   float64 - camera world position
        ``cam_xmat``      : (3, 3) float64 - camera-to-world rotation
        ``palm_pos_world``: (3,)   float64 - palm body world position
        ``palm_xmat``     : (3, 3) float64 - palm-to-world rotation
        ``config``        : RaycastConfig used
        ``grid_shape``    : (int, int) = (H, W)
        ``hit_count``     : int
        ``total_rays``    : int
    """
    if config is None:
        config = RaycastConfig()

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.site_name)
    if site_id < 0:
        raise ValueError(
            f"Site '{config.site_name}' not found in model. "
            "Ensure robot.xml includes the pointnet_camera_site."
        )
    palm_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, config.palm_body_name
    )
    if palm_body_id < 0:
        raise ValueError(f"Body '{config.palm_body_name}' not found in model.")

    cam_pos = data.site_xpos[site_id].copy()
    cam_xmat = data.site_xmat[site_id].reshape(3, 3).copy()

    palm_pos = data.xpos[palm_body_id].copy()
    palm_xmat = data.xmat[palm_body_id].reshape(3, 3).copy()
    palm_xmat_inv = palm_xmat.T

    dirs_cam = generate_ray_grid(config.grid_h, config.grid_w, config.fovy_deg)
    n_rays = dirs_cam.shape[0]

    dirs_world = (cam_xmat @ dirs_cam.T).T

    geomgroup = _make_geomgroup_array(config.geomgroup)

    parent_body_id = int(model.site_bodyid[site_id])
    body_exclude = parent_body_id if config.exclude_parent_body else -1

    hit_distances = np.full(n_rays, -1.0, dtype=np.float64)
    hit_geom_ids = np.full(n_rays, -1, dtype=np.int32)
    hit_normals = np.zeros((n_rays, 3), dtype=np.float64)
    points_world = np.zeros((n_rays, 3), dtype=np.float64)

    geomid_buf = np.array([-1], dtype=np.int32)
    normal_buf = np.zeros(3, dtype=np.float64)

    for i in range(n_rays):
        geomid_buf[0] = -1
        normal_buf[:] = 0.0
        dist = mujoco.mj_ray(
            model,
            data,
            cam_pos,
            dirs_world[i],
            geomgroup,
            config.flg_static,
            body_exclude,
            geomid_buf,
            normal_buf,
        )
        if dist >= 0 and dist <= config.max_dist:
            hit_distances[i] = dist
            hit_geom_ids[i] = geomid_buf[0]
            hit_normals[i] = normal_buf.copy()
            points_world[i] = cam_pos + dist * dirs_world[i]

    hit_mask = hit_distances > 0

    points_palm = np.zeros_like(points_world)
    points_palm[hit_mask] = (
        palm_xmat_inv @ (points_world[hit_mask] - palm_pos).T
    ).T

    ray_origins = np.tile(cam_pos, (n_rays, 1))

    return {
        "points_world": points_world,
        "points_palm": points_palm,
        "hit_distances": hit_distances,
        "hit_mask": hit_mask,
        "hit_geom_ids": hit_geom_ids,
        "hit_normals": hit_normals,
        "ray_origins": ray_origins,
        "ray_dirs_world": dirs_world,
        "cam_pos_world": cam_pos,
        "cam_xmat": cam_xmat,
        "palm_pos_world": palm_pos,
        "palm_xmat": palm_xmat,
        "config": config,
        "grid_shape": (config.grid_h, config.grid_w),
        "hit_count": int(hit_mask.sum()),
        "total_rays": n_rays,
    }
