"""Online conditioner interface for geometry-conditioned BRC.

The online conditioner computes geometry embeddings from current environment
state at every policy step.  This module provides:

- ``OnlineRaycastConditioner``: extracts wrist raycasts from live MuJoCo
  environments and encodes them with a :class:`MaskAwarePointNet`.
- Helper to build initial (random-parameter) conditioners for smoke testing.

The key contract is ``c_t = E_theta(x_t)`` where ``x_t`` is the current
geometry observation.  Wrist-raycast embeddings must NOT be cached per object;
they must be recomputed from the current pointcloud each step.

Replay integration note
-----------------------
Because wrist-raycast embeddings change every timestep, the training loop must
store them alongside observations in the replay buffer so that replayed
transitions carry the embedding that was active when the transition was
collected.  The recommended approach is to concatenate the embedding to the
observation before inserting into the replay buffer, so BRC sees
``obs_dim + embed_dim`` as its observation space.  This is handled by
the training loop, not by BRC internals.
"""

from __future__ import annotations

import json
import os
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from jaxrl.pointnet import (
    DEFAULT_EMBED_DIM,
    DEFAULT_HIDDEN_DIMS,
    NORMALIZATION_SCALE,
    RAYCAST_POINT_CHANNELS,
    MaskAwarePointNet,
)
from jaxrl.raycast_conditioner import RaycastConfig, get_raycast_pointcloud


class OnlineRaycastConditioner:
    """Extracts wrist raycasts from environments and encodes them online.

    Parameters
    ----------
    encoder_def : MaskAwarePointNet
        Flax module definition (not initialized).
    encoder_params : dict
        Flax parameter dict for the encoder.
    raycast_config : RaycastConfig, optional
        Ray grid and sensor settings.
    normalize_scale : float
        Divide palm-frame xyz by this before encoding.
    """

    def __init__(
        self,
        encoder_def: MaskAwarePointNet,
        encoder_params: Any,
        raycast_config: RaycastConfig | None = None,
        normalize_scale: float = NORMALIZATION_SCALE,
    ):
        self.encoder_def = encoder_def
        self.encoder_params = encoder_params
        self.raycast_config = raycast_config or RaycastConfig()
        self.normalize_scale = normalize_scale
        self.embed_dim = encoder_def.output_dim

    def extract_raycasts(self, envs: list) -> tuple[np.ndarray, np.ndarray]:
        """Extract palm-frame pointclouds and masks from live environments.

        Parameters
        ----------
        envs : list of gym.Env
            Each must expose ``.unwrapped.model`` and ``.unwrapped.data``.

        Returns
        -------
        points : (num_envs, N, 3) float32 -- normalized palm-frame xyz
        masks  : (num_envs, N)    bool     -- True where ray hit geometry
        """
        points_list = []
        masks_list = []
        for env in envs:
            uw = env.unwrapped
            result = get_raycast_pointcloud(uw.model, uw.data, self.raycast_config)
            pts = result["points_palm"].astype(np.float32) / self.normalize_scale
            points_list.append(pts)
            masks_list.append(result["hit_mask"])
        return np.stack(points_list), np.stack(masks_list)

    def encode(
        self, points: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Run the PointNet encoder on pre-extracted pointclouds.

        Parameters
        ----------
        points : (batch, N, 3) float32
        masks  : (batch, N) bool

        Returns
        -------
        embeddings : (batch, embed_dim) float32
        """
        embeddings = self.encoder_def.apply(
            {"params": self.encoder_params},
            jnp.array(points),
            jnp.array(masks),
        )
        return np.asarray(embeddings)

    def extract_and_encode(self, envs: list) -> np.ndarray:
        """One-call convenience: extract raycasts then encode.

        Returns
        -------
        embeddings : (num_envs, embed_dim) float32
        """
        points, masks = self.extract_raycasts(envs)
        return self.encode(points, masks)


    def save_checkpoint(self, path: str, extra_metadata: dict | None = None):
        """Save encoder parameters and metadata to a checkpoint directory.

        Creates ``path/encoder_params.bin`` and ``path/metadata.json``.
        """
        os.makedirs(path, exist_ok=True)

        params_path = os.path.join(path, "encoder_params.bin")
        with open(params_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.encoder_params))

        cfg = self.raycast_config
        meta = {
            "encoder_type": "MaskAwarePointNet",
            "hidden_dims": list(self.encoder_def.hidden_dims),
            "output_dim": self.encoder_def.output_dim,
            "embed_dim": self.embed_dim,
            "normalize_scale": self.normalize_scale,
            "point_channels": RAYCAST_POINT_CHANNELS,
            "n_points": cfg.grid_h * cfg.grid_w,
            "mask_convention": "True where ray hit geometry",
            "coordinate_frame": "palm",
            "raycast_config": {
                "grid_h": cfg.grid_h,
                "grid_w": cfg.grid_w,
                "fovy_deg": cfg.fovy_deg,
                "max_dist": cfg.max_dist,
                "site_name": cfg.site_name,
                "palm_body_name": cfg.palm_body_name,
            },
        }
        if extra_metadata:
            meta.update(extra_metadata)

        meta_path = os.path.join(path, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_checkpoint(cls, path: str) -> "OnlineRaycastConditioner":
        """Restore an OnlineRaycastConditioner from a saved checkpoint."""
        meta_path = os.path.join(path, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)

        hidden_dims = tuple(meta["hidden_dims"])
        output_dim = meta["output_dim"]
        normalize_scale = meta.get("normalize_scale", NORMALIZATION_SCALE)
        n_points = meta["n_points"]

        rc = meta.get("raycast_config", {})
        raycast_config = RaycastConfig(
            grid_h=rc.get("grid_h", 32),
            grid_w=rc.get("grid_w", 32),
            fovy_deg=rc.get("fovy_deg", 40.0),
            max_dist=rc.get("max_dist", 0.34),
            site_name=rc.get("site_name", "pointnet_camera_site"),
            palm_body_name=rc.get("palm_body_name", "robot0:palm"),
        )

        encoder_def = MaskAwarePointNet(
            hidden_dims=hidden_dims, output_dim=output_dim
        )
        dummy_points = jnp.zeros((1, n_points, RAYCAST_POINT_CHANNELS))
        dummy_mask = jnp.ones((1, n_points), dtype=bool)
        rng = jax.random.PRNGKey(0)
        variables = encoder_def.init(rng, dummy_points, dummy_mask)
        template_params = variables["params"]

        params_path = os.path.join(path, "encoder_params.bin")
        with open(params_path, "rb") as f:
            encoder_params = flax.serialization.from_bytes(
                template_params, f.read()
            )

        return cls(
            encoder_def=encoder_def,
            encoder_params=encoder_params,
            raycast_config=raycast_config,
            normalize_scale=normalize_scale,
        )


def build_raycast_conditioner(
    seed: int = 0,
    output_dim: int = DEFAULT_EMBED_DIM,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    raycast_config: RaycastConfig | None = None,
) -> OnlineRaycastConditioner:
    """Create an OnlineRaycastConditioner with randomly initialized encoder.

    This is intended for smoke testing and interface validation.  The encoder
    parameters are random and untrained.
    """
    config = raycast_config or RaycastConfig()
    n_points = config.grid_h * config.grid_w

    encoder_def = MaskAwarePointNet(
        hidden_dims=hidden_dims, output_dim=output_dim
    )

    rng = jax.random.PRNGKey(seed)
    dummy_points = jnp.zeros((1, n_points, RAYCAST_POINT_CHANNELS))
    dummy_mask = jnp.ones((1, n_points), dtype=bool)
    variables = encoder_def.init(rng, dummy_points, dummy_mask)
    encoder_params = variables["params"]

    return OnlineRaycastConditioner(
        encoder_def=encoder_def,
        encoder_params=encoder_params,
        raycast_config=config,
    )


class ShardedRaycastConditioner:
    """Online raycast conditioner for ShardedMjlabParallelEnv.

    Reads GPU-native raycast sensor data from mjlab shards (no CPU mj_ray).
    Encodes per-step pointclouds with a MaskAwarePointNet. The inference
    contract is sensor-only: no object ID, no object pose, no mesh geometry.

    Frame handling:
    - If ``palm_frame=True`` and the palm body is accessible, hit positions
      are transformed to palm frame before encoding.
    - If palm body data is unavailable, falls back to world-frame and
      prints a warning. The ``frame`` attribute records which was used.
    """

    def __init__(
        self,
        encoder_def: MaskAwarePointNet,
        encoder_params,
        normalize_scale: float = NORMALIZATION_SCALE,
        palm_frame: bool = True,
    ):
        self.encoder_def = encoder_def
        self.encoder_params = encoder_params
        self.normalize_scale = normalize_scale
        self.embed_dim = encoder_def.output_dim
        self._palm_frame_requested = palm_frame
        self.frame = None  # set on first call

    @classmethod
    def load_checkpoint(cls, path: str) -> "ShardedRaycastConditioner":
        """Restore a sharded raycast conditioner from the standard checkpoint.

        The sharded and Gymnasium raycast paths share the same PointNet
        parameter format.  Only the extraction backend differs.
        """
        base = OnlineRaycastConditioner.load_checkpoint(path)
        return cls(
            encoder_def=base.encoder_def,
            encoder_params=base.encoder_params,
            normalize_scale=base.normalize_scale,
            palm_frame=True,
        )

    def extract_and_encode_sharded(self, env) -> np.ndarray:
        """Extract raycast pointclouds from shards and encode.

        Parameters
        ----------
        env : ShardedMjlabParallelEnv with enable_raycast_sensor=True

        Returns
        -------
        embeddings : (num_slots, embed_dim) float32
        """
        rc = env.extract_raycast_pointclouds()
        hit_pos = rc["hit_pos_w"]
        normals = rc["normals_w"]
        distances = rc["distances"]

        mask = distances > 0.0

        if self._palm_frame_requested:
            try:
                palm_pos, palm_quat = env.extract_palm_poses()
                palm_rot = self._quat_to_rotmat(palm_quat)
                points = self._to_palm_frame(hit_pos, palm_pos, palm_rot)
                self.frame = "palm"
            except (AttributeError, KeyError):
                if self.frame is None:
                    import sys
                    print("WARNING: ShardedRaycastConditioner: palm body data "
                          "unavailable, using world-frame pointclouds.", file=sys.stderr)
                points = hit_pos.copy()
                self.frame = "world"
        else:
            points = hit_pos.copy()
            self.frame = "world"

        points = np.where(mask[..., None], points, 0.0)
        points = points / self.normalize_scale

        embeddings = self.encoder_def.apply(
            {"params": self.encoder_params},
            jnp.array(points),
            jnp.array(mask),
        )
        return np.asarray(embeddings)

    @staticmethod
    def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
        """Convert (N, 4) quaternion (w,x,y,z) to (N, 3, 3) rotation matrix."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        R = np.zeros((len(quat), 3, 3), dtype=np.float32)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        return R

    @staticmethod
    def _to_palm_frame(
        hit_pos: np.ndarray,
        palm_pos: np.ndarray,
        palm_rot: np.ndarray,
    ) -> np.ndarray:
        """Transform (N, R, 3) hit positions to palm frame."""
        rel = hit_pos - palm_pos[:, None, :]
        return np.einsum("nij,nrj->nri", palm_rot.transpose(0, 2, 1), rel)


def build_sharded_raycast_conditioner(
    seed: int = 0,
    output_dim: int = DEFAULT_EMBED_DIM,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    n_points: int = 1024,
    palm_frame: bool = True,
) -> ShardedRaycastConditioner:
    """Create a ShardedRaycastConditioner with random encoder parameters."""
    encoder_def = MaskAwarePointNet(
        hidden_dims=hidden_dims, output_dim=output_dim
    )
    rng = jax.random.PRNGKey(seed)
    dummy_points = jnp.zeros((1, n_points, RAYCAST_POINT_CHANNELS))
    dummy_mask = jnp.ones((1, n_points), dtype=bool)
    variables = encoder_def.init(rng, dummy_points, dummy_mask)
    encoder_params = variables["params"]

    return ShardedRaycastConditioner(
        encoder_def=encoder_def,
        encoder_params=encoder_params,
        palm_frame=palm_frame,
    )
