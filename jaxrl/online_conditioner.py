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

from typing import Any

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
