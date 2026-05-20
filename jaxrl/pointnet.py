"""Mask-aware PointNet-style encoder for geometry-conditioned BRC.

Applies shared per-point MLPs followed by masked max-pooling.  Invalid points
(where the hit mask is False) are set to -inf before the max so they never
contribute to the pooled feature.  When all points are invalid the output is
the zero vector.

Tensor contract for wrist-raycast input
----------------------------------------
- ``points``: ``(batch, N, 3)`` float32 -- palm-frame xyz divided by
  ``NORMALIZATION_SCALE`` (default 0.34 m = max raycast distance).
- ``mask``:   ``(batch, N)``    bool    -- True where the ray hit geometry.
- N = ``grid_h * grid_w`` = 1024 for the default 32x32 ray grid.
- Invalid ray slots carry zeros in the point tensor and False in the mask.
- Coordinate frame: palm frame (see ``raycast_conditioner.py``).
- Channels: xyz only.  Normals or distance can be appended later by
  increasing the input channel count.

The same module is reused for mesh-shape encoding.  In that case ``mask`` is
all-True because every sampled mesh point is valid.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

RAYCAST_POINT_CHANNELS = 3
DEFAULT_EMBED_DIM = 64
DEFAULT_HIDDEN_DIMS = (64, 128, 256)
NORMALIZATION_SCALE = 0.34


class MaskAwarePointNet(nn.Module):
    """Mask-aware PointNet encoder with masked max-pooling."""

    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    output_dim: int = DEFAULT_EMBED_DIM

    @nn.compact
    def __call__(self, points: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Encode a batch of masked point clouds.

        Parameters
        ----------
        points : (batch, N, C) float32
        mask   : (batch, N) bool -- True where valid

        Returns
        -------
        embedding : (batch, output_dim) float32
        """
        x = points
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        mask_exp = mask[..., None]
        x = jnp.where(mask_exp, x, -jnp.inf)
        x = jnp.max(x, axis=-2)

        all_miss = ~jnp.any(mask, axis=-1, keepdims=True)
        x = jnp.where(all_miss, 0.0, x)

        x = nn.Dense(self.output_dim)(x)
        return x
