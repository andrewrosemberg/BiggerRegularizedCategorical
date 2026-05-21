"""Train a mesh PointNet encoder for shape embeddings.

Samples surface points from canonical object meshes in the asset tree,
then trains a MaskAwarePointNet shape encoder.  The encoder maps sampled
mesh pointclouds to shape embedding vectors, trained with:
  - supervised contrastive loss (same object -> similar embedding),
  - optional reconstruction proxy (predict mesh extent/centroid).

This produces a learned replacement for the deterministic 8D mesh
descriptor used by ``mesh_shape`` and ``mesh_pose`` modes.

Usage (short sanity run):
    module load python/3.11.9 && source .venv/bin/activate
    python scripts/train_mesh_pointnet.py \
        --manifest=manifests/shadowhand_split_v1.json \
        --epochs=5 --n_points=512 --batch_size=32 \
        --checkpoint_dir=checkpoints/mesh_pointnet_smoke

Full training:
    python scripts/train_mesh_pointnet.py \
        --manifest=manifests/shadowhand_split_v1.json \
        --epochs=100 --n_points=1024 --batch_size=64 \
        --checkpoint_dir=checkpoints/mesh_pointnet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import flax.serialization

from jaxrl.mesh_conditioner import (
    load_manifest_objects,
    resolve_stl_path,
    load_binary_stl,
    load_binary_stl_triangles,
    compute_mesh_features,
    save_mesh_encoder_checkpoint,
    load_mesh_encoder_checkpoint,
    MESH_FEATURE_DIM,
)
from jaxrl.pointnet import MaskAwarePointNet


def sample_mesh_surface_points(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    n_points: int,
    rng: np.random.RandomState,
    center: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """Sample points from mesh surface with triangle-area weighting.

    Each triangle is sampled with probability proportional to its area.
    Points are placed uniformly within the selected triangle using
    barycentric coordinates.
    """
    e1 = v2 - v1
    e2 = v3 - v1
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=-1)
    total_area = areas.sum()
    if total_area < 1e-12:
        pts = v1[:min(n_points, len(v1))].copy()
        if len(pts) < n_points:
            idx = rng.choice(len(pts), size=n_points, replace=True)
            pts = pts[idx]
    else:
        probs = areas / total_area
        tri_idx = rng.choice(len(areas), size=n_points, replace=True, p=probs)
        r1 = rng.uniform(size=n_points).astype(np.float32)
        r2 = rng.uniform(size=n_points).astype(np.float32)
        sqrt_r1 = np.sqrt(r1)
        u = 1.0 - sqrt_r1
        v = sqrt_r1 * (1.0 - r2)
        w = sqrt_r1 * r2
        pts = (u[:, None] * v1[tri_idx]
               + v[:, None] * v2[tri_idx]
               + w[:, None] * v3[tri_idx])

    if center:
        pts -= pts.mean(axis=0, keepdims=True)
    if scale:
        max_dist = np.linalg.norm(pts, axis=-1).max()
        if max_dist > 1e-8:
            pts /= max_dist

    return pts.astype(np.float32)


def build_mesh_dataset(
    object_names: list[str],
    assets_dir: str,
    n_points: int,
    augmentations_per_object: int = 10,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build a dataset of sampled mesh pointclouds."""
    import dex_envs  # noqa: F401

    rng = np.random.RandomState(seed)
    all_points = []
    all_obj_ids = []
    all_descriptors = []

    for obj_idx, obj_name in enumerate(object_names):
        stl_path = resolve_stl_path(obj_name, assets_dir)
        vertices, normals = load_binary_stl(stl_path)
        desc = compute_mesh_features(vertices)
        v1, v2, v3, _ = load_binary_stl_triangles(stl_path)

        for _ in range(augmentations_per_object):
            pts = sample_mesh_surface_points(v1, v2, v3, n_points, rng)
            all_points.append(pts)
            all_obj_ids.append(obj_idx)
            all_descriptors.append(desc)

    return {
        "points": np.stack(all_points),
        "object_ids": np.array(all_obj_ids, dtype=np.int32),
        "descriptors": np.stack(all_descriptors),
    }


def _make_mesh_encoder_class(hidden_dims, trunk_dim, descriptor_dim=MESH_FEATURE_DIM):
    """Build a MeshEncoderWithHeads class at call time to avoid Flax dataclass issues."""
    class MeshEncoderWithHeads(nn.Module):
        @nn.compact
        def __call__(self, points, mask):
            x = points
            for dim in hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.relu(x)
            mask_exp = mask[..., None]
            x = jnp.where(mask_exp, x, -jnp.inf)
            x = jnp.max(x, axis=-2)
            all_miss = ~jnp.any(mask, axis=-1, keepdims=True)
            x = jnp.where(all_miss, 0.0, x)
            trunk_out = nn.Dense(trunk_dim)(x)
            descriptor_pred = nn.Dense(descriptor_dim, name="descriptor_head")(trunk_out)
            return trunk_out, descriptor_pred
    return MeshEncoderWithHeads()


def _contrastive_loss(codes: jnp.ndarray, labels: jnp.ndarray, tau: float = 0.1) -> jnp.ndarray:
    codes_norm = codes / (jnp.linalg.norm(codes, axis=-1, keepdims=True) + 1e-8)
    sim = codes_norm @ codes_norm.T / tau
    n = codes.shape[0]
    mask_self = ~jnp.eye(n, dtype=bool)
    mask_pos = (labels[:, None] == labels[None, :]) & mask_self
    exp_sim = jnp.exp(sim - jnp.max(sim, axis=-1, keepdims=True))
    exp_sim = exp_sim * mask_self
    denom = jnp.sum(exp_sim, axis=-1, keepdims=True) + 1e-8
    log_prob = jnp.log(exp_sim / denom + 1e-8)
    n_pos = jnp.sum(mask_pos, axis=-1)
    has_pos = n_pos > 0
    loss_per = -jnp.sum(log_prob * mask_pos, axis=-1) / jnp.maximum(n_pos, 1)
    return jnp.mean(jnp.where(has_pos, loss_per, 0.0))


def train_step(params, model, opt_state, tx, batch, lambda_contrastive=1.0, lambda_desc=1.0):
    def loss_fn(p):
        trunk, desc_pred = model.apply({"params": p}, batch["points"], batch["masks"])
        loss_contrastive = _contrastive_loss(trunk, batch["object_ids"])
        loss_desc = jnp.mean(optax.huber_loss(desc_pred, batch["descriptors"], delta=0.5))
        total = lambda_contrastive * loss_contrastive + lambda_desc * loss_desc
        return total, {"loss_contrastive": loss_contrastive, "loss_desc": loss_desc, "total": total}

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, info


def evaluate_embeddings(params, model, dataset: dict, object_names: list[str]) -> dict:
    """Evaluate embedding quality: intra-class similarity, inter-class distance."""
    trunk, desc_pred = model.apply(
        {"params": params},
        jnp.array(dataset["points"]),
        jnp.array(dataset["masks"]),
    )
    trunk_np = np.array(trunk)
    trunk_norm = trunk_np / (np.linalg.norm(trunk_np, axis=-1, keepdims=True) + 1e-8)
    obj_ids = dataset["object_ids"]

    unique_ids = np.unique(obj_ids)
    centroids = []
    for oid in unique_ids:
        mask = obj_ids == oid
        centroids.append(trunk_norm[mask].mean(axis=0))
    centroids = np.stack(centroids)

    centroid_norms = centroids / (np.linalg.norm(centroids, axis=-1, keepdims=True) + 1e-8)
    sim_matrix = centroid_norms @ centroid_norms.T
    off_diag = sim_matrix[~np.eye(len(unique_ids), dtype=bool)]

    desc_err = np.mean(np.abs(np.array(desc_pred) - dataset["descriptors"]), axis=-1)

    return {
        "mean_inter_class_sim": float(np.mean(off_diag)),
        "mean_intra_class_sim": float(np.mean([
            np.mean(trunk_norm[obj_ids == oid] @ trunk_norm[obj_ids == oid].T)
            for oid in unique_ids
        ])),
        "mean_desc_mae": float(np.mean(desc_err)),
        "n_objects": int(len(unique_ids)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train mesh PointNet shape encoder")
    parser.add_argument("--manifest", required=True, help="Split manifest JSON path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--augmentations_per_object", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=str, default="64,128,256")
    parser.add_argument("--trunk_dim", type=int, default=64)
    parser.add_argument("--checkpoint_dir", default="checkpoints/mesh_pointnet")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    import dex_envs
    assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), "assets")

    train_objects = load_manifest_objects(args.manifest, "train")
    test_objects = load_manifest_objects(args.manifest, "test")
    print(f"Train objects: {len(train_objects)}, Test objects: {len(test_objects)}")

    print("Building train dataset...")
    t0 = time.time()
    train_data = build_mesh_dataset(
        train_objects, assets_dir, args.n_points,
        augmentations_per_object=args.augmentations_per_object, seed=args.seed,
    )
    train_data["masks"] = np.ones((train_data["points"].shape[0], args.n_points), dtype=bool)
    print(f"  {train_data['points'].shape[0]} samples in {time.time() - t0:.1f}s")

    print("Building test dataset...")
    t0 = time.time()
    test_data = build_mesh_dataset(
        test_objects, assets_dir, args.n_points,
        augmentations_per_object=max(3, args.augmentations_per_object // 4),
        seed=args.seed + 1000,
    )
    test_data["masks"] = np.ones((test_data["points"].shape[0], args.n_points), dtype=bool)
    test_data["object_ids"] = test_data["object_ids"] + len(train_objects)
    print(f"  {test_data['points'].shape[0]} samples in {time.time() - t0:.1f}s")

    print("Building model...")
    model = _make_mesh_encoder_class(hidden_dims, args.trunk_dim)
    rng = jax.random.PRNGKey(args.seed)
    dummy_pts = jnp.zeros((1, args.n_points, 3))
    dummy_mask = jnp.ones((1, args.n_points), dtype=bool)
    variables = model.init(rng, dummy_pts, dummy_mask)
    params = variables["params"]

    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    jit_train_step = jax.jit(lambda p, os, b: train_step(p, model, os, tx, b))

    n_train = train_data["points"].shape[0]
    rng_np = np.random.RandomState(args.seed)

    # Normalize descriptors for training
    desc_mean = train_data["descriptors"].mean(axis=0)
    desc_std = train_data["descriptors"].std(axis=0)
    desc_std = np.where(desc_std < 1e-8, 1.0, desc_std)
    train_data["descriptors"] = (train_data["descriptors"] - desc_mean) / desc_std
    test_data["descriptors"] = (test_data["descriptors"] - desc_mean) / desc_std

    print(f"\nTraining for {args.epochs} epochs, {n_train} samples, batch_size={args.batch_size}")
    for epoch in range(1, args.epochs + 1):
        perm = rng_np.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch = {
                "points": jnp.array(train_data["points"][idx]),
                "masks": jnp.array(train_data["masks"][idx]),
                "object_ids": jnp.array(train_data["object_ids"][idx]),
                "descriptors": jnp.array(train_data["descriptors"][idx]),
            }
            params, opt_state, info = jit_train_step(params, opt_state, batch)
            epoch_losses.append(float(info["total"]))

        mean_loss = np.mean(epoch_losses)
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            train_metrics = evaluate_embeddings(params, model, train_data, train_objects)
            test_metrics = evaluate_embeddings(params, model, test_data, test_objects)
            print(
                f"Epoch {epoch:4d} | loss={mean_loss:.4f} | "
                f"train intra={train_metrics['mean_intra_class_sim']:.3f} "
                f"inter={train_metrics['mean_inter_class_sim']:.3f} "
                f"desc_mae={train_metrics['mean_desc_mae']:.3f} | "
                f"test intra={test_metrics['mean_intra_class_sim']:.3f} "
                f"inter={test_metrics['mean_inter_class_sim']:.3f}"
            )
        else:
            print(f"Epoch {epoch:4d} | loss={mean_loss:.4f}")

    # Save checkpoint - extract trunk-compatible params
    trunk_keys = [k for k in sorted(params.keys()) if k != "descriptor_head"]
    trunk_params = {k: params[k] for k in trunk_keys}

    extra_meta = {
        "manifest_path": args.manifest,
        "train_objects": len(train_objects),
        "test_objects": len(test_objects),
        "epochs": args.epochs,
        "n_points": args.n_points,
        "augmentations_per_object": args.augmentations_per_object,
        "seed": args.seed,
        "descriptor_normalize_mean": desc_mean.tolist(),
        "descriptor_normalize_std": desc_std.tolist(),
    }
    save_mesh_encoder_checkpoint(
        args.checkpoint_dir,
        encoder_params=trunk_params,
        hidden_dims=hidden_dims,
        output_dim=args.trunk_dim,
        n_points=args.n_points,
        extra_metadata=extra_meta,
    )

    full_path = os.path.join(args.checkpoint_dir, "full_model_params.bin")
    with open(full_path, "wb") as f:
        f.write(flax.serialization.to_bytes(params))

    final_train = evaluate_embeddings(params, model, train_data, train_objects)
    final_test = evaluate_embeddings(params, model, test_data, test_objects)
    results = {
        "train_metrics": final_train,
        "test_metrics": final_test,
        "config": vars(args),
    }
    results["config"]["hidden_dims"] = list(hidden_dims)
    results_path = os.path.join(args.checkpoint_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCheckpoint saved to {args.checkpoint_dir}")
    print(f"Train: intra_sim={final_train['mean_intra_class_sim']:.3f}, inter_sim={final_train['mean_inter_class_sim']:.3f}")
    print(f"Test:  intra_sim={final_test['mean_intra_class_sim']:.3f}, inter_sim={final_test['mean_inter_class_sim']:.3f}")


if __name__ == "__main__":
    main()
