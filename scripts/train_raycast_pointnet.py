"""Train a wrist-raycast PointNet encoder with supervised state-and-shape targets.

Collects wrist-raycast pointclouds from live ShadowHand environments using
train-split objects, then trains a MaskAwarePointNet to predict:
  - object position in palm frame (Huber loss),
  - object rotation in palm frame as rot6d (geodesic loss),
  - a morphology code trained with supervised contrastive loss.

Usage (short sanity run):
    module load python/3.11.9 && source .venv/bin/activate
    MUJOCO_GL=egl python scripts/train_raycast_pointnet.py \
        --manifest=manifests/shadowhand_split_v1.json \
        --epochs=2 --samples_per_object=5 --batch_size=32 \
        --checkpoint_dir=checkpoints/raycast_pointnet_smoke

Full training:
    MUJOCO_GL=egl python scripts/train_raycast_pointnet.py \
        --manifest=manifests/shadowhand_split_v1.json \
        --epochs=50 --samples_per_object=200 --batch_size=128 \
        --checkpoint_dir=checkpoints/raycast_pointnet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.mesh_conditioner import load_manifest_objects, extract_object_pose_palm_frame
from jaxrl.pointnet import MaskAwarePointNet, NORMALIZATION_SCALE, RAYCAST_POINT_CHANNELS
from jaxrl.raycast_conditioner import RaycastConfig, get_raycast_pointcloud


POS_DIM = 3
ROT6D_DIM = 6
POS_SCALE = NORMALIZATION_SCALE


def _rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    return rotation_matrix[:, :2].flatten()


def _geodesic_loss(pred_rot6d: jnp.ndarray, target_rot6d: jnp.ndarray) -> jnp.ndarray:
    """Approximate geodesic loss from 6D rotation representations."""
    def _to_rotmat(r6):
        a1, a2 = r6[:3], r6[3:6]
        b1 = a1 / (jnp.linalg.norm(a1) + 1e-8)
        b2 = a2 - jnp.dot(b1, a2) * b1
        b2 = b2 / (jnp.linalg.norm(b2) + 1e-8)
        b3 = jnp.cross(b1, b2)
        return jnp.stack([b1, b2, b3], axis=-1)

    R_pred = jax.vmap(_to_rotmat)(pred_rot6d)
    R_target = jax.vmap(_to_rotmat)(target_rot6d)
    RtR = jax.vmap(lambda a, b: a.T @ b)(R_pred, R_target)
    trace = jax.vmap(jnp.trace)(RtR)
    cos_angle = jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return jnp.mean(jnp.arccos(cos_angle))


def _contrastive_loss(codes: jnp.ndarray, labels: jnp.ndarray, tau: float = 0.1) -> jnp.ndarray:
    """Supervised contrastive loss over morphology codes."""
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


def collect_dataset(
    object_names: list[str],
    samples_per_object: int,
    raycast_config: RaycastConfig,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Collect raycast pointclouds and targets from live environments."""
    import dex_envs  # noqa: F401
    import gymnasium as gym
    import mujoco

    rng = np.random.RandomState(seed)
    all_points = []
    all_masks = []
    all_pos = []
    all_rot6d = []
    all_obj_ids = []

    for obj_idx, obj_name in enumerate(object_names):
        env = gym.make(f"{obj_name}-rotate-v1", reward_type="sparse")
        for s in range(samples_per_object):
            env.reset(seed=int(rng.randint(0, 1_000_000)))
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

            n_random_steps = rng.randint(0, 20)
            for _ in range(n_random_steps):
                action = env.action_space.sample()
                env.step(action)
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

            result = get_raycast_pointcloud(env.unwrapped.model, env.unwrapped.data, raycast_config)
            pts = result["points_palm"].astype(np.float32) / NORMALIZATION_SCALE
            hit_mask = result["hit_mask"]

            pose = extract_object_pose_palm_frame(env)
            pos_target = pose[:3] / POS_SCALE
            rot6d_target = pose[3:]

            all_points.append(pts)
            all_masks.append(hit_mask)
            all_pos.append(pos_target)
            all_rot6d.append(rot6d_target)
            all_obj_ids.append(obj_idx)

        env.close()
        if (obj_idx + 1) % 10 == 0:
            print(f"  collected {obj_idx + 1}/{len(object_names)} objects")

    return {
        "points": np.stack(all_points),
        "masks": np.stack(all_masks),
        "positions": np.stack(all_pos),
        "rot6d": np.stack(all_rot6d),
        "object_ids": np.array(all_obj_ids, dtype=np.int32),
    }


def _make_encoder_with_heads(hidden_dims, trunk_dim, morph_dim):
    """Build an EncoderWithHeads instance to avoid Flax dataclass issues."""
    import flax.linen as nn

    class EncoderWithHeads(nn.Module):
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
            trunk = nn.Dense(trunk_dim)(x)
            pos_pred = nn.Dense(POS_DIM, name="pos_head")(trunk)
            rot_pred = nn.Dense(ROT6D_DIM, name="rot_head")(trunk)
            morph_code = nn.Dense(morph_dim, name="morph_head")(trunk)
            return trunk, pos_pred, rot_pred, morph_code

    return EncoderWithHeads()


def build_model_and_params(
    n_points: int,
    hidden_dims: tuple[int, ...],
    trunk_dim: int,
    morph_dim: int,
    seed: int,
):
    """Build PointNet trunk with prediction heads."""
    rng = jax.random.PRNGKey(seed)
    dummy_pts = jnp.zeros((1, n_points, RAYCAST_POINT_CHANNELS))
    dummy_mask = jnp.ones((1, n_points), dtype=bool)

    model = _make_encoder_with_heads(hidden_dims, trunk_dim, morph_dim)
    variables = model.init(rng, dummy_pts, dummy_mask)
    return model, variables["params"]


def train_step(params, model, opt_state, tx, batch, lambda_pos=1.0, lambda_rot=0.5, lambda_shape=0.5):
    """One training step."""
    def loss_fn(p):
        trunk, pos_pred, rot_pred, morph_code = model.apply(
            {"params": p}, batch["points"], batch["masks"]
        )
        loss_pos = jnp.mean(optax.huber_loss(pos_pred, batch["positions"], delta=0.1))
        loss_rot = _geodesic_loss(rot_pred, batch["rot6d"])
        loss_shape = _contrastive_loss(morph_code, batch["object_ids"])
        total = lambda_pos * loss_pos + lambda_rot * loss_rot + lambda_shape * loss_shape
        return total, {"loss_pos": loss_pos, "loss_rot": loss_rot, "loss_shape": loss_shape, "total": total}

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, info


def evaluate_split(params, model, dataset: dict) -> dict:
    """Compute position and rotation errors on a dataset split."""
    trunk, pos_pred, rot_pred, morph_code = model.apply(
        {"params": params},
        jnp.array(dataset["points"]),
        jnp.array(dataset["masks"]),
    )
    pos_err = jnp.sqrt(jnp.sum((pos_pred - jnp.array(dataset["positions"])) ** 2, axis=-1))
    pos_err_m = pos_err * POS_SCALE

    def _single_geo(pred, target):
        def to_r(r6):
            a1, a2 = r6[:3], r6[3:]
            b1 = a1 / (jnp.linalg.norm(a1) + 1e-8)
            b2 = a2 - jnp.dot(b1, a2) * b1
            b2 = b2 / (jnp.linalg.norm(b2) + 1e-8)
            b3 = jnp.cross(b1, b2)
            return jnp.stack([b1, b2, b3], axis=-1)
        Rp = to_r(pred)
        Rt = to_r(target)
        tr = jnp.trace(Rp.T @ Rt)
        return jnp.arccos(jnp.clip((tr - 1.0) / 2.0, -1.0, 1.0))

    rot_err = jax.vmap(_single_geo)(rot_pred, jnp.array(dataset["rot6d"]))
    rot_err_deg = jnp.degrees(rot_err)

    return {
        "mean_pos_err_m": float(jnp.mean(pos_err_m)),
        "mean_rot_err_deg": float(jnp.mean(rot_err_deg)),
        "median_pos_err_m": float(jnp.median(pos_err_m)),
        "median_rot_err_deg": float(jnp.median(rot_err_deg)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train wrist-raycast PointNet encoder")
    parser.add_argument("--manifest", required=True, help="Split manifest JSON path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples_per_object", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=str, default="64,128,256")
    parser.add_argument("--trunk_dim", type=int, default=64)
    parser.add_argument("--morph_dim", type=int, default=16)
    parser.add_argument("--checkpoint_dir", default="checkpoints/raycast_pointnet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_samples_per_object", type=int, default=10)
    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    raycast_config = RaycastConfig()
    n_points = raycast_config.grid_h * raycast_config.grid_w

    train_objects = load_manifest_objects(args.manifest, "train")
    test_objects = load_manifest_objects(args.manifest, "test")
    print(f"Train objects: {len(train_objects)}, Test objects: {len(test_objects)}")

    print("Collecting train data...")
    t0 = time.time()
    train_data = collect_dataset(train_objects, args.samples_per_object, raycast_config, seed=args.seed)
    print(f"  collected {train_data['points'].shape[0]} samples in {time.time() - t0:.1f}s")

    print("Collecting held-out eval data...")
    t0 = time.time()
    test_data = collect_dataset(test_objects, args.eval_samples_per_object, raycast_config, seed=args.seed + 1000)
    print(f"  collected {test_data['points'].shape[0]} samples in {time.time() - t0:.1f}s")

    print("Building model...")
    model, params = build_model_and_params(
        n_points=n_points,
        hidden_dims=hidden_dims,
        trunk_dim=args.trunk_dim,
        morph_dim=args.morph_dim,
        seed=args.seed,
    )

    tx = optax.adam(args.lr)
    opt_state = tx.init(params)

    jit_train_step = jax.jit(
        lambda p, os, b: train_step(p, model, os, tx, b)
    )

    n_train = train_data["points"].shape[0]
    rng = np.random.RandomState(args.seed)

    print(f"\nTraining for {args.epochs} epochs, {n_train} samples, batch_size={args.batch_size}")
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch = {
                "points": jnp.array(train_data["points"][idx]),
                "masks": jnp.array(train_data["masks"][idx]),
                "positions": jnp.array(train_data["positions"][idx]),
                "rot6d": jnp.array(train_data["rot6d"][idx]),
                "object_ids": jnp.array(train_data["object_ids"][idx]),
            }
            params, opt_state, info = jit_train_step(params, opt_state, batch)
            epoch_losses.append(float(info["total"]))

        mean_loss = np.mean(epoch_losses)

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            train_metrics = evaluate_split(params, model, train_data)
            test_metrics = evaluate_split(params, model, test_data)
            print(
                f"Epoch {epoch:4d} | loss={mean_loss:.4f} | "
                f"train pos={train_metrics['mean_pos_err_m']*100:.2f}cm "
                f"rot={train_metrics['mean_rot_err_deg']:.1f}deg | "
                f"test pos={test_metrics['mean_pos_err_m']*100:.2f}cm "
                f"rot={test_metrics['mean_rot_err_deg']:.1f}deg"
            )
        else:
            print(f"Epoch {epoch:4d} | loss={mean_loss:.4f}")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Extract trunk params (same structure as MaskAwarePointNet)
    from jaxrl.online_conditioner import OnlineRaycastConditioner
    trunk_encoder = MaskAwarePointNet(hidden_dims=hidden_dims, output_dim=args.trunk_dim)

    # The params have trunk layers (Dense_0..Dense_N, Dense_{N+1} for output)
    # plus head layers. Extract trunk-compatible params.
    all_param_keys = sorted(params.keys())
    trunk_keys = [k for k in all_param_keys if k not in ("pos_head", "rot_head", "morph_head")]
    trunk_params = {k: params[k] for k in trunk_keys}

    conditioner = OnlineRaycastConditioner(
        encoder_def=trunk_encoder,
        encoder_params=trunk_params,
        raycast_config=raycast_config,
    )

    extra_meta = {
        "manifest_path": args.manifest,
        "train_objects": len(train_objects),
        "test_objects": len(test_objects),
        "epochs": args.epochs,
        "samples_per_object": args.samples_per_object,
        "morph_dim": args.morph_dim,
        "seed": args.seed,
    }
    conditioner.save_checkpoint(args.checkpoint_dir, extra_metadata=extra_meta)

    # Also save the full model params (including heads) for evaluation
    import flax.serialization
    full_params_path = os.path.join(args.checkpoint_dir, "full_model_params.bin")
    with open(full_params_path, "wb") as f:
        f.write(flax.serialization.to_bytes(params))

    final_train = evaluate_split(params, model, train_data)
    final_test = evaluate_split(params, model, test_data)
    results = {
        "train_metrics": final_train,
        "test_metrics": final_test,
        "config": {
            "hidden_dims": list(hidden_dims),
            "trunk_dim": args.trunk_dim,
            "morph_dim": args.morph_dim,
            "epochs": args.epochs,
            "samples_per_object": args.samples_per_object,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "manifest": args.manifest,
        },
    }
    results_path = os.path.join(args.checkpoint_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCheckpoint saved to {args.checkpoint_dir}")
    print(f"Final train: pos={final_train['mean_pos_err_m']*100:.2f}cm, rot={final_train['mean_rot_err_deg']:.1f}deg")
    print(f"Final test:  pos={final_test['mean_pos_err_m']*100:.2f}cm, rot={final_test['mean_rot_err_deg']:.1f}deg")


if __name__ == "__main__":
    main()
