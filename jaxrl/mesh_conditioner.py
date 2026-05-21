"""Mesh-based conditioning for geometry-conditioned BRC.

Provides:

- Deterministic 8D mesh descriptors (placeholder, still valid for
  ``mesh_shape`` mode until a learned PointNet replaces them).
- ``MeshPoseConditioner``: online conditioner for ``mesh_pose`` mode.
  Combines static mesh-shape embeddings with per-step object pose
  variables extracted from the simulator.

Mesh-pose design decision
--------------------------
The ``mesh_pose`` conditioner uses **structured append**: the per-step
conditioning vector is ``[z_m, p_t, rot6d(R_t)]`` where

- ``z_m`` is the static mesh-shape embedding (same as ``mesh_shape``),
- ``p_t`` is the object position in palm frame (3D),
- ``rot6d(R_t)`` is the object orientation in palm frame encoded as the
  first two columns of the rotation matrix (6D).

This representation was chosen over re-encoding transformed mesh points
because:

1. It clearly separates static shape from dynamic pose.
2. It makes the privileged information (simulator object pose) explicit
   and auditable.
3. The mesh-shape encoder is shared with ``mesh_shape`` mode.
4. Appending 9 pose dimensions is cheaper than running a second PointNet
   on transformed mesh vertices every step.

**Privileged information label**: ``mesh_pose`` uses simulator object pose
directly.  Results from this mode must be labeled *privileged* or
*upper-bound* unless an equivalent real-time pose estimation pipeline is
demonstrated.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

MESH_FEATURE_NAMES = [
    "extent_x",
    "extent_y",
    "extent_z",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "diagonal",
    "log_vertex_count",
]

MESH_FEATURE_DIM = len(MESH_FEATURE_NAMES)


def load_binary_stl(path: str):
    """Load vertices and face normals from a binary STL file."""
    with open(path, "rb") as f:
        f.read(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]
        dt = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("v3", "<f4", (3,)),
                ("attr", "<u2"),
            ]
        )
        data = np.fromfile(f, dtype=dt, count=num_triangles)
    normals = data["normal"]
    vertices = np.concatenate([data["v1"], data["v2"], data["v3"]], axis=0)
    return vertices, normals


def load_binary_stl_triangles(path: str):
    """Load triangle vertex arrays and face normals from a binary STL file.

    Returns
    -------
    v1, v2, v3 : ndarray, shape (num_triangles, 3) each
    normals : ndarray, shape (num_triangles, 3)
    """
    with open(path, "rb") as f:
        f.read(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]
        dt = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("v3", "<f4", (3,)),
                ("attr", "<u2"),
            ]
        )
        data = np.fromfile(f, dtype=dt, count=num_triangles)
    return data["v1"], data["v2"], data["v3"], data["normal"]


def compute_mesh_features(vertices: np.ndarray) -> np.ndarray:
    """Compute deterministic shape descriptor from mesh vertices.

    Returns an 8-dim vector:  bbox extents (3), centroid (3),
    diagonal (1), log vertex count (1).
    """
    centroid = vertices.mean(axis=0)
    bb_min = vertices.min(axis=0)
    bb_max = vertices.max(axis=0)
    extents = bb_max - bb_min
    diagonal = np.linalg.norm(extents)
    log_vcount = np.log1p(float(len(vertices)))
    features = np.concatenate(
        [extents, centroid, [diagonal], [log_vcount]]
    ).astype(np.float32)
    return features


def resolve_stl_path(object_name: str, assets_dir: str) -> str:
    """Resolve the STL file for *object_name* by parsing its MuJoCo XML."""
    xml_path = os.path.join(assets_dir, "hand", f"manipulate_{object_name}.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "../stls/hand") if compiler is not None else "../stls/hand"
    for elem in root.iter("mesh"):
        if elem.get("name") == "mesh:object":
            stl_rel = elem.get("file")
            stl_path = os.path.join(os.path.dirname(xml_path), meshdir, stl_rel)
            return os.path.normpath(stl_path)
    raise ValueError(f"No mesh:object found in {xml_path}")


def _compute_raw_features(object_names: list[str], assets_dir: str) -> np.ndarray:
    """Compute raw (unnormalized) feature matrix for a list of objects."""
    all_features = []
    for name in object_names:
        stl_path = resolve_stl_path(name, assets_dir)
        vertices, _ = load_binary_stl(stl_path)
        features = compute_mesh_features(vertices)
        all_features.append(features)
    return np.stack(all_features)


def build_conditioner_features(
    object_names: list[str],
    assets_dir: str,
    normalize: bool = True,
) -> tuple[np.ndarray, dict]:
    """Build a conditioner feature matrix for a list of objects.

    Returns
    -------
    features : ndarray, shape (num_objects, MESH_FEATURE_DIM)
    meta : dict with feature names, normalization stats, and object list
    """
    features = _compute_raw_features(object_names, assets_dir)

    meta: dict = {
        "feature_names": list(MESH_FEATURE_NAMES),
        "feature_dim": int(features.shape[1]),
        "num_objects": len(object_names),
        "object_names": list(object_names),
        "normalized": normalize,
    }

    if normalize:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        features = (features - mean) / std
        meta["normalize_mean"] = mean.tolist()
        meta["normalize_std"] = std.tolist()

    return features.astype(np.float32), meta


def build_conditioner_features_from_manifest(
    object_names: list[str],
    assets_dir: str,
    manifest_path: str,
) -> tuple[np.ndarray, dict]:
    """Build features for *object_names*, normalized with train-split stats.

    Validates that every requested object appears in the manifest.
    Normalization statistics are computed from train-split objects only,
    preventing held-out normalization leakage.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    train_objects = manifest["train"]
    test_objects = manifest["test"]
    all_manifest = set(train_objects) | set(test_objects)

    missing = [n for n in object_names if n not in all_manifest]
    if missing:
        raise ValueError(
            f"Objects not found in manifest {manifest_path}: {missing}"
        )

    train_raw = _compute_raw_features(train_objects, assets_dir)
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    raw = _compute_raw_features(object_names, assets_dir)
    features = ((raw - mean) / std).astype(np.float32)

    meta: dict = {
        "feature_names": list(MESH_FEATURE_NAMES),
        "feature_dim": MESH_FEATURE_DIM,
        "num_objects": len(object_names),
        "object_names": list(object_names),
        "normalized": True,
        "normalize_source": "train_split",
        "normalize_mean": mean.tolist(),
        "normalize_std": std.tolist(),
        "train_objects_count": len(train_objects),
        "manifest_path": manifest_path,
    }

    return features, meta


def build_learned_mesh_features(
    object_names: list[str],
    assets_dir: str,
    encoder_checkpoint_path: str,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Build per-object learned mesh features using a trained PointNet encoder.

    Loads the mesh PointNet checkpoint, samples canonical mesh points
    deterministically for each object, encodes them, and returns a feature
    matrix aligned to *object_names*.

    Returns
    -------
    features : ndarray, shape (num_objects, output_dim)
    meta : dict with encoder metadata and object list
    """
    import jax.numpy as jnp
    encoder_def, encoder_params, ckpt_meta = load_mesh_encoder_checkpoint(
        encoder_checkpoint_path,
    )
    n_points = ckpt_meta["n_points"]

    all_features = []
    for obj_name in object_names:
        obj_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{obj_name}".encode()).digest()[:4],
            "little",
        )
        rng = np.random.RandomState(obj_seed)

        stl_path = resolve_stl_path(obj_name, assets_dir)
        v1, v2, v3, _ = load_binary_stl_triangles(stl_path)

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
        pts = pts.astype(np.float32)
        pts -= pts.mean(axis=0, keepdims=True)
        max_dist = np.linalg.norm(pts, axis=-1).max()
        if max_dist > 1e-8:
            pts /= max_dist

        pts_jnp = jnp.array(pts[None])
        mask_jnp = jnp.ones((1, n_points), dtype=bool)
        feat = encoder_def.apply({"params": encoder_params}, pts_jnp, mask_jnp)
        all_features.append(np.asarray(feat[0]))

    features = np.stack(all_features).astype(np.float32)
    meta = {
        "feature_source": "learned_mesh_pointnet",
        "encoder_checkpoint": encoder_checkpoint_path,
        "feature_dim": int(features.shape[1]),
        "num_objects": len(object_names),
        "object_names": list(object_names),
        "n_points": n_points,
        "encoder_hidden_dims": ckpt_meta.get("hidden_dims"),
        "encoder_output_dim": ckpt_meta.get("output_dim"),
    }
    return features, meta


def load_manifest_objects(manifest_path: str, split: str = "train") -> list[str]:
    """Load object names from a split manifest JSON."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest[split]


# ---------------------------------------------------------------------------
# Mesh-pose conditioner (privileged: uses simulator object pose)
# ---------------------------------------------------------------------------

POSE_DIM = 9  # position (3) + rot6d (6)
PALM_BODY_NAME = "robot0:palm"


def _rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Extract 6D rotation representation (first two columns of R)."""
    return rotation_matrix[:, :2].flatten()


def extract_object_pose_palm_frame(env) -> np.ndarray:
    """Extract object position and rotation in palm frame from a live env.

    Returns a 9D vector: [palm-frame position (3), rot6d (6)].
    """
    import mujoco

    uw = env.unwrapped
    model, data = uw.model, uw.data

    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY_NAME)
    palm_pos = data.xpos[palm_id]
    palm_xmat = data.xmat[palm_id].reshape(3, 3)

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_id < 0:
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object:joint")
    if obj_id < 0:
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "object" in name.lower():
                obj_id = i
                break
    assert obj_id >= 0, "Could not find object body in model"

    obj_pos_world = data.xpos[obj_id]
    obj_xmat_world = data.xmat[obj_id].reshape(3, 3)

    pos_palm = palm_xmat.T @ (obj_pos_world - palm_pos)
    rot_palm = palm_xmat.T @ obj_xmat_world
    rot6d = _rot6d(rot_palm)

    return np.concatenate([pos_palm, rot6d]).astype(np.float32)


class MeshPoseConditioner:
    """Online conditioner that appends pose variables to mesh-shape embeddings.

    The embedding for each task at step t is::

        c_t = [z_m, p_t / pos_scale, rot6d(R_t)]

    where z_m is the static mesh-shape feature and (p_t, R_t) are extracted
    from the simulator.

    Attributes
    ----------
    embed_dim : int
        Total embedding dimension = mesh_feature_dim + POSE_DIM.
    """

    def __init__(
        self,
        shape_features: np.ndarray,
        pos_scale: float = 0.34,
    ):
        self.shape_features = shape_features.astype(np.float32)
        self.pos_scale = pos_scale
        self.embed_dim = shape_features.shape[1] + POSE_DIM

    def extract_and_encode(self, envs: list) -> np.ndarray:
        """Compute per-task mesh-pose embeddings from live environments.

        Parameters
        ----------
        envs : list of gym.Env

        Returns
        -------
        embeddings : (num_envs, embed_dim) float32
        """
        embeddings = []
        for i, env in enumerate(envs):
            pose = extract_object_pose_palm_frame(env)
            pose[:3] /= self.pos_scale
            emb = np.concatenate([self.shape_features[i], pose])
            embeddings.append(emb)
        return np.stack(embeddings)


def build_mesh_pose_conditioner(
    object_names: list[str],
    assets_dir: str,
    manifest_path: str,
    seed: int = 0,
    embed_dim: int = 64,
) -> MeshPoseConditioner:
    """Build a MeshPoseConditioner with manifest-normalized mesh features.

    The ``embed_dim`` argument is accepted for API symmetry with the raycast
    conditioner but is not used: the mesh-pose embedding dimension is
    ``MESH_FEATURE_DIM + POSE_DIM``.
    """
    features, _meta = build_conditioner_features_from_manifest(
        object_names, assets_dir, manifest_path,
    )
    return MeshPoseConditioner(shape_features=features)


# ---------------------------------------------------------------------------
# Mesh encoder checkpoint save/load
# ---------------------------------------------------------------------------

def save_mesh_encoder_checkpoint(
    path: str,
    encoder_params,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    n_points: int,
    point_channels: int = 3,
    extra_metadata: dict | None = None,
):
    """Save a mesh PointNet encoder checkpoint."""
    import flax.serialization

    os.makedirs(path, exist_ok=True)

    params_path = os.path.join(path, "encoder_params.bin")
    with open(params_path, "wb") as f:
        f.write(flax.serialization.to_bytes(encoder_params))

    meta = {
        "encoder_type": "MaskAwarePointNet",
        "usage": "mesh_shape",
        "hidden_dims": list(hidden_dims),
        "output_dim": output_dim,
        "n_points": n_points,
        "point_channels": point_channels,
        "mask_convention": "all True (mesh points always valid)",
        "coordinate_frame": "object canonical frame, centered and scaled",
    }
    if extra_metadata:
        meta.update(extra_metadata)

    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_mesh_encoder_checkpoint(path: str):
    """Load a mesh PointNet encoder from a checkpoint directory.

    Returns (encoder_def, encoder_params, metadata).
    """
    import flax.serialization
    import jax
    import jax.numpy as jnp
    from jaxrl.pointnet import MaskAwarePointNet

    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    hidden_dims = tuple(meta["hidden_dims"])
    output_dim = meta["output_dim"]
    n_points = meta["n_points"]
    point_channels = meta.get("point_channels", 3)

    encoder_def = MaskAwarePointNet(
        hidden_dims=hidden_dims, output_dim=output_dim
    )
    dummy_points = jnp.zeros((1, n_points, point_channels))
    dummy_mask = jnp.ones((1, n_points), dtype=bool)
    rng = jax.random.PRNGKey(0)
    variables = encoder_def.init(rng, dummy_points, dummy_mask)
    template_params = variables["params"]

    params_path = os.path.join(path, "encoder_params.bin")
    with open(params_path, "rb") as f:
        encoder_params = flax.serialization.from_bytes(
            template_params, f.read()
        )

    return encoder_def, encoder_params, meta
