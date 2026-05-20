"""Deterministic mesh-based shape conditioning for geometry-conditioned BRC.

The features here are hand-designed placeholders for the conditioning
infrastructure.  The interface is designed so a learned PointNet embedding
can replace them later without changing the BRC plumbing.
"""

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


def load_manifest_objects(manifest_path: str, split: str = "train") -> list[str]:
    """Load object names from a split manifest JSON."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest[split]
