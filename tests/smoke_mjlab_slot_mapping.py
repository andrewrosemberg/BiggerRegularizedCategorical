"""Smoke test for mjlab multi-object slot metadata.

The mjlab backend can place several object meshes inside one batched
ManagerBasedRlEnv by assigning a mesh variant to each parallel environment
slot. BRC still needs to know which object each slot represents so replay,
reward normalization, and logging can aggregate by object rather than by raw
slot index. This test verifies that the adapter exposes a deterministic,
balanced slot-to-object mapping and that the mapping matches the order supplied
by the caller.

Run with the mjlab environment:

    .venv_mjlab_check/bin/python tests/smoke_mjlab_slot_mapping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jaxrl.mjlab_envs import MjlabParallelEnv, _balanced_variant_assignment


def main() -> None:
    assignment = _balanced_variant_assignment(num_objects=3, num_envs=8)
    expected_assignment = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
    np.testing.assert_array_equal(assignment, expected_assignment)

    env = MjlabParallelEnv(
        ["cube", "ball", "apple"],
        seed=0,
        num_envs=12,
    )
    try:
        assert env.unique_object_names == ("cube", "ball", "apple")
        np.testing.assert_array_equal(
            env.object_ids,
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            env.object_names_by_slot,
            np.array(
                [
                    "cube", "cube", "cube", "cube",
                    "ball", "ball", "ball", "ball",
                    "apple", "apple", "apple", "apple",
                ],
                dtype=object,
            ),
        )
        assert env.object_id_by_name == {"cube": 0, "ball": 1, "apple": 2}
        assert env.slot_counts_by_object == {"cube": 4, "ball": 4, "apple": 4}

        obs = env.reset()
        assert obs.shape == (12, 68), obs.shape
    finally:
        env.close()

    print("mjlab slot mapping smoke: PASS")


if __name__ == "__main__":
    main()
