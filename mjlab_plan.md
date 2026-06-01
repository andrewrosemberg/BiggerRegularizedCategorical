# Temporary Plan: mjlab GPU Environment Port

Date: 2026-06-01

This note evaluates whether this repository should replace the current CPU
Gymnasium/MuJoCo environment loop with a GPU mjlab environment path. The goal is
to reduce wall-clock time for ShadowHand object-rotation teacher training while
preserving the BRC replay, timeout, and geometry-conditioning contracts.

## 1. Motivation

The current training loop uses `jaxrl.envs.ParallelEnv`. For ShadowHand tasks it
creates one Gymnasium environment per object and steps them sequentially in
Python. A single training step over 85 train objects therefore performs 85 CPU
MuJoCo environment steps before the JAX BRC update.

The current 85-object benchmark projected the following 1M-step wall-clock
times:

| Mode | Projected 1M runtime |
|---|---:|
| `categorical` | 4.5 days |
| `none` | 4.3 days |
| `mesh_shape` | 3.6 days |
| `mesh_pose` | 3.1 days |
| `wrist_raycast` | 20.7 days |

The expensive part is not only policy optimization. It is also environment
throughput, especially for `wrist_raycast`, where each environment step computes
a 1024-ray wrist pointcloud. We want a vectorized GPU simulator path that can:

- train many parallel environments of the same object for single-object runs;
- train many parallel environments across many objects for multi-object runs;
- keep object identity available for replay balancing and logging, without
  feeding learned object-id embeddings to geometry-conditioned policies;
- keep online conditioners online: wrist raycasts and mesh-pose features must be
  computed from the current environment state.

## 2. Current Training Contract

The current loop in `train.py` has this order:

1. `actions = agent.sample_actions(observations)`;
2. `next_raw_obs, rewards, terms, truns, goals = env.step(actions)`;
3. online conditioner appends geometry embeddings to `next_raw_obs`;
4. `masks = env.generate_masks(terms, truns)`;
5. insert `(observations, actions, rewards, masks, next_observations)` into
   `ParallelReplayBuffer`;
6. reset only the environments where `terms` or `truns` is true;
7. use reset observations only as the starting observations for the next step.

The mask convention is:

$$
\text{mask}_t =
1 - \mathbb{1}[\text{terminated}_t \land \neg \text{truncated}_t].
$$

So true terminations have `mask=0`, while time-limit truncations have `mask=1`.
This matters because the critic target in `jaxrl/agent/update.py` uses

$$
r_t + \gamma\,\text{mask}_t\,V(o_{t+1}, c_{t+1}).
$$

Therefore a truncated transition must store the true timeout next observation.
If a vectorized simulator auto-resets and returns the reset observation as
`next_observations`, then truncation targets would incorrectly bootstrap from
the beginning of a new episode.

The current replay and normalization code also assume that the leading axis is
the object/task axis:

- `ParallelReplayBuffer` stores arrays with shape
  `(num_tasks, capacity, feature_dim)`;
- sampled `task_ids` are used by `RewardNormalizer`;
- geometry modes keep `task_ids` for storage, balanced sampling, normalization,
  and logging, not for learned categorical policy input.

A GPU environment port must preserve these semantics or replace them with an
equivalent object-aware design.

## 3. mjlab Findings

mjlab is a manager-based RL framework built on MuJoCo Warp and PyTorch. Its
README describes it as a GPU-accelerated MuJoCo framework and shows training
examples with thousands of parallel environments. The latest inspected release
is `v1.4.0`; the current `main` branch points to commit
`898a700514f8d3c92146ec158190e6f2e4d8c11a`.

### Auto-Reset

The installed mjlab copy in the existing mjlab project is older and defaults to
auto-resetting done environments inside `ManagerBasedRlEnv.step()`. That version
returns post-reset observations for environments that terminated or timed out.

The newest inspected mjlab release (`v1.4.0`) adds
`ManagerBasedRlEnvCfg.auto_reset`. With `auto_reset=False`, `step()` returns the
true terminal or timeout observation and requires the caller to call
`reset(env_ids=...)` before stepping those environment slots again. This matches
the reset order needed by this repository.

**Answer to the auto-reset check:** yes, newest mjlab supports auto-reset, and
it can also disable auto-reset. For BRC we should use `auto_reset=False`.

### Truncation

mjlab separates terminations from time-limit truncations through
`TerminationTermCfg(time_out=True)`. `ManagerBasedRlEnv.step()` returns separate
`terminated` and `truncated` tensors. Its RSL-RL wrapper also exposes
`extras["time_outs"] = truncated` when the task is not finite-horizon.

**Answer to the truncation check:** yes, newest mjlab has explicit truncation
semantics. For BRC, time limits should be configured as timeout terms and mapped
to the existing `truns` array.

### Multi-Object Batching

The existing mjlab project already contains an implementation reference for
same-process multi-object simulation with per-world mesh variants. It uses
`VariantEntityCfg` to assign different mesh-backed object variants to different
parallel worlds. The key constraint is that variants must share the same
kinematic topology; the mesh geometry may vary per world, but the body/joint
structure must be compatible.

This is promising for our 85-object setting because most ShadowHand object XMLs
have the same hand plus one free object body with one mesh geom. Any primitive
or structurally different object must be checked explicitly. If `cube` is a
primitive box in the current assets, it may need conversion to a mesh-backed
variant or a fallback single-object batch.

### Raycast Sensors

The existing mjlab project already uses `RayCastSensorCfg` with a
`PinholeCameraPatternCfg` and stores batched tensors such as hit positions,
normals, and distances. That is the right kind of primitive for replacing the
current CPU `mujoco.mj_ray` loop. The current wrist-camera pose and ray settings
from this repository should be ported, not retuned, so the perception contract
does not change.

## 4. Viability Assessment

Using newest mjlab is viable, but not as a drop-in replacement. The simulator
side has the features we need:

- GPU batched stepping through MuJoCo Warp;
- many parallel worlds in one environment;
- manual reset mode through `auto_reset=False`;
- separate terminated and truncated tensors;
- per-world mesh variants for same-topology mesh objects;
- batched raycast sensors suitable for the wrist-raycast conditioner.

The main engineering work is the adapter between mjlab's PyTorch vector
environment and this repository's JAX/NumPy BRC loop.

The highest-risk semantic issue is now manageable: if we use
`auto_reset=False`, BRC can insert the true terminal or timeout
`next_observations` before resetting done slots, preserving the current critic
target semantics. If we used older mjlab or `auto_reset=True`, truncation
bootstrapping would be wrong unless mjlab exposed a separate final observation.

## 5. Proposed Adapter Contract

Add a new optional environment path, for example `jaxrl/mjlab_envs.py`, that
presents the same logical API expected by `train.py`:

```python
obs = env.reset()
next_obs, rewards, terms, truns, goals = env.step(actions)
masks = env.generate_masks(terms, truns)
obs_after_reset, terms, truns = env.reset_where_done(next_obs, terms, truns)
```

Internally this adapter should:

- construct an mjlab `ManagerBasedRlEnv` with `auto_reset=False`;
- keep a fixed `env_slot -> object_id` array;
- return flattened actor observations with the same meaning as the current
  `FlattenObservationShadowhandWrapper`;
- map mjlab `terminated` to `terms` and mjlab `truncated` to `truns`;
- compute `goals` from an mjlab success metric matching the current
  `info["is_success"]`/`info["success"]` behavior;
- implement `reset_where_done()` by calling `mjlab_env.reset(env_ids=...)`;
- expose object ids for replay balancing, reward normalization, and logging.

For single-object training, this adapter should build one batched mjlab env with
`K` parallel slots all mapped to the same object id.

For multi-object training, the preferred path is one batched mjlab env with
per-world mesh variants and an assignment such as:

$$
\text{object\_id}_e \in \{1,\ldots,85\}, \qquad e=1,\ldots,E,
$$

where $e$ indexes parallel environment slots. The assignment should be balanced
so every train object receives approximately the same number of slots.

If per-world variants fail for some current objects, fallback to multiple mjlab
env instances grouped by object or by compatible object families. That is less
efficient but still gives within-object GPU batching.

## 6. Required BRC Changes

The current replay buffer cannot fully exploit multiple environment slots per
object without modification. It assumes one rollout stream per object. For mjlab
we should replace it with an object-aware slot buffer:

- store transitions from all environment slots;
- store `object_ids` for every transition;
- sample uniformly by object first, then by transition, to preserve balanced
  multi-object learning;
- pass `task_ids=object_ids` through the existing BRC interface. When
  `multitask=False`, the critic ignores those ids for learned embeddings, while
  `RewardNormalizer` still uses them for per-object reward scaling;
- keep geometry-conditioned modes with `multitask=False`, as they do now.

The reward normalizer should normalize by object id, not by environment slot id.
The statistics recorder should report both aggregate and per-object metrics.

The JAX/PyTorch boundary must be profiled. The simplest adapter can convert
PyTorch tensors to NumPy/JAX arrays each step, but that may lose much of the GPU
benefit. A better path is to use GPU tensor exchange, preferably DLPack, between
Torch and JAX for observations, actions, rewards, and conditioner inputs.

## 7. Conditioning Changes

The current online conditioning code assumes `env.envs` is a Python list of live
Gymnasium environments. That will not work with mjlab.

The mjlab adapter should provide batched geometry tensors:

- `wrist_raycast`: batched ray hit positions and valid-hit masks from mjlab
  raycast sensors, using this repository's wrist-camera pose and ray settings;
- `mesh_shape`: static mesh PointNet embedding per object id;
- `mesh_pose`: static mesh PointNet embedding per object id plus current batched
  object pose from mjlab tensors.

The existing trained PointNet checkpoints are Flax/JAX. We have two options:

1. keep PointNet inference in JAX and transfer pointcloud tensors from Torch to
   JAX each step;
2. port the frozen PointNet inference to PyTorch and load equivalent parameters,
   then transfer only the final augmented observation to JAX.

The first option is fastest to validate. The second option is likely better if
raycast conditioning remains the dominant runtime cost.

## 8. Implementation Milestones

1. **Install and version-pin newest mjlab.** Use at least `v1.4.0`, because the
   manual reset path is required for correct BRC truncation targets.
2. **Minimal reset/truncation smoke test.** Build a tiny mjlab env with
   `auto_reset=False`, force one timeout, and verify:
   - `step()` returns `truncated=True`;
   - the returned observation is the timeout observation;
   - `reset(env_ids=...)` clears the pending manual reset state;
   - BRC's mask convention gives `mask=1` for the timeout.
3. **ShadowHand single-object parity prototype.** Port or wrap one current
   ShadowHand object into mjlab. Compare observation dimensions, action
   semantics, episode length, reward scale, success metric, and reset
   randomization against the current Gymnasium environment.
4. **Single-object batched training smoke test.** Train `none` mode with many
   parallel slots of one object. Confirm replay insertion, reward normalization,
   and evaluation are stable.
5. **Multi-object mesh-variant prototype.** Build a small set of compatible
   objects in one mjlab env using per-world mesh variants. Verify object-id
   assignment, balanced sampling, per-object logging, and held-out-object
   evaluation.
6. **Conditioner port.** Add batched `mesh_shape`, `mesh_pose`, and
   `wrist_raycast` extraction from mjlab state/sensors. Confirm the same
   checkpoints can run online.
7. **Throughput benchmark.** Compare CPU Gymnasium `ParallelEnv` against mjlab
   for `none`, `mesh_pose`, and `wrist_raycast` using the same number of
   environment transitions and updates.
8. **Full plan update.** If the above benchmarks show a clear speedup without
   semantic drift, update `plan.md` and make mjlab the default path for large
   runs.

## 9. Immediate Recommendation

Proceed with a small mjlab adapter prototype before launching more full
85-object CPU training. The first prototype should not try to solve the full
multi-object problem. It should prove the reset/truncation contract and one
batched ShadowHand object under `auto_reset=False`.

The decision criterion is:

- if the one-object mjlab path preserves observations, rewards, resets,
  success, and timeout masks, continue to multi-object variants;
- if the adapter cannot preserve timeout next observations or requires changing
  BRC's mask semantics, do not use it for BRC without an explicit algorithmic
  change;
- if PyTorch-to-JAX transfer dominates runtime, test DLPack or a Torch-side
  frozen conditioner before porting all objects.

## 10. Evidence Checked

- Current BRC loop: `train.py`, `jaxrl/envs.py`, `jaxrl/replay_buffer.py`,
  `jaxrl/normalizer.py`, `jaxrl/agent/update.py`.
- Current runtime estimate: `agent_reports/section7_85object_scaling_20260522.md`.
- Existing mjlab reference project:
  `/storage/home/hcoda1/9/arosemberg3/scratch/in-hand-rotation-mjlab`.
- Existing mjlab multi-object reference:
  `src/in_hand_rotation_mjlab/tasks/hand_cube/hand_cube_env_cfg.py`.
- Existing mjlab raycast/PointNet reference:
  `src/in_hand_rotation_mjlab/tasks/hand_cube/mdp/observations.py`.
- mjlab `v1.4.0` README:
  `https://github.com/mujocolab/mjlab/tree/v1.4.0`.
- mjlab `v1.4.0` reset and timeout source:
  `https://raw.githubusercontent.com/mujocolab/mjlab/v1.4.0/src/mjlab/envs/manager_based_rl_env.py`.
- mjlab `v1.4.0` termination source:
  `https://raw.githubusercontent.com/mujocolab/mjlab/v1.4.0/src/mjlab/managers/termination_manager.py`.
- mjlab `v1.4.0` per-world variant source:
  `https://raw.githubusercontent.com/mujocolab/mjlab/v1.4.0/src/mjlab/entity/variants.py`.
