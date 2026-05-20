# Geometry-Conditioned BRC for Multi-Object Dexterous Rotation

Date: 2026-05-19

## 1. Purpose

This project will adapt the BRC reinforcement-learning package to train dexterous
manipulation teacher policies that can condition on object geometry instead of
categorical task identity. The immediate goal is not policy distillation. The
immediate goal is to determine which object embedding is suitable for training
multi-object and single-object dexterous rotation policies, with special attention
to generalization to objects that were not used for policy training.

The current BRC implementation supports multi-task learning by assigning each task
an integer task id and learning a normalized embedding vector for that id. This is
useful when the evaluation tasks are known during training, but it cannot support
the intended setting: evaluating policies on unseen objects. For unseen objects,
the conditioning signal must be computed from the object's shape and, where the
information contract permits it, its current pose. It must not be a learned lookup
table indexed by object name.

We will compare two geometry-conditioned families:

1. A wrist-raycast state-and-shape embedding. This is the preferred deployable
   candidate because its runtime input is a plausible wrist-camera pointcloud.
2. A mesh-based embedding. The minimal version is shape-only. A stronger version
   also includes object position and orientation, because dexterous control
   depends on the current object state, not only on static morphology.

The first major experiment is a multi-object policy comparison. Policies will be
trained on the repository's existing ShadowHand train split and evaluated on both
train objects and the existing held-out test split. The second major experiment is
a single-object comparison on seven selected objects from the multi-object train
split. Each phase will train and evaluate both embedding families.

## 2. Starting Point in This Repository

The repository contains the BRC algorithm and several environment families. The
relevant files for dexterous manipulation are:

- `train.py`: main BRC training loop.
- `jaxrl/agent/brc_learner.py`: agent construction, action sampling, and update
  orchestration.
- `jaxrl/networks.py`: actor, critic, residual MLP blocks, categorical task
  embedding, and categorical value output.
- `jaxrl/envs.py`: environment creation, ShadowHand observation flattening, and
  multi-task stepping through `ParallelEnv`.
- `jaxrl/replay_buffer.py`: replay storage with a separate first dimension for
  tasks.
- `dex_envs/`: ShadowHand rotation environments and assets.

The current ShadowHand path has these concrete properties:

- `dex_envs/assets/hand/` contains 114 `manipulate_*.xml` object environments.
- `dex_envs/configs.py` defines `MESH_NAMES` with 114 object names.
- The package already has an 85-object training list and a 29-object test list.
  This existing split will be used as the canonical split for this project.
- `dex_envs/__init__.py` registers version `v1` environments, with z-axis target
  rotation, sparse reward, randomized initial position, randomized initial
  rotation, and `max_episode_steps=100`.
- The ShadowHand wrapper in `jaxrl/envs.py` concatenates the Gymnasium Robotics
  `observation` vector with `desired_goal`.
- Existing ShadowHand result arrays are present under `results/shadowhand/`:
  85 single-task arrays and 86 multi-task arrays, where the multi-task directory
  includes `aggregate.npy`.

The current BRC multi-task conditioning is categorical:

- `TaskEmbedding(num_tasks, embedding_size)` creates a trainable embedding table.
- Each task id is mapped to an embedding vector and normalized by its Euclidean
  norm.
- The actor input is the observation concatenated with the task embedding.
- The critic input is observation, action, and the same task embedding.
- The replay buffer stores `task_ids`; sampling remains balanced over tasks.

This categorical mechanism should remain available as a diagnostic baseline, but
it is not an acceptable final conditioner for unseen-object evaluation.

The dependency file is known to be incomplete. It includes JAX, Flax, Gymnasium,
dm-control, Shimmy, Distrax, W&B, NumPy, and SciPy, but it does not list every
dependency needed by the ShadowHand path. Installation and smoke testing are the
first engineering task after these planning documents are created.

## 3. Scientific Question

Let $m$ denote a manipulated object, $s_t$ the full environment state at time
$t$, $o_t$ the policy observation, $a_t$ the action, and $c_t$ a geometry
conditioning vector. We need to learn a policy
$$
\pi(a_t \mid o_t, c_t)
$$
that works across a training set of objects and transfers to held-out objects.

For dynamic geometry observations, the conditioner must be online:
$$
c_t = E_\theta(x_t),
$$
where $x_t$ is the current geometry input available at policy time. For a
wrist-raycast policy, $x_t$ is the current visible pointcloud and its valid-hit
mask. For mesh-pose, $x_t$ is the current pose-aware mesh representation.
Shape-only mesh is static: its input is the canonical mesh, so the same encoder
can compute one fixed vector $c(m)$ per object without using object identity as
the conditioner.

Cached per-object wrist-raycast embeddings are not part of the main method. They
would replace the current sensor observation with an object-level summary and
would not test the intended state-and-shape conditioner.

The central question is:

Can a geometry-derived conditioning vector replace categorical task identity
without losing the ability to learn strong dexterous rotation policies?

The answer must be based on policy performance, not only on embedding pretraining
loss. Offline embedding metrics are useful gates, but the decisive measurements
are per-object success and return under the same BRC training and evaluation
protocol.

## 4. Object Split

The object universe for the first experiments is the 114-object ShadowHand set.
This repository already defines a canonical split:

- 85 train objects.
- 29 held-out test objects.

We will use that split rather than creating a new split. The split should
still be exported to a machine-readable manifest before training so that every
embedding checkpoint, policy run, and report can reference the exact object lists
used.

Embedding pretraining must obey the same split unless a run is explicitly marked
as a transductive or upper-bound diagnostic. For the main experiment:

- no policy training on held-out objects;
- no embedding-gradient updates using held-out objects;
- held-out meshes or held-out raycast observations may be used only for evaluation;
- no categorical id or object-name lookup is allowed in the deployable policy.

## 5. Wrist-Raycast State-and-Shape Embedding

### 5.1 Information Contract

The wrist-raycast state-and-shape embedding is the realistic sensor-based
embedding. At runtime it receives a pointcloud generated by a wrist/palm-mounted
raycast sensor in a plausible camera position. The intended sensor is mounted near
the wrist or back of the hand and tilted toward the object in the palm. The
ShadowHand-calibrated camera model uses these concrete values:

- parent body: `robot0:palm`;
- site name: `pointnet_camera_site`;
- camera name: `pointnet_camera`;
- position in parent frame: `(0.0024, -0.2019, -0.0613)`;
- quaternion in parent frame: `(0.466397, 0.884213, 0.016008, -0.019628)`;
- field of view: 40 deg;
- default ray grid for policy conditioning: 32 by 32 rays, giving 1024 points;
- max ray distance: 0.34 m.

This pose was chosen for the current ShadowHand assets because it places the
camera above the wrist/palm and points it toward objects held by the fingers. A
hand-specific calibration is necessary: a plausible pose for one hand model can
be inside the forearm or miss the grasped object on another hand model.

The raycast settings were selected by a sensor sweep over 13 train-split objects
and five random initializations per object. The selected `32x32`, 40 deg setting
averaged 462.7 valid hits and 220.4 object-geometry hits from 1024 rays, compared
with 59.2 valid hits and 28.2 object hits for the initial `16x16`, 60 deg smoke
setting. The tracked analysis figures and summary are in
`docs/raycast_sensor_sweep/`.

The pointcloud is expressed in the palm frame. If normals are available, each point
is represented as $(x_i, n_i)$, where $x_i \in \mathbb{R}^3$ is position and
$n_i \in \mathbb{R}^3$ is a surface normal. Points are scaled by a fixed length
scale, typically 0.34 m.

This sensor is honest in the intended sense: the policy is not handed the object
name, a categorical id, the canonical mesh, or the simulator's object pose. The
encoder must infer useful information from the visible partial pointcloud. In
practice, raycast hits can include hand geometry as well as object geometry, so
the encoder must either separate the object signal or learn to use the hand
configuration as context.

### 5.2 Mathematical Definition

The ray grid has a fixed size, but some rays miss all geometry. Let
$$
P_t = \{(x_i, n_i, \delta_i)\}_{i=1}^{N}
$$
be the wrist-raycast tensor at time $t$, transformed into palm frame and scaled
by $s_x$. Here $x_i \in \mathbb{R}^3$ is the hit point, $n_i \in \mathbb{R}^3$
is an optional normal, and $\delta_i \in \{0,1\}$ marks whether ray $i$ hit valid
geometry.

A single-stream PointNet encoder computes per-ray features
$$
g_i = f_\theta([x_i / s_x, n_i]),
$$
then pools only valid hits:
$$
h_t = T_\theta(P_t)
    = \max_{i:\delta_i=1} g_i,
$$
where the max is channelwise. In implementation, invalid rays are assigned
$-\infty$ before max pooling. This keeps the tensor shape fixed while preventing
missed rays from acting like real surface points.

The encoder adds structured prediction heads:
$$
\hat p_t = W_p h_t + b_p \in \mathbb{R}^3,
$$
$$
u_t = W_r h_t + b_r \in \mathbb{R}^6,
$$
$$
\hat R_t = \Pi_{SO(3)}(u_t),
$$
$$
z_t = W_z h_t + b_z \in \mathbb{R}^{d_z},
$$
with an optional visibility or confidence head
$$
\hat v_t \in [0, 1].
$$

The map $\Pi_{SO(3)}$ converts a 6D rotation representation into a rotation
matrix. If $u_t = [a_1, a_2]$, with $a_1,a_2 \in \mathbb{R}^3$, then
$$
b_1 = \frac{a_1}{\|a_1\|_2},
$$
$$
b_2 = \frac{a_2 - (b_1^\top a_2)b_1}
           {\|a_2 - (b_1^\top a_2)b_1\|_2},
$$
$$
b_3 = b_1 \times b_2,
$$
and
$$
\hat R_t = [b_1\ b_2\ b_3].
$$

The embedding supplied to the policy can be either the trunk feature $h_t$, the
structured vector $[\hat p_t, u_t, z_t, \hat v_t]$, or both. The first
implementation should record this choice explicitly in the checkpoint metadata and
run config. Using $h_t$ matches the common PointNet-conditioning pattern; exposing
the structured heads makes debugging easier.

### 5.3 Training Objective

The wrist-raycast encoder is trained with a structured state-and-shape loss:
$$
\mathcal{L}_{ray} =
\lambda_{pos}\mathcal{L}_{pos}
+ \lambda_{rot}\mathcal{L}_{rot}
+ \lambda_{shape}\mathcal{L}_{shape}
+ \lambda_{cons}\mathcal{L}_{cons}
+ \lambda_{vis}\mathcal{L}_{vis}.
$$

The position loss is robust palm-frame regression:
$$
\mathcal{L}_{pos}
= \operatorname{Huber}\left(\frac{\hat p_t - p_t^\star}{s_p}\right),
$$
where $p_t^\star$ is the simulator ground-truth object center in palm frame and
$s_p$ is a normalization scale.

The rotation loss is symmetry-aware geodesic error:
$$
\mathcal{L}_{rot}
= \min_{S \in \mathcal{G}(m)}
d_{geo}(\hat R_t, R_t^\star S),
$$
where $R_t^\star$ is the ground-truth palm-frame object orientation and
$\mathcal{G}(m)$ is the set of orientation symmetries treated as equivalent for
object $m$. The geodesic distance is
$$
d_{geo}(R_1, R_2)
=
\arccos
\left(
\operatorname{clip}
\left(
\frac{\operatorname{tr}(R_1^\top R_2)-1}{2},
-1,
1
\right)
\right).
$$
For spherical or nearly orientation-ambiguous objects, this loss should be masked
or downweighted rather than forcing arbitrary orientation labels.

The morphology code is normalized,
$$
\tilde z_t = \frac{z_t}{\|z_t\|_2},
$$
and trained with supervised contrastive learning over object identity during
embedding training only:
$$
\mathcal{L}_{shape}
=
\sum_i
-\frac{1}{|P(i)|}
\sum_{j \in P(i)}
\log
\frac{\exp(\tilde z_i^\top \tilde z_j / \tau)}
{\sum_{k \ne i}\exp(\tilde z_i^\top \tilde z_k / \tau)}.
$$
Here $P(i)$ is the set of other samples of the same object in the batch. Object
identity is allowed as a training label for the encoder, but it is not a runtime
input to the policy.

The consistency loss compares predictions under light perturbations of the same
sensor observation:
$$
\mathcal{L}_{cons}
=
\|\hat p_t^{(1)} - \hat p_t^{(2)}\|_1
+ d_{geo}(\hat R_t^{(1)}, \hat R_t^{(2)})
+ \|z_t^{(1)} - z_t^{(2)}\|_2^2.
$$

The visibility loss is optional:
$$
\mathcal{L}_{vis} = (\hat v_t - v_t^\star)^2,
$$
where $v_t^\star$ should be an object-observability target, such as an object-hit
fraction if the simulator can provide it.

### 5.4 Existing Evidence for the Wrist-Raycast Embedding

Preliminary sensor studies show that the wrist raycast carries meaningful but
partial information:

| Object | Hit fraction mean +/- std | Cross-reset Jaccard | Single-frame unique voxels | 20-frame unique voxels | 20-frame gain |
|---|---:|---:|---:|---:|---:|
| orange | 30.5% +/- 0.4% | 0.221 +/- 0.046 | 3,911 | 5,992 | 1.53x |
| hammer | 29.2% +/- 0.6% | 0.147 +/- 0.056 | 3,950 | 5,696 | 1.44x |
| d_marbles | 25.7% +/- 0.1% | 0.280 +/- 0.053 | 3,367 | 5,291 | 1.57x |

The interpretation is:

- orange is likely feasible from a single frame;
- hammer provides strong asymmetric visual evidence but is sensitive to viewpoint;
- d_marbles is compact and relatively ambiguous, so temporal history may help.

A three-object wrist-raycast state-and-shape offline study produced these final
metrics:

| Condition | Object | Position error | Rotation error |
|---|---|---:|---:|
| shared small | orange | 3.76 cm | not applicable |
| shared small | hammer | 3.05 cm | 17.3 deg |
| shared small | d_marbles | 3.20 cm | 35.4 deg |
| shared medium | orange | 13.52 cm | not applicable |
| shared medium | hammer | 14.45 cm | 12.2 deg |
| shared medium | d_marbles | 12.68 cm | 21.9 deg |
| shared large | orange | 5.37 cm | not applicable |
| shared large | hammer | 7.29 cm | 13.2 deg |
| shared large | d_marbles | 5.26 cm | 23.9 deg |
| per-object hammer small | hammer | 2.53 cm | 6.0 deg |
| per-object hammer medium | hammer | 5.64 cm | 5.3 deg |
| per-object orange small | orange | 1.65 cm | not applicable |
| per-object d_marbles small | d_marbles | 3.98 cm | 11.8 deg |

An all-object shared-medium wrist-raycast state-and-shape run over 114 objects
improved position quality but did not solve rotation prediction:

- mean position error over all 114 objects: 0.046430 m;
- minimum position error: 0.026803 m;
- maximum position error: 0.070733 m;
- mean rotation error: 88.68 deg;
- on a 13-object representative subset, mean position error was 0.045171 m and
  mean rotation error was 90.02 deg.

These numbers imply that the embedding can carry useful object and position
information, while explicit rotation prediction remains difficult at this scale.
The BRC policy experiments are therefore necessary: RL performance may depend more
on usable geometry and state cues than on the encoder's standalone rotation error.

## 6. Mesh-Based Embedding

### 6.1 Information Contract

The mesh-based embedding uses the object mesh rather than a wrist raycast. It must
still avoid categorical identity. A held-out object should be conditioned by
running the same mesh encoder on its mesh, not by assigning a new learned object id.

There are two variants:

1. **Mesh shape-only embedding.** The input is the canonical object mesh. The output
   is a static shape vector $z_m$. This is a clean test of whether morphology
   alone is enough to condition a multi-object policy when the base observation
   already contains whatever state the environment provides.
2. **Mesh shape-plus-pose embedding.** The input includes both mesh shape and the
   current object pose relative to the palm. This is the stronger control-state
   version. If it uses simulator pose directly, it is privileged and should be
   interpreted as an upper bound unless an equivalent real perception pipeline is
   available.

### 6.2 Mathematical Definition

Let the canonical mesh for object $m$ be
$$
M_m = (V_m, F_m),
$$
where $V_m$ are vertices and $F_m$ are faces. Sample $N$ surface points and
normals:
$$
Q_m = \{(q_i, n_i)\}_{i=1}^{N}.
$$

For shape-only conditioning:
$$
z_m = \phi_\psi(Q_m),
$$
where $\phi_\psi$ is a PointNet-style mesh encoder. The policy receives
$$
c_t = z_m.
$$
This vector is constant throughout an episode. It may be cached after applying
the encoder to the mesh because the mesh is static; it must not be replaced by a
learned object-id table.

For shape-plus-pose conditioning, let $p_t^m$ and $R_t^m$ be the object position
and orientation in palm frame. The direct structured version is
$$
c_t(m) = [z_m,\ p_t^m,\ \operatorname{rot6d}(R_t^m)].
$$
An alternative is to transform the sampled mesh into the current palm-frame pose,
$$
Q_{m,t}^{palm}
=
\{(R_t^m q_i + p_t^m,\ R_t^m n_i)\}_{i=1}^{N},
$$
and encode it with another PointNet:
$$
c_t(m) = \phi_\psi(Q_{m,t}^{palm}).
$$

The direct structured version is easier to audit. The transformed-pointcloud version
is closer in form to the wrist-raycast embedding but uses complete mesh
information instead of a partial sensor view. If simulator pose provides
$p_t^m$ or $R_t^m$, the result is a privileged upper bound unless an equivalent
real pose-estimation pipeline is supplied.

### 6.3 Why Pose Matters

Static shape alone cannot tell the policy where the object currently sits in the
hand or how it is oriented. If the flattened ShadowHand observation already contains
reliable object pose, then shape-only mesh conditioning may be enough. If not,
shape-only conditioning will be under-specified for control. Therefore the mesh
study should include:

- a shape-only condition, because it matches the developer's original suggestion;
- a shape-plus-pose condition, because it tests the best geometry-state signal;
- explicit labeling of any pose-based mesh result as privileged if simulator pose
  is used directly.

## 7. BRC Adaptation Strategy

The minimal change is to keep BRC's replay, reward normalization, evaluation, and
multi-task stepping structure, while replacing categorical task embeddings with
geometry embeddings when requested.

The proposed conditioning modes are:

- `categorical`: current BRC behavior; task id lookup table is active.
- `none`: no task conditioner; useful for ablations.
- `wrist_raycast`: append a wrist-raycast geometry embedding.
- `mesh_shape`: append a mesh shape embedding.
- `mesh_pose`: append a mesh shape-plus-pose embedding.

For geometry-conditioned multi-object training:

- keep `task_ids` in the replay buffer for balanced sampling, per-object logging,
  and reward normalization;
- disable the learned categorical task embedding in actor and critic inputs;
- append the external geometry embedding to the observation or pass it through a
  clearly named conditioner path;
- assert at runtime that geometry-conditioned runs are not also using learned task
  lookup embeddings unless the run is explicitly marked as a hybrid diagnostic.

This preserves the BRC training style while changing only the information used to
condition the policy.

## 8. Experiment Plan

### Phase 1: Installation and Smoke Testing

Install and test the repository. The expected issues
are missing dependencies and version mismatches. The minimum smoke tests are:

1. import the package modules used by `train.py`;
2. instantiate one ShadowHand environment, such as `orange-rotate-v1`;
3. reset and step the environment with random actions;
4. instantiate `ParallelEnv` with two objects;
5. run a tiny BRC loop long enough to verify replay insertion, sampling, update,
   and evaluation code paths.

No scientific claims should be made from smoke tests.

### Phase 2: Split Manifest

Export the repository's existing 85/29 ShadowHand split to a deterministic
machine-readable manifest. Save:

- train object names;
- held-out object names;
- object source list hash or exact object source list;
- source symbols or files used to derive the split;
- date and authoring note.

All subsequent embedding checkpoints and policy runs must record the split manifest
they used.

### Phase 3: Embedding Infrastructure

Implement a geometry-conditioning layer with a small number of explicit modes. The
implementation should be compatible with the existing BRC code style:

- avoid parallel ad hoc training loops when `train.py` can be extended cleanly;
- keep task ids for logging and balanced replay;
- avoid hidden object-name lookup in the policy path;
- save conditioner metadata with every policy run.

The first implementation slice should establish the shared BRC conditioning
plumbing before adding the full learned encoders. In particular, it should:

- preserve categorical behavior as the default diagnostic baseline;
- add `none` as an ablation mode;
- add an initial `mesh_shape` mode with deterministic mesh features, so the
  policy path can be tested before a learned mesh encoder exists;
- require the canonical split manifest for geometry-conditioned runs that need
  train-split-only normalization;
- keep multi-object batch sizing tied to the number of tasks rather than to the
  presence of a learned categorical embedding.

The first implementation slice is not complete until smoke tests verify action
sampling, replay-buffer updates, diagnostics, and one-step `train.py` execution
for the new conditioning path.

The remaining infrastructure should include object metadata utilities:

- object name to XML path;
- object name to mesh path;
- mesh sampling and normalization;
- optional current pose extraction from the environment;
- raycast pointcloud extraction or generation.

### Phase 4: Wrist-Raycast Sensor Integration and Visual Validation

Implement the wrist-raycast path before launching geometry-conditioned policy
experiments. This phase is required because the wrist-raycast embedding is one of
the two main methods under test and the preferred deployable candidate.

The raycast path needs special attention because Gymnasium Robotics environments
may not expose a ready-made raycast sensor interface for the current ShadowHand
wrapper. If the current XMLs do not include the needed camera, site, or sensor
support, add the minimal MuJoCo XML and wrapper support required to generate the
wrist-raycast pointcloud.

This phase should deliver:

- a documented wrist/palm-mounted camera or site using the pose in Section 5.1
  unless a measured repository-specific correction is justified;
- a deterministic 2-D ray grid with max distance 0.34 m;
- pointcloud output in a clearly documented coordinate frame, with a transform to
  the palm frame for policy conditioning;
- hit masks or equivalent metadata distinguishing valid ray hits from misses;
- tests covering output shape, finite values, frame conventions, deterministic
  reset behavior, and at least three different objects;
- a `wrist_raycast` conditioning-mode scaffold only after the raw pointcloud path
  has been validated.

As a visual sanity check, this phase must generate diagnostic figures for three
randomly selected train-split objects, recording the random seed and object names.
Each object must have one figure with three views:

1. a bird's-eye rendered view of the full hand-object scene;
2. the wrist-camera RGB view used to define the ray grid;
3. the raycast pointcloud of the camera-visible geometry, rendered from the same
   bird's-eye viewpoint as the scene view.

The pointcloud panel should report hit count and total ray count, color points by
distance to the wrist camera when useful, and use axis labels that state the
coordinate frame. These figures are not a quantitative result; they are a guard
against wrong camera placement, flipped frames, empty raycasts, and accidental use
of the wrong geometry.

### Phase 5: Online Conditioner Interface

Before training the learned encoders, define and test the online conditioner
interface that BRC will use. This phase prevents the learned embedding work from
quietly drifting into cached object-level features.

The key contract is:

$$
c_t = E_\theta(x_t),
$$

where $x_t$ is the current geometry input at policy time. For the wrist-raycast
conditioner, $x_t$ is the current raycast pointcloud and hit mask. For mesh-pose,
$x_t$ is the current transformed mesh representation. Shape-only mesh is the only
main exception: because the canonical object mesh is static during an episode, its
shape embedding can be computed once per object by applying the same encoder to
that object's mesh.

This phase should:

- define the raycast point tensor, valid-hit mask, coordinate frame, normal and
  distance channels, and normalization constants;
- implement mask-aware PointNet-style pooling so invalid ray slots do not behave
  as real surface points;
- add a `wrist_raycast` BRC conditioning mode that computes features from the
  current raycast observation online;
- explicitly forbid cached per-object wrist-raycast embeddings for the main
  deployable method;
- define the mesh conditioner interface for both shape-only and mesh-pose modes;
- decide what information mesh-pose may use, and label it privileged if it depends
  on simulator object pose;
- add one-step smoke tests proving that BRC can initialize and step with
  `wrist_raycast` without learned categorical task embeddings.

### Phase 6: PointNet Training in This Repository

Add a path to train geometry encoders and feed their fixed-size outputs into BRC
inside this repository. This phase trains two learned encoders:

1. a wrist-raycast PointNet that maps current wrist pointcloud observations to
   online policy-conditioning vectors;
2. a mesh PointNet that maps sampled mesh geometry to shape embeddings, and whose
   pose-conditioned variant is specified by the mesh-pose decision from Phase 5.

Wrist-raycast PointNet training should:

- collect wrist-raycast pointclouds from train-split objects only;
- train state and shape heads with the structured raycast loss;
- save both the feature encoder and the full supervised checkpoint;
- evaluate offline metrics on train and held-out objects separately;
- record whether the policy will consume trunk features, structured heads, or both;
- define the exact point tensor shape, mask convention, coordinate frame, and
  normalization constants used by the encoder;
- load trained encoder parameters into the online `wrist_raycast` conditioning
  path without reintroducing a learned object-id lookup or cached per-object
  raycast feature;
- save conditioner metadata with the BRC run: encoder checkpoint path, manifest
  path, pointcloud configuration, feature dimension, and normalization statistics;
- run a one-step BRC smoke test for `wrist_raycast`, analogous to the
  `mesh_shape` smoke test, before any long training run.

Mesh PointNet training should:

- sample object meshes consistently;
- train a shape encoder without object-name lookup at inference;
- train the shape-only mesh encoder as a separate PointNet from the wrist-raycast
  encoder;
- decide and implement the mesh-pose encoding: either append pose variables to
  the mesh-shape embedding, or encode mesh points transformed into the current
  palm/object pose with a PointNet;
- explicitly mark mesh-pose results as privileged if simulator object pose is
  used;
- evaluate train and held-out mesh embeddings without policy training on held-out
  objects.

Because this repository is JAX/Flax-based, the default implementation should prefer
Flax/JAX unless dependency or MuJoCo integration constraints make a small additional
dependency clearly better.

### Phase 7: Multi-Object BRC Policy Training

Train and evaluate the following policies on the existing 85/29 split:

| Run family | Train objects | Held-out evaluation | Conditioner | Interpretation |
|---|---:|---:|---|---|
| categorical BRC | 85 | no valid unseen-object test | learned task id | train-object diagnostic upper bound |
| no-conditioner BRC | 85 | yes | none | ablation |
| wrist-raycast BRC | 85 | yes | wrist raycast | primary deployable candidate |
| mesh shape BRC | 85 | yes | mesh shape | shape-only geometry baseline |
| mesh pose BRC | 85 | yes | mesh plus pose | privileged or state-rich upper bound |

The initial BRC hyperparameters should start from the repository defaults:

- `max_steps=1000000`;
- `start_training=5000`;
- `eval_interval=50000`;
- `eval_episodes=10`;
- `updates_per_step=2`;
- `width_critic=4096`;
- multi-task batch size 1024.

These defaults can be changed only after smoke testing and a documented reason.
Any change must be shared across embedding families unless the run is explicitly a
tuning run.

Evaluation must report, at minimum:

- per-object success;
- per-object return;
- train-object aggregate mean, median, and bottom quartile;
- held-out aggregate mean, median, and bottom quartile;
- train-to-heldout generalization gap;
- wall-clock time and environment steps;
- exact object split and conditioner checkpoint.

### Phase 8: Single-Object BRC Policy Training

After the multi-object runs, select seven objects from the 85-object train split.
The selection should cover easy, medium, and hard cases and should be recorded
before launching the single-object comparison. A provisional candidate set, if all
are in the train split, is:

- `orange`;
- `tuna_fish_can`;
- `water_bottle`;
- `a_cups`;
- `g_lego_duplo`;
- `d_marbles`;
- one hard asymmetric object such as `hammer` or `flat_screwdriver`.

For each selected object, train and evaluate:

- wrist-raycast conditioned single-object policy;
- mesh-conditioned single-object policy.

The single-object study answers whether poor multi-object performance is caused by
the embedding itself, shared-policy interference, or object-specific difficulty.

## 9. Decision Criteria

The wrist-raycast state-and-shape embedding is the preferred winner if it
satisfies all of the following:

- it trains reliably on the 85-object multi-object setting;
- held-out success is meaningfully above the no-conditioner ablation;
- the train-to-heldout gap is not dominated by a few object families;
- it does not use object identity, mesh templates, or simulator pose at inference;
- single-object results do not reveal that the embedding is unusable on selected
  hard objects.

The mesh shape embedding is a strong alternative if:

- it substantially improves held-out performance over no-conditioner and the
  wrist-raycast embedding;
- its shape-only information contract is acceptable for the intended deployment;
- it generalizes by applying the same encoder to unseen meshes.

The mesh pose embedding should be interpreted carefully. If it wins but the
wrist-raycast embedding fails, the likely conclusion is that the policy benefits
from object geometry and pose, but the realistic wrist-raycast perception stack is
the bottleneck.

## 10. Known Risks

1. **Incomplete dependencies.** The repository requirements are incomplete. All
   dependency additions should be minimal and pinned when possible.
2. **Hidden categorical leakage.** Geometry-conditioned runs must not silently keep
   the learned task embedding active.
3. **Held-out leakage.** The 29 held-out objects must not be used for policy
   training or main embedding training.
4. **Privileged mesh interpretation.** Mesh-pose results must not be presented as
   deployable sensor results unless the same information can be obtained at runtime.
5. **Raycast integration.** The wrist-raycast embedding may require MuJoCo XML or
   wrapper changes because the current ShadowHand wrapper only flattens existing
   Gym observations.
6. **Sequential multi-task stepping.** `ParallelEnv` loops through one environment
   per task in Python. Training 85 objects may be slow. First preserve the existing
   style, then optimize only if profiling shows it is necessary.
7. **Offline metrics can mislead.** Low wrist-raycast position or rotation loss is
   not sufficient evidence. The deciding metric is policy success and return.
8. **Evaluation variance.** Sparse-reward dexterous manipulation can be noisy.
   Important conclusions should use multiple seeds or enough evaluation episodes
   to separate real effects from noise.

## 11. Immediate Next Actions

1. Treat Phases 1 and 2 as complete after verification of the working environment
   and the canonical 85/29 split manifest.
2. Treat the first slice of Phase 3 as complete after committing the
   geometry-conditioning switch, the `none` ablation, the initial `mesh_shape`
   path, manifest-enforced train-split normalization, and smoke tests.
3. Treat Phase 4 as complete after committing the wrist-raycast pointcloud path,
   ShadowHand camera/site support, smoke tests, and three-object visual sanity
   check.
4. Start Phase 5 next: define and smoke-test the online conditioner interface for
   `wrist_raycast`, mesh shape, and mesh pose. Do not use cached per-object
   wrist-raycast embeddings as the main method.
5. In Phase 6, train the wrist-raycast PointNet and mesh PointNet, and implement
   the chosen mesh-pose representation.
6. Launch the first small two- or three-object geometry-conditioned BRC pilot only
   after both embedding families have validated conditioning paths.
