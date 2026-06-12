# Geometry-Conditioned BRC for Dexterous Object Rotation

Date: 2026-05-21

## 1. Goal

This project adapts the BRC reinforcement-learning codebase to train
multi-object ShadowHand rotation policies conditioned on object geometry. The
scientific question is whether geometry-derived conditioning can replace a
learned categorical task identity while preserving policy learning on training
objects and enabling evaluation on held-out objects.

The immediate goal is teacher-policy training and evaluation, not distillation.
We compare two geometry families:

- **Wrist-raycast conditioning**: an online partial pointcloud from a
  palm/wrist-mounted raycast camera. This is the primary deployable candidate.
- **Mesh conditioning**: a PointNet embedding of the canonical object mesh, with
  either shape only or shape plus simulator object pose. Shape-only mesh is a
  geometry baseline; mesh plus pose is a privileged upper bound unless equivalent
  pose estimates are available at runtime.

All main experiments use the existing ShadowHand split: 85 train objects and 29
held-out objects, saved in `manifests/shadowhand_split_v1.json`.

## 2. Notation and Policy Contract

Let:

- $m$ be the manipulated object;
- $s_t$ be the full simulator state at time $t$;
- $o_t$ be the policy observation;
- $x_t$ be the geometry input available to the conditioner;
- $a_t$ be the action;
- $c_t$ be the conditioning vector supplied to the policy.

The geometry-conditioned control pipeline is

$$
o_t = \Omega(s_t),
$$

$$
x_t = \Gamma(s_t,m),
$$

$$
c_t = E_\eta(x_t),
$$

$$
a_t \sim \pi_\phi(\cdot \mid o_t,c_t).
$$

Here $\Omega$ is the environment's observation function, $\Gamma$ is the
geometry-extraction function, $E_\eta$ is the mode-specific geometry
conditioner, and $\pi_\phi$ is the BRC policy. The parameter symbol $\eta$ is
generic: the wrist-raycast encoder uses parameters $\theta$, while the mesh
encoder uses parameters $\psi$. The important contract is that $c_t$ must come
from geometry through $E_\eta(x_t)$, not from a learned object-id table.

The three geometry inputs used in this project are

$$
x_t^{\mathrm{ray}} = (P_t^{\mathrm{ray}},\delta_t^{\mathrm{ray}}),
$$

$$
x_m^{\mathrm{mesh}} = M_m,
$$

$$
x_t^{\mathrm{mesh+pose}} = (M_m,p_t^m,R_t^m).
$$

For wrist-raycast and mesh-pose, $x_t$ changes during the episode and must be
read online. For shape-only mesh, $x_m^{\mathrm{mesh}}$ is static, so

$$
c_t = z_m = E_\psi^{\mathrm{mesh}}(M_m)
$$

can be computed once per object after applying the same encoder to that object's
mesh. Task ids are still allowed internally for replay storage, balanced
sampling, reward normalization, and per-object logging. They must not be used as
learned policy inputs in the geometry-conditioned modes.

## 3. Conditioning Modes

The policy comparison uses five modes:

| Mode | Runtime conditioner | Use |
|---|---|---|
| `categorical` | learned task-id embedding | train-object diagnostic baseline |
| `none` | no conditioner | ablation |
| `wrist_raycast` | online wrist pointcloud encoder | primary deployable geometry candidate |
| `mesh_shape` | static learned mesh-shape vector | shape-only geometry baseline |
| `mesh_pose` | learned mesh-shape vector plus pose | privileged geometry-state upper bound |

Only `categorical` may use a learned task embedding. In all other modes,
actor and critic inputs must receive either no conditioner or a geometry-derived
conditioner.

Implementation note: the original BRC code uses a flag named `multitask` to
decide whether `build_actor_input()` appends a learned categorical task
embedding. In this project, `multitask=True` means "use the categorical
embedding path", not merely "there are multiple objects". Geometry modes set
that flag to `False`; their conditioning vectors are already concatenated into
the observation before the actor and critic update functions run.

## 4. Shared PointNet Block

This section defines the reusable PointNet block $\operatorname{PN}_\eta$ used
by both geometry families when their input is represented as a pointcloud. It is
not a separate conditioning mode. Sections 5 and 6 specify how raycast and mesh
geometry are converted into point tensors, then call this same block with
different parameters.

For a training or policy batch of pointcloud geometry samples,

$$
P \in \mathbb{R}^{B \times N \times d},
\qquad
\delta \in \{0,1\}^{B \times N},
\qquad
h = \operatorname{PN}_\eta(P,\delta)\in\mathbb{R}^{B\times64},
$$

where

- $B$ is the number of samples in the batch;
- $N$ is the number of point slots;
- $d$ is the number of scalar channels stored per point;
- $b\in\{1,\ldots,B\}$ indexes samples in the batch;
- $i\in\{1,\ldots,N\}$ indexes point slots;
- $P$ is the point tensor;
- $P_{b,i} \in \mathbb{R}^d$ is the vector stored in point slot $i$ of
  sample $b$;
- $\delta$ is the valid-point mask;
- $\delta_{b,i}=1$ means point slot $i$ of sample $b$ contains a real
  point;
- $\delta_{b,i}=0$ means that slot is padding or a missed ray.

For the current wrist-raycast and mesh checkpoints, $d=3$ and the channels are
the point-location coordinates $(x,y,z)$ after the coordinate-frame transform and
normalization described below. Other point attributes, such as surface normals,
ray distance, color, or segmentation logits, would increase $d$ by adding more
scalar channels per point. They are not used in the current checkpoints.

For wrist raycasts, $N=1024$ and invalid slots are rays that missed all geometry.
For mesh pointclouds, $N=1024$ and all $\delta_{b,i}=1$.

The PointNet applies the same multilayer perceptron to every point:

Let $H_0=d$ be the input width and $H_\ell$ the width of layer $\ell$. For the
current checkpoints, the hidden widths are $(H_1,H_2,H_3)=(64,128,256)$.

$$
u_{b,i}^{(0)} = P_{b,i} \in \mathbb{R}^{H_0},
$$

$$
u_{b,i}^{(\ell)}
= \sigma\!\left(W_\ell u_{b,i}^{(\ell-1)} + b_\ell\right),
\qquad \ell=1,\ldots,L,
$$

where $W_\ell\in\mathbb{R}^{H_\ell\times H_{\ell-1}}$,
$b_\ell\in\mathbb{R}^{H_\ell}$, $u_{b,i}^{(\ell)}\in\mathbb{R}^{H_\ell}$, and
$\sigma$ is ReLU. For the current checkpoints, $L=3$. The final per-point feature is

$$
g_{b,i} = u_{b,i}^{(L)} \in \mathbb{R}^{H},
\qquad H=H_L.
$$

The reduction from $N$ point features to one vector is a **masked channelwise
maximum over point slots within the same sample**. It is not a maximum over
training samples, environments, objects, or timesteps.

Let $k\in\{1,\ldots,H\}$ index feature channels. For each sample $b$ and
each feature channel $k$, define

$$
\bar g_{b,k}
=
\begin{cases}
\displaystyle
\max_{\substack{i\in\{1,\ldots,N\}\\ \delta_{b,i}=1}}
g_{b,i,k},
&
\text{if }
\sum_{i=1}^{N}\delta_{b,i} > 0,
\\[1.0em]
0,
&
\text{if }
\sum_{i=1}^{N}\delta_{b,i} = 0.
\end{cases}
$$

The pooled tensor is $\bar g\in\mathbb{R}^{B\times H}$, and row
$\bar g_b\in\mathbb{R}^{H}$ is the pooled feature for sample $b$.
This maximum is taken separately for each sample and each feature channel.
It does **not** select one single point as the representative pointcloud. Within
the same sample $b$, channel $1$ may be maximized by one valid point,
channel $2$ by another, and so on.

In code this is implemented as

$$
\tilde g_{b,i,k}
=
\begin{cases}
g_{b,i,k}, & \delta_{b,i}=1,\\
-\infty, & \delta_{b,i}=0,
\end{cases}
$$

where $\tilde g\in(\mathbb{R}\cup\{-\infty\})^{B\times N\times H}$, followed by

$$
\bar g_{b,k}
=
\max_{i\in\{1,\ldots,N\}}
\tilde g_{b,i,k}.
$$

Therefore missed rays cannot contribute zeros or other artificial values to the
pooled feature. If all $\delta_{b,i}=0$ for a given sample $b$, the row
$\bar g_b$ is explicitly set to zero. The batch index $b$ is never maximized
over. Each sample receives its own pooled vector $\bar g_b$ and its own
embedding.

The output feature for sample $b$ is

$$
h_b = W_h \bar g_b + b_h \in \mathbb{R}^{64}.
$$

Here $W_h\in\mathbb{R}^{64\times H}$ and $b_h\in\mathbb{R}^{64}$.
The parameter set $\eta$ consists of the pointwise MLP weights and biases
$(W_\ell,b_\ell)_{\ell=1}^L$ plus the output projection $(W_h,b_h)$.
The matrix $h\in\mathbb{R}^{B\times64}$ contains one row $h_b$ per sample. This
is the complete PointNet convention used below. The raycast and mesh sections do
not define new pointwise networks or pooling rules; they only define how their
geometry is written into $P$ and $\delta$ before applying
$\operatorname{PN}_\eta$.

## 5. Wrist-Raycast Conditioner

The wrist-raycast conditioner is the deployable perception candidate. Its
runtime input is a partial pointcloud from a palm-mounted ray grid, not object
name, object id, canonical mesh, or simulator pose.

The MuJoCo camera/site is mounted on `robot0:palm`:

- site: `pointnet_camera_site`;
- camera: `pointnet_camera`;
- position: `(0.0024, -0.2019, -0.0613)` in the palm frame;
- quaternion: `(0.466397, 0.884213, 0.016008, -0.019628)`;
- ray grid: $32 \times 32$;
- field of view: $40$ degrees;
- maximum ray distance: $0.34$ m;
- point frame: palm frame.

Let $\mathcal{W}$ denote the world frame and $\mathcal{P}$ the palm frame. At
time $t$:

- $p_{\mathcal{P},t}^{\mathcal{W}}\in\mathbb{R}^3$ is the palm origin in world
  coordinates;
- $R_{\mathcal{P},t}^{\mathcal{W}}\in SO(3)$ maps palm-frame vectors into the
  world frame;
- $q_t^{\mathcal{W}}\in\mathbb{R}^3$ is the ray sensor origin in world
  coordinates;
- $i\in\{1,\ldots,1024\}$ indexes one ray in the $32\times32$ grid;
- $d_{t,i}^{\mathcal{W}}\in\mathbb{R}^3$ is the world-frame unit direction of ray
  $i$;
- $r_{t,i}\in\mathbb{R}_{\ge 0}$ is the MuJoCo hit distance for ray $i$, defined
  only when the ray hits geometry.

If ray $i$ hits geometry within the maximum distance, its world-frame hit point is

$$
y_{t,i}^{\mathcal{W}}
= q_t^{\mathcal{W}} + r_{t,i} d_{t,i}^{\mathcal{W}}
\in\mathbb{R}^3.
$$

The same point in palm coordinates is

$$
x_{t,i}^{\mathcal{P}}
= (R_{\mathcal{P},t}^{\mathcal{W}})^\top
  (y_{t,i}^{\mathcal{W}} - p_{\mathcal{P},t}^{\mathcal{W}})
\in\mathbb{R}^3.
$$

This produces one unbatched raycast pointcloud at time $t$,
$P_t^{\mathrm{ray}}\in\mathbb{R}^{1024\times3}$, with rows

$$
P_{t,i}^{\mathrm{ray}} =
\begin{cases}
x_{t,i}^{\mathcal{P}} / s_x, & \delta_{t,i}^{\mathrm{ray}}=1,\\
(0,0,0), & \delta_{t,i}^{\mathrm{ray}}=0,
\end{cases}
\qquad
s_x = 0.34\text{ m},
$$

where $P_{t,i}^{\mathrm{ray}}\in\mathbb{R}^3$ and the mask
$\delta_t^{\mathrm{ray}}\in\{0,1\}^{1024}$ is

$$
\delta_{t,i}^{\mathrm{ray}} =
\begin{cases}
1, & \text{ray } i \text{ hit geometry within } 0.34\text{ m},\\
0, & \text{otherwise}.
\end{cases}
$$

The current point channels are only scaled xyz coordinates, so the channel
dimension is $d=3$. Surface normals or ray distances can be added later by
increasing $d$, but they are not used by the current checkpoint.

To apply the batched PointNet convention from Section 4, a policy or training
batch with samples from timesteps $t_1,\ldots,t_B$ is formed by setting

$$
P_{b,i} = P_{t_b,i}^{\mathrm{ray}},
\qquad
\delta_{b,i} = \delta_{t_b,i}^{\mathrm{ray}}.
$$

Let $\operatorname{PN}_\theta$ be the shared PointNet block from Section 4 with
raycast parameters $\theta$. The online raycast feature is

$$
h_t^{\mathrm{ray}}
= \operatorname{PN}_\theta(P_t^{\mathrm{ray}},\delta_t^{\mathrm{ray}})
\in \mathbb{R}^{64},
$$

where the single-timestep call means applying $\operatorname{PN}_\theta$ with a
batch dimension of one. The online raycast conditioner uses this feature
directly:

$$
c_t^{\mathrm{ray}}
= h_t^{\mathrm{ray}}
\in \mathbb{R}^{64}.
$$

Equivalently, the mode-specific conditioner from Section 2 is
$E_\theta^{\mathrm{ray}}(x_t^{\mathrm{ray}})=c_t^{\mathrm{ray}}$.

During BRC training and evaluation, the effective policy observation is

$$
\tilde o_t = [o_t,\ c_t]\in\mathbb{R}^{D_o+64},
$$

where $o_t\in\mathbb{R}^{D_o}$ is the raw BRC observation and $D_o$ is the raw
observation dimension for the active environment set.

The raycast embedding is recomputed from the live MuJoCo state at every policy
step. It is stored in replay together with the transition observation that used
it, but it is not cached as a per-object lookup table.

For the planned BRC policy comparison, the pretrained encoder parameters
$\theta$ are frozen. This means BRC does not update the PointNet weights while
training the actor and critic. The embedding value $h_t^{\mathrm{ray}}$ is still
an online quantity: it changes when the sensed pointcloud changes due to object
pose, hand pose, occlusion, or viewpoint. Joint fine-tuning of PointNet with the
policy is a separate future experiment, not part of the first policy comparison.

Before policy training, the raycast PointNet is pretrained with auxiliary heads
that read from $h_t^{\mathrm{ray}}$. These heads force the embedding to contain
object-state and shape information, but they are not used by BRC. BRC receives
$h_t^{\mathrm{ray}}$, not the auxiliary predictions.

For a training sample from object $m$, the supervised targets are

$$
p_t^\star =
(R_{\mathcal{P},t}^{\mathcal{W}})^\top
(p_{m,t}^{\mathcal{W}} - p_{\mathcal{P},t}^{\mathcal{W}}) / s_x
\in\mathbb{R}^3,
$$

$$
R_t^\star =
(R_{\mathcal{P},t}^{\mathcal{W}})^\top R_{m,t}^{\mathcal{W}}
\in SO(3),
\qquad
u_t^\star = \operatorname{rot6d}(R_t^\star)\in\mathbb{R}^{6},
$$

where $p_{m,t}^{\mathcal{W}}\in\mathbb{R}^3$ and
$R_{m,t}^{\mathcal{W}}\in SO(3)$ are simulator object pose values used only as
supervised training labels.

Rotations use the 6D representation

$$
\operatorname{rot6d}(R) = [R_{:,1}; R_{:,2}] \in \mathbb{R}^{6},
$$

where $R\in SO(3)$ and $R_{:,1},R_{:,2}\in\mathbb{R}^3$ are its first two
columns. A predicted vector $u=[a_1;a_2]\in\mathbb{R}^6$, with
$a_1,a_2\in\mathbb{R}^3$, is mapped back to a rotation matrix by

$$
b_1 = \frac{a_1}{\|a_1\|_2+\epsilon},
$$

$$
\tilde b_2 = a_2 - (b_1^\top a_2)b_1,
\qquad
b_2 = \frac{\tilde b_2}{\|\tilde b_2\|_2+\epsilon},
$$

$$
b_3 = b_1 \times b_2,
\qquad
\Pi(u) = [b_1\ b_2\ b_3] \in SO(3).
$$

where $b_1,b_2,b_3\in\mathbb{R}^3$. Rotation error between
$R_1,R_2\in SO(3)$ is the geodesic distance

$$
d(R_1, R_2) =
\arccos\left(\operatorname{clip}\left(
\frac{\operatorname{tr}(R_1^\top R_2)-1}{2},
-1 + \epsilon,\ 1 - \epsilon
\right)\right),
$$

with a small $\epsilon$ for stable gradients.

The raycast pretraining model predicts

$$
\hat p_t = W_p h_t^{\mathrm{ray}}+b_p\in\mathbb{R}^3,
$$

$$
\hat u_t = W_R h_t^{\mathrm{ray}}+b_R\in\mathbb{R}^6,
$$

$$
z_t = W_z h_t^{\mathrm{ray}}+b_z\in\mathbb{R}^{16}.
$$

Here $W_p\in\mathbb{R}^{3\times64}$,
$W_R\in\mathbb{R}^{6\times64}$, and $W_z\in\mathbb{R}^{16\times64}$.
The biases have dimensions $b_p\in\mathbb{R}^3$, $b_R\in\mathbb{R}^6$, and
$b_z\in\mathbb{R}^{16}$. The vector $z_t$ is a morphology code trained to group
samples from the same object. The loss is

$$
\mathcal{L}_{ray}
=
\lambda_p\,\operatorname{Huber}(\hat p_t-p_t^\star)
+ \lambda_R\,d(\Pi(\hat u_t),R_t^\star)
+ \lambda_z\,\mathcal{L}_{supcon}(z_t,\ell_m),
$$

with current weights $\lambda_p=1$, $\lambda_R=0.5$, and $\lambda_z=0.5$.
Here $\ell_m\in\{1,\ldots,85\}$ is a train-split object label used only during
encoder pretraining.

The supervised contrastive term uses normalized morphology codes
$\bar z_b=z_b/\|z_b\|_2$ for batch element $b$. For a batch of size $B$, let
$b\in\{1,\ldots,B\}$ index samples and let
$A(b)=\{j\in\{1,\ldots,B\}:j\ne b,\ \ell_j=\ell_b\}$ be the other samples of the
same object. The temperature $\tau>0$ is a scalar. Then

$$
\mathcal{L}_{supcon}
=
\sum_{b=1}^{B}
-\frac{1}{|A(b)|}
\sum_{j\in A(b)}
\log
\frac{\exp(\bar z_b^\top \bar z_j / \tau)}
{\sum_{k\ne b}\exp(\bar z_b^\top \bar z_k / \tau)}.
$$

If $A(b)$ is empty, that sample contributes zero to the contrastive loss.
The object labels in this loss are training labels for the encoder, not runtime
inputs to the policy.

The trained policy conditioner uses the 64D encoder feature $h$, not the
prediction heads. Current checkpoint:

`/storage/scratch1/9/arosemberg3/BiggerRegularizedCategorical/checkpoints/phase6c/raycast_pointnet_v1`

Final held-out metrics for this checkpoint:

- mean position error: $0.0119$ m;
- mean rotation error: $103.6^\circ$.

The position result indicates useful state information. Rotation prediction
remains weak at the 114-object scale, so policy performance is the decisive test.

## 6. Mesh Conditioners

Let the canonical mesh for object $m$ be

$$
M_m=(V_m,F_m),
$$

where $V_m\in\mathbb{R}^{n_m\times3}$ is the vertex array, $n_m$ is the number
of vertices, and $F_m\in\{1,\ldots,n_m\}^{f_m\times3}$ is the triangle-face
array with $f_m$ faces. The mesh encoder samples $N=1024$ surface points with
probability proportional to triangle area. The sampled pointcloud is

$$
Q_m\in\mathbb{R}^{1024\times3},
$$

where row $Q_{m,i}\in\mathbb{R}^3$ is sampled point $i$. The points are centered
and scaled to a unit-radius cloud before encoding. Mesh samples use an all-valid
mask $\delta^{\mathrm{mesh}}\in\{0,1\}^{1024}$:

$$
\delta_i^{\mathrm{mesh}}=1,\qquad i=1,\ldots,1024.
$$

To apply the batched PointNet convention from Section 4 to mesh samples, a batch
of objects $m_1,\ldots,m_B$ is formed by setting

$$
P_{b,i} = Q_{m_b,i},
\qquad
\delta_{b,i}=1.
$$

The mesh encoder uses the same PointNet architecture as Section 4 but has its own
parameters $\psi$, separate from the raycast encoder parameters $\theta$. Thus
the static mesh-shape feature is

$$
z_m = \operatorname{PN}_\psi(Q_m,\delta^{\mathrm{mesh}}) \in \mathbb{R}^{64}.
$$

Equivalently, the shape-only mesh conditioner from Section 2 is
$E_\psi^{\mathrm{mesh}}(M_m)=z_m$, where $Q_m$ is the sampled and normalized
pointcloud representation of $M_m$.

For `mesh_shape`, the policy conditioner is:

$$
c_t = z_m \in \mathbb{R}^{64}.
$$

Because $Q_m$ is sampled from a canonical mesh and does not change during the
episode, $z_m$ can be cached after applying $\operatorname{PN}_\psi$. This is
different from `wrist_raycast`, where the input pointcloud and therefore the
embedding can change at every timestep.

For `mesh_pose`, the conditioner appends the current simulator object pose in the
palm frame:

$$
c_t = [z_m,\ p_t^m / 0.34,\ \operatorname{rot6d}(R_t^m)]
\in\mathbb{R}^{73}.
$$

Here $p_t^m\in\mathbb{R}^3$ is the object position in palm frame,
$R_t^m\in SO(3)$ is the object orientation in palm frame, and
$\operatorname{rot6d}(R_t^m)\in\mathbb{R}^6$. In Section 2 notation,
$E_\psi^{\mathrm{mesh+pose}}(M_m,p_t^m,R_t^m)=c_t$. Because `mesh_pose` reads
simulator object pose directly, its results must be labeled privileged.

Current mesh checkpoint:

`/storage/scratch1/9/arosemberg3/BiggerRegularizedCategorical/checkpoints/phase6c/mesh_pointnet_v1`

Final held-out metrics for this checkpoint:

- intra-object embedding similarity: $0.988$;
- inter-object embedding similarity: $0.051$;
- descriptor MAE: $0.492$.

The mesh embedding is well-separated across objects and ready for policy tests.

## 7. Evaluation Logic

The main result is policy performance, not encoder loss. Each policy run should
report:

- per-object success and return;
- train-object mean, median, and bottom-quartile success;
- held-out mean, median, and bottom-quartile success;
- train-to-heldout generalization gap;
- wall-clock time, environment steps, conditioning mode, and checkpoint paths.

The wrist-raycast method is preferred if it learns reliably, beats the
no-conditioner ablation on held-out objects, and does so without object identity,
mesh templates, or simulator pose. Mesh-shape is a strong alternative if static
shape is enough. Mesh-pose is an upper bound for geometry plus state.

## 8. Phase Plan

### Phase 1: Installation and Smoke Testing

- [x] Establish the working Python environment with
  `module load python/3.11.9`, `.venv`, `MUJOCO_GL=egl`, JAX/Flax, MuJoCo,
  Gymnasium, and Gymnasium-Robotics.
- [x] Instantiate and step at least one ShadowHand environment.
- [x] Verify `ParallelEnv`, replay insertion, sampling, update, and evaluation
  paths.
- [x] Record dependency caveats needed to recreate the environment.

### Phase 2: Split Manifest

- [x] Export the repository's existing 85-object train and 29-object held-out
  split to `manifests/shadowhand_split_v1.json`.
- [x] Save object names, source files/symbols, counts, and manifest metadata.
- [x] Require this manifest for all main geometry and policy reports.

### Phase 3: Geometry-Conditioning Infrastructure

- [x] Preserve categorical BRC as a diagnostic baseline.
- [x] Add `none` and `mesh_shape` modes.
- [x] Add `wrist_raycast` and `mesh_pose` modes.
- [x] Keep task ids for replay, reward normalization, and logging.
- [x] Disable learned task embeddings in geometry modes.
- [x] Add smoke tests for action sampling, replay, diagnostics, and one-step
  `train.py` execution.

### Phase 4: Wrist-Raycast Sensor Integration

- [x] Add the wrist/palm camera and raycast site.
- [x] Implement deterministic ray-grid casting, palm-frame point output, and hit
  masks.
- [x] Select $32 \times 32$, $40^\circ$, $0.34$ m raycast settings from sensor
  sweeps.
- [x] Save tracked visual diagnostics in `docs/raycast_sensor_sweep/`.
- [x] Add smoke tests for geometry, frame conventions, and multiple objects.

### Phase 5: Online Conditioner Interface

- [x] Implement `OnlineRaycastConditioner`.
- [x] Implement mask-aware PointNet pooling.
- [x] Implement `MeshPoseConditioner`.
- [x] Ensure online modes append current geometry embeddings at every training and
  evaluation step.
- [x] Add tests proving online geometry modes do not contain learned task
  embeddings.

### Phase 6: PointNet Training and Checkpoint Integration

- [x] Train the wrist-raycast PointNet on train-split raycast observations.
- [x] Train the mesh PointNet on train-split mesh samples.
- [x] Save encoder checkpoints and metadata.
- [x] Load trained checkpoints through `train.py`:
  - `--conditioner_checkpoint` for `wrist_raycast`;
  - `--mesh_encoder_checkpoint` for `mesh_shape` and `mesh_pose`.
- [x] Verify checkpoint-load smokes for all three geometry modes.
- [x] Make learned mesh features deterministic per object and seed, independent
  of request order.
- [x] Keep encoder parameters frozen for the planned BRC policy comparison.

### Phase 7: Multi-Object BRC Policy Training

First run an 8-object pilot before full 85-object training. The pilot should
compare:

- [x] `none`;
- [x] `wrist_raycast`;
- [x] `mesh_shape`;
- [x] `mesh_pose`.

Pilot object set:

`orange,cube,mug,hammer,water_bottle,a_cups,tuna_fish_can,flat_screwdriver`

Pilot settings:

- `max_steps=50000`;
- `start_training=5000`;
- `eval_interval=10000`;
- `eval_episodes=3`;
- `updates_per_step=2`;
- `width_critic=4096`;
- `offline_evaluation=true`;
- `render=false`.

The 8-object pilot is a stability check, not a full-runtime estimate. It
completed successfully, but full training must not be launched from the 8-object
wall-clock numbers alone. `ParallelEnv` steps environments sequentially, so the
rough wall-clock scale from an 8-object, 50k-step pilot to an 85-object,
1M-step run is

$$
\frac{85}{8}\cdot\frac{1{,}000{,}000}{50{,}000}\approx 212.5.
$$

Before full policy training, run an 85-object feasibility benchmark on the
train split:

- [x] measure training throughput for `categorical`, `none`, `wrist_raycast`,
  `mesh_shape`, and `mesh_pose`;
- [x] measure or estimate offline evaluation overhead with `eval_episodes=10`;
- [x] report projected wall-clock for each 1M-step full run;
- [x] decide whether full `wrist_raycast` training is feasible as implemented or
  requires a runtime change before launch.

The 85-object benchmark completed without crashes, NaNs, or memory issues. The
projected wall-clock for 1M steps plus evaluation is approximately:

| Mode | Projected runtime | Decision |
|---|---:|---|
| `categorical` | 4.5 days | feasible |
| `none` | 4.3 days | feasible |
| `mesh_shape` | 3.6 days | feasible |
| `mesh_pose` | 3.1 days | feasible |
| `wrist_raycast` | 20.7 days | mitigate before full launch |

The non-raycast full runs can proceed with ordinary long SLURM allocations.
The `wrist_raycast` run should not be launched for 1M steps until long-run
checkpoint/resume is available or a runtime mitigation is chosen.

Full comparison to run:

| Run family | Train objects | Held-out evaluation | Conditioner | Interpretation |
|---|---:|---:|---|---|
| categorical BRC | 85 | no valid unseen-object test | learned task id | train-object diagnostic |
| no-conditioner BRC | 85 | yes | none | ablation |
| wrist-raycast BRC | 85 | yes | online raycast | primary deployable candidate |
| mesh-shape BRC | 85 | yes | mesh shape | shape-only geometry baseline |
| mesh-pose BRC | 85 | yes | mesh shape plus pose | privileged upper bound |

Default full-run settings:

- `max_steps=1000000`;
- `start_training=5000`;
- `eval_interval=50000`;
- `eval_episodes=10`;
- `updates_per_step=2`;
- `width_critic=4096`;
- multi-task batch size `1024`.

Any hyperparameter change should be shared across comparable modes unless the run
is explicitly labeled as tuning.

Before launching the full `wrist_raycast` run, choose and implement at least one
of:

- checkpoint/resume support for policy training across multiple SLURM jobs;
- a reduced or lower-frequency raycast configuration, followed by a matching
  encoder compatibility check;
- a shorter shared step budget for the first full comparison.

### Phase 8: Single-Object BRC Policy Training

After multi-object results, choose seven train-split objects covering easy,
medium, and hard cases. Provisional set:

- `orange`;
- `tuna_fish_can`;
- `water_bottle`;
- `a_cups`;
- `g_lego_duplo`;
- `d_marbles`;
- `hammer` or `flat_screwdriver`.

For each selected object, train and evaluate:

- wrist-raycast conditioned policy;
- mesh-conditioned policy.

This study distinguishes embedding quality from multi-object policy
interference.

## 9. Known Risks

1. **Raycast runtime cost.** `wrist_raycast` performs 1024 raycasts per
   environment per policy step. Full 85-object training may be slow.
2. **Hidden categorical leakage.** Geometry modes must not re-enable learned task
   embeddings.
3. **Held-out leakage.** Held-out objects must not be used for policy training or
   main encoder-gradient updates.
4. **Privileged mesh-pose interpretation.** Mesh-pose uses simulator pose and is
   not a deployable perception method by itself.
5. **Sparse-reward variance.** Important conclusions may require multiple seeds
   or more evaluation episodes.
6. **Offline metrics can mislead.** Encoder losses gate experiments, but policy
   success and return decide the embedding comparison.
