import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agent.update import build_actor_input, update_actor, update_critic, update_target_critic, update_temperature

from jaxrl.networks import NormalTanhPolicy, Critic, Temperature
from jaxrl.utils import Model, PRNGKey, Batch


@functools.partial(jax.jit, static_argnames=('discount', 'target_entropy', 'num_bins', 'v_max', 'multitask'),)
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, None, None, None, None, None))
def _get_infos(
    rng: PRNGKey, 
    actor: Model, 
    critic: Model, 
    target_critic: Model, 
    temp: Model, 
    batch: Batch, 
    discount: float, 
    target_entropy: float, 
    num_bins: int, 
    v_max: float,
    multitask: bool
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    _, critic_info = update_critic(critic_key, actor, critic, target_critic, temp, batch, discount, num_bins, v_max, multitask)
    _, actor_info = update_actor(actor_key, actor, critic, temp, batch, num_bins, v_max, multitask) 
    _, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
    return {
        **critic_info,
        **actor_info,
        **alpha_info,
    }

@jax.jit
def _get_temperature(temp):
    temp_val = temp()
    return temp_val
    
@jax.jit
def _sample_actions(
    rng: PRNGKey,
    actor: Model,
    inputs: np.ndarray,
    temperature: float = 1.0,
):
    dist = actor(inputs, temperature)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    return rng, actions

def _update(
    rng: PRNGKey, 
    actor: Model, 
    critic: Model, 
    target_critic: Model, 
    temp: Model, 
    batch: Batch, 
    discount: float, 
    tau: float, 
    target_entropy: float, 
    num_bins: int, 
    v_max: float,
    multitask: bool
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    new_critic, critic_info = update_critic(critic_key, actor, critic, target_critic, temp, batch, discount, num_bins, v_max, multitask)
    new_target_critic = update_target_critic(new_critic, target_critic, tau)
    new_actor, actor_info = update_actor(actor_key, actor, new_critic, temp, batch, num_bins, v_max, multitask) 
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)
    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info,
    }

@functools.partial(jax.jit, static_argnames=('discount', 'tau', 'target_entropy', 'num_bins', 'v_max', 'multitask', 'num_updates'))
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batches: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    num_bins: int,
    v_max: float,
    multitask: bool, 
    step: int,    
    num_updates: int
):
    def one_step(i, state):
        step, rng, actor, critic, target_critic, temp, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update(
            rng,
            actor,
            critic,
            target_critic,
            temp,
            jax.tree.map(lambda x: jnp.take(x, i, axis=0), batches),
            discount,
            tau,
            target_entropy,
            num_bins,
            v_max,
            multitask
        )
        return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, info

    step, rng, actor, critic, target_critic, temp, info = one_step(0, (step, rng, actor, critic, target_critic, temp, {}))
    return jax.lax.fori_loop(1, num_updates, one_step, (step, rng, actor, critic, target_critic, temp, info))

class BRC(object):

    VALID_CONDITIONING_MODES = (
        "categorical", "none", "mesh_shape", "wrist_raycast", "mesh_pose",
    )

    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        num_tasks: int,
        embedding_size: int = 32,
        ensemble_size: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 0.1,
        updates_per_step: int = 10,
        width_critic: int = 512,
        width_actor: int = 256,
        num_bins: int = 101,
        v_max: float = 10.0,
        conditioning_mode: str = "categorical",
        conditioner_features: Optional[np.ndarray] = None,
    ) -> None:

        assert conditioning_mode in self.VALID_CONDITIONING_MODES, (
            f"Unknown conditioning_mode={conditioning_mode!r}; "
            f"valid: {self.VALID_CONDITIONING_MODES}"
        )

        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.seed = seed
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.num_bins = num_bins
        self.v_max = v_max

        self.num_tasks = num_tasks
        self.embedding_size = embedding_size
        self.task_ids = jnp.arange(num_tasks, dtype=jnp.int32)
        self.conditioning_mode = conditioning_mode

        task_ids_init = self.task_ids[:1]

        # Historical naming: self.multitask controls only the original learned
        # categorical task embedding. Geometry-conditioned runs can still have
        # many tasks, but they must keep this False so update.py does not append
        # categorical embeddings.
        if conditioning_mode == "categorical":
            self.multitask = True if num_tasks > 1 else False
            self.conditioner_features = None
            conditioner_dim = embedding_size if self.multitask else 0
        elif conditioning_mode == "none":
            self.multitask = False
            self.conditioner_features = None
            conditioner_dim = 0
        elif conditioning_mode == "mesh_shape":
            assert conditioner_features is not None, (
                "mesh_shape mode requires conditioner_features"
            )
            assert conditioner_features.shape[0] == num_tasks, (
                f"conditioner_features has {conditioner_features.shape[0]} rows "
                f"but num_tasks={num_tasks}"
            )
            self.multitask = False
            self.conditioner_features = jnp.array(conditioner_features, dtype=jnp.float32)
            conditioner_dim = conditioner_features.shape[1]
        elif conditioning_mode in ("wrist_raycast", "mesh_pose"):
            self.multitask = False
            self.conditioner_features = None
            conditioner_dim = 0

        assert not (conditioning_mode != "categorical" and self.multitask), (
            "Learned categorical embedding must be disabled in non-categorical mode"
        )

        if conditioning_mode == "categorical":
            if self.multitask:
                actor_init = jnp.concatenate(
                    (observations, jnp.zeros((1, embedding_size))), axis=-1
                )
            else:
                actor_init = observations
            critic_obs_init = observations
        elif conditioning_mode == "mesh_shape":
            actor_init = jnp.concatenate(
                (observations, jnp.zeros((1, conditioner_dim))), axis=-1
            )
            critic_obs_init = jnp.concatenate(
                (observations, jnp.zeros((1, conditioner_dim))), axis=-1
            )
        else:
            actor_init = observations
            critic_obs_init = observations

        self._online_conditioned = conditioning_mode in ("wrist_raycast", "mesh_pose")

        multitask_for_critic = self.multitask

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
            actor_def = NormalTanhPolicy(action_dim=action_dim, hidden_dims=width_actor)
            critic_def = Critic(num_tasks=num_tasks, embedding_size=embedding_size, ensemble_size=ensemble_size, hidden_dims=width_critic, depth=2, output_nodes=num_bins, multitask=multitask_for_critic)
            actor = Model.create(actor_def, inputs=[actor_key, actor_init], tx=optax.adamw(learning_rate=actor_lr))
            critic = Model.create(critic_def, inputs=[critic_key, critic_obs_init, actions, task_ids_init], tx=optax.adamw(learning_rate=critic_lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, critic_obs_init, actions, task_ids_init])
            temp = Model.create(Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5))
            return actor, critic, target_critic, temp, rng

        self.init_models = jax.jit(_init_models)
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seed)
        self.step = 1

    def _augment_obs(self, observations, task_ids):
        if self.conditioner_features is None:
            return observations
        features = self.conditioner_features[task_ids]
        return jnp.concatenate((observations, features), axis=-1)

    def _augment_batch(self, batch: Batch) -> Batch:
        if self.conditioner_features is None:
            return batch
        obs = jnp.concatenate(
            (batch.observations, self.conditioner_features[batch.task_ids]), axis=-1
        )
        next_obs = jnp.concatenate(
            (batch.next_observations, self.conditioner_features[batch.task_ids]), axis=-1
        )
        return Batch(
            observations=obs,
            actions=batch.actions,
            rewards=batch.rewards,
            masks=batch.masks,
            next_observations=next_obs,
            task_ids=batch.task_ids,
        )

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0):
        observations = self._augment_obs(observations, self.task_ids)
        inputs = build_actor_input(self.critic, observations, self.task_ids, self.multitask)
        rng, actions = _sample_actions(self.rng, self.actor, inputs, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def update(self, batch: Batch, num_updates: int, env_step: int):
        batch = self._augment_batch(batch)

        step, rng, actor, critic, target_critic, temp, info = _do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.num_bins,
            self.v_max,
            self.multitask,
            self.step,
            num_updates
        )
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        return info
    
    def get_infos(self, batch: Batch):
        batch = self._augment_batch(batch)
        infos = _get_infos(
                    self.rng,
                    self.actor,
                    self.critic,
                    self.target_critic,
                    self.temp,
                    batch,
                    self.discount,
                    self.target_entropy,
                    self.num_bins,
                    self.v_max,
                    self.multitask)
        return infos
    
    def get_temperature(self):
        return _get_temperature(self.temp)

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seeds)
        
    def save(self, path):
        self.actor.save(f'{path}/actor.txt')
        self.critic.save(f'{path}/critic.txt')
        self.target_critic.save(f'{path}/target_critic.txt')
        self.temp.save(f'{path}/temp.txt')
        
    def load(self, path):
        self.actor = self.actor.load(f'{path}/actor.txt')
        self.critic = self.actor.load(f'{path}/critic.txt')
        self.target_critic = self.actor.load(f'{path}/target_critic.txt')
        self.temp = self.actor.load(f'{path}/temp.txt')
