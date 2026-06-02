import os
import sys

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from absl import app, flags

from jaxrl.agent.brc_learner import BRC
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.envs import ParallelEnv
from jaxrl.normalizer import RewardNormalizer
from jaxrl.logger import EpisodeRecorder
from jaxrl.env_names import get_environment_list

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('start_training', 5000,'Number of training steps to start training.')
flags.DEFINE_string('env_names', 'cheetah-run', 'Environment name.')
flags.DEFINE_boolean('log_to_wandb', True, 'Whether to log to wandb.')
flags.DEFINE_boolean('offline_evaluation', True, 'Whether to perform evaluations with temperature=0.')
flags.DEFINE_boolean('render', True, 'Whether to log the rendering to wandb.')
flags.DEFINE_integer('updates_per_step', 2, 'Number of updates per step.')
flags.DEFINE_integer('width_critic', 4096, 'Width of the critic network.')
flags.DEFINE_string('conditioning_mode', 'categorical',
                    'Task conditioning: categorical | none | mesh_shape | wrist_raycast | mesh_pose.')
flags.DEFINE_string('split_manifest', None,
                    'Path to split manifest JSON (required for mesh_shape).')
flags.DEFINE_integer('conditioner_embed_dim', 64,
                     'Embedding dimension for online conditioners (wrist_raycast, mesh_pose).')
flags.DEFINE_string('conditioner_checkpoint', None,
                    'Path to trained raycast conditioner checkpoint (for wrist_raycast mode).')
flags.DEFINE_string('mesh_encoder_checkpoint', None,
                    'Path to trained mesh PointNet encoder checkpoint (for mesh_shape / mesh_pose).')
flags.DEFINE_string('env_backend', 'gymnasium',
                    'Environment backend: gymnasium | mjlab.')
flags.DEFINE_integer('mjlab_num_envs', 64,
                     'Number of parallel environments for mjlab backend.')

_VALID_BACKENDS = {'gymnasium', 'mjlab'}

def main(_):
    if FLAGS.env_backend not in _VALID_BACKENDS:
        print(f"Error: --env_backend must be one of {sorted(_VALID_BACKENDS)}; got '{FLAGS.env_backend}'",
              file=sys.stderr)
        sys.exit(1)

    if FLAGS.log_to_wandb:
        import wandb
        wandb.init(
            config=FLAGS,
            entity='',
            project='',
            group=f'{FLAGS.env_names}',
            name=f'{FLAGS.seed}'
        )
        
    env_names = get_environment_list(FLAGS.env_names)

    if FLAGS.env_backend == 'mjlab':
        unsupported = [n for n in env_names if n != 'cube']
        if unsupported:
            print(f"Error: mjlab backend only supports cube; got {unsupported}", file=sys.stderr)
            sys.exit(1)
        if FLAGS.conditioning_mode != 'none':
            print(f"Error: mjlab backend requires --conditioning_mode=none; got {FLAGS.conditioning_mode}", file=sys.stderr)
            sys.exit(1)
        if FLAGS.offline_evaluation:
            print("Warning: mjlab backend does not support offline evaluation; forcing --offline_evaluation=False", file=sys.stderr)
            FLAGS.offline_evaluation = False
        if FLAGS.render:
            print("Warning: mjlab backend does not support render; forcing --render=False", file=sys.stderr)
            FLAGS.render = False
        from jaxrl.mjlab_envs import MjlabParallelEnv
        env = MjlabParallelEnv(env_names, seed=FLAGS.seed, num_envs=FLAGS.mjlab_num_envs)
        eval_env = None
    else:
        env = ParallelEnv(env_names, seed=FLAGS.seed)
        if FLAGS.offline_evaluation:
            eval_env = ParallelEnv(env_names, seed=FLAGS.seed+42)
        else:
            eval_env = None
        
    eval_interval = FLAGS.eval_interval if FLAGS.offline_evaluation else 5000
        
    # Kwargs setup
    kwargs = {}
    kwargs['updates_per_step'] = FLAGS.updates_per_step
    kwargs['width_critic'] = FLAGS.width_critic
    kwargs['conditioning_mode'] = FLAGS.conditioning_mode

    num_tasks = len(env.envs)

    online_conditioner = None

    if FLAGS.conditioning_mode == 'mesh_shape':
        import dex_envs
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), 'assets')
        if FLAGS.mesh_encoder_checkpoint:
            from jaxrl.mesh_conditioner import build_learned_mesh_features
            conditioner_features, conditioner_meta = build_learned_mesh_features(
                env_names, assets_dir, FLAGS.mesh_encoder_checkpoint, seed=FLAGS.seed,
            )
            print(f"Loaded mesh encoder from {FLAGS.mesh_encoder_checkpoint} "
                  f"(feature_dim={conditioner_features.shape[1]})")
        else:
            if not FLAGS.split_manifest:
                print("Error: --split_manifest is required when conditioning_mode=mesh_shape "
                      "and no --mesh_encoder_checkpoint is provided",
                      file=sys.stderr)
                sys.exit(1)
            from jaxrl.mesh_conditioner import build_conditioner_features_from_manifest
            conditioner_features, conditioner_meta = build_conditioner_features_from_manifest(
                env_names, assets_dir, FLAGS.split_manifest,
            )
            print("WARNING: mesh_shape uses deterministic 8D descriptors (no learned encoder). "
                  "Use --mesh_encoder_checkpoint to load a trained PointNet.",
                  file=sys.stderr)
        kwargs['conditioner_features'] = conditioner_features

    elif FLAGS.conditioning_mode == 'wrist_raycast':
        if FLAGS.conditioner_checkpoint:
            from jaxrl.online_conditioner import OnlineRaycastConditioner
            online_conditioner = OnlineRaycastConditioner.load_checkpoint(
                FLAGS.conditioner_checkpoint,
            )
            print(f"Loaded raycast conditioner from {FLAGS.conditioner_checkpoint} "
                  f"(embed_dim={online_conditioner.embed_dim})")
        else:
            from jaxrl.online_conditioner import build_raycast_conditioner
            online_conditioner = build_raycast_conditioner(
                seed=FLAGS.seed, output_dim=FLAGS.conditioner_embed_dim,
            )
            print("WARNING: wrist_raycast conditioner is UNTRAINED (random parameters). "
                  "Use --conditioner_checkpoint to load a trained encoder.",
                  file=sys.stderr)

    elif FLAGS.conditioning_mode == 'mesh_pose':
        import dex_envs
        from jaxrl.mesh_conditioner import MeshPoseConditioner
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), 'assets')
        if FLAGS.mesh_encoder_checkpoint:
            from jaxrl.mesh_conditioner import build_learned_mesh_features
            shape_features, mesh_meta = build_learned_mesh_features(
                env_names, assets_dir, FLAGS.mesh_encoder_checkpoint, seed=FLAGS.seed,
            )
            online_conditioner = MeshPoseConditioner(shape_features=shape_features)
            print(f"Loaded mesh encoder from {FLAGS.mesh_encoder_checkpoint} "
                  f"(shape_dim={shape_features.shape[1]}, total_embed_dim={online_conditioner.embed_dim})")
        else:
            if not FLAGS.split_manifest:
                print("Error: --split_manifest is required when conditioning_mode=mesh_pose "
                      "and no --mesh_encoder_checkpoint is provided",
                      file=sys.stderr)
                sys.exit(1)
            from jaxrl.mesh_conditioner import build_mesh_pose_conditioner
            online_conditioner = build_mesh_pose_conditioner(
                env_names, assets_dir, FLAGS.split_manifest,
                seed=FLAGS.seed, embed_dim=FLAGS.conditioner_embed_dim,
            )
            print("WARNING: mesh_pose uses deterministic 8D shape descriptors (no learned encoder). "
                  "Use --mesh_encoder_checkpoint to load a trained PointNet.",
                  file=sys.stderr)

    obs_sample = env.observation_space.sample()[:1]
    if online_conditioner is not None:
        import numpy as _np
        obs_sample = _np.concatenate(
            [obs_sample, _np.zeros((1, online_conditioner.embed_dim), dtype=_np.float32)],
            axis=-1,
        )

    agent = BRC(
        FLAGS.seed,
        obs_sample,
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        **kwargs,
    )
    
    batch_size = 1024 if num_tasks > 1 else 256

    if online_conditioner is not None:
        import gymnasium as _gym
        aug_dim = env.observation_space.shape[-1] + online_conditioner.embed_dim
        aug_obs_space = _gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_tasks, aug_dim), dtype=np.float32,
        )
        replay_buffer = ParallelReplayBuffer(aug_obs_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_tasks=num_tasks)
    else:
        replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_tasks=num_tasks)

    reward_normalizer = RewardNormalizer(num_tasks, target_entropy=agent.target_entropy, discount=agent.discount)

    statistics_recorder = EpisodeRecorder(num_tasks)

    def _augment_obs(raw_obs):
        if online_conditioner is None:
            return raw_obs
        embeddings = online_conditioner.extract_and_encode(env.envs)
        return np.concatenate([raw_obs, embeddings], axis=-1)

    def _eval_augment_obs(raw_obs, envs):
        if online_conditioner is None:
            return raw_obs
        embeddings = online_conditioner.extract_and_encode(envs)
        return np.concatenate([raw_obs, embeddings], axis=-1)

    observations = _augment_obs(env.reset())

    for i in range(1, FLAGS.max_steps + 1):
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions(observations, temperature=1.0)
        next_raw_obs, rewards, terms, truns, goals = env.step(actions)
        next_observations = _augment_obs(next_raw_obs)
        reward_normalizer.update(rewards, terms, truns)
        statistics_recorder.update(rewards, goals, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_observations)
        observations = next_observations
        raw_obs_after_reset = observations[:, :env.observation_space.shape[-1]]
        raw_obs_after_reset, terms, truns = env.reset_where_done(raw_obs_after_reset, terms, truns)
        observations = _augment_obs(raw_obs_after_reset)
        if i >= FLAGS.start_training:
            batches = replay_buffer.sample(batch_size, FLAGS.updates_per_step)
            batches = reward_normalizer.normalize(batches, agent.get_temperature())
            _ = agent.update(batches, FLAGS.updates_per_step, i)
            if i % eval_interval == 0 and i >= FLAGS.start_training:  
                info_dict = statistics_recorder.log(FLAGS, agent, replay_buffer, reward_normalizer, i, eval_env, render=FLAGS.render, obs_augment_fn=_eval_augment_obs if online_conditioner is not None else None)
                eval_summary = {k: info_dict[k] for k in ('goal', 'return', 'goal_online', 'return_online') if k in info_dict}
                print(f"step={i} {eval_summary}")


if __name__ == '__main__':
    app.run(main)
