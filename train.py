import os
import sys

os.environ['MUJOCO_GL'] = 'egl'

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
                    'Task conditioning: categorical | none | mesh_shape.')
flags.DEFINE_string('split_manifest', None,
                    'Path to split manifest JSON (required for mesh_shape).')
        
def main(_):
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

    if FLAGS.conditioning_mode == 'mesh_shape':
        if not FLAGS.split_manifest:
            print("Error: --split_manifest is required when conditioning_mode=mesh_shape",
                  file=sys.stderr)
            sys.exit(1)
        from jaxrl.mesh_conditioner import build_conditioner_features_from_manifest
        import dex_envs
        assets_dir = os.path.join(os.path.dirname(dex_envs.__file__), 'assets')
        conditioner_features, conditioner_meta = build_conditioner_features_from_manifest(
            env_names, assets_dir, FLAGS.split_manifest,
        )
        kwargs['conditioner_features'] = conditioner_features

    agent = BRC(
        FLAGS.seed,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        **kwargs,
    )
    
    batch_size = 1024 if num_tasks > 1 else 256

    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_tasks=num_tasks)   
    
    reward_normalizer = RewardNormalizer(num_tasks, target_entropy=agent.target_entropy, discount=agent.discount)
        
    statistics_recorder = EpisodeRecorder(num_tasks)
    
    observations = env.reset()

    for i in range(1, FLAGS.max_steps + 1):
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions(observations, temperature=1.0)
        next_observations, rewards, terms, truns, goals = env.step(actions)
        reward_normalizer.update(rewards, terms, truns)
        statistics_recorder.update(rewards, goals, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_observations)
        observations = next_observations
        observations, terms, truns = env.reset_where_done(observations, terms, truns)
        if i >= FLAGS.start_training:
            batches = replay_buffer.sample(batch_size, FLAGS.updates_per_step)
            batches = reward_normalizer.normalize(batches, agent.get_temperature())
            _ = agent.update(batches, FLAGS.updates_per_step, i)
            if i % eval_interval == 0 and i >= FLAGS.start_training:  
                info_dict = statistics_recorder.log(FLAGS, agent, replay_buffer, reward_normalizer, i, eval_env, render=FLAGS.render)

            
if __name__ == '__main__':
    app.run(main)
