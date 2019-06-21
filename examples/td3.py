"""
(From Vitchyr, was for defaults:)
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.

(Daniel: I added to change support to other envs, and adjust hyperparameters.)
"""
from gym.envs.mujoco import (
        AntEnv,
        InvertedPendulumEnv,
        HalfCheetahEnv,
        HopperEnv,
        ReacherEnv,
        SwimmerEnv,
        Walker2dEnv,
)
import argparse

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def experiment(variant, args):
    # Doesn't work :(
    #import gym
    #expl_env = NormalizedBoxEnv( gym.make(args.env) )
    #eval_env = NormalizedBoxEnv( gym.make(args.env) )

    if 'Ant' in args.env:
        expl_env = NormalizedBoxEnv( AntEnv() )
        eval_env = NormalizedBoxEnv( AntEnv() )
    elif 'InvertedPendulum' in args.env:
        expl_env = NormalizedBoxEnv( InvertedPendulumEnv() )
        eval_env = NormalizedBoxEnv( InvertedPendulumEnv() )
    elif 'HalfCheetah' in args.env:
        expl_env = NormalizedBoxEnv( HalfCheetahEnv() )
        eval_env = NormalizedBoxEnv( HalfCheetahEnv() )
    elif 'Hopper' in args.env:
        expl_env = NormalizedBoxEnv( HopperEnv() )
        eval_env = NormalizedBoxEnv( HopperEnv() )
    elif 'Reacher' in args.env:
        expl_env = NormalizedBoxEnv( ReacherEnv() )
        eval_env = NormalizedBoxEnv( ReacherEnv() )
    elif 'Swimmer' in args.env:
        expl_env = NormalizedBoxEnv( SwimmerEnv() )
        eval_env = NormalizedBoxEnv( SwimmerEnv() )
    elif 'Walker2d' in args.env:
        expl_env = NormalizedBoxEnv( Walker2dEnv() )
        eval_env = NormalizedBoxEnv( Walker2dEnv() )
    else:
        raise ValueError(args.env)

    # Back to normal.
    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    es = GaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--env', type=str)
    args = pp.parse_args()

    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        replay_buffer_size=int(1E6),
    )
    # By default GPU usage is True
    ptu.set_gpu_mode(True)
    setup_logger('rlkit-td3-{}'.format(args.env), variant=variant)
    experiment(variant, args)
