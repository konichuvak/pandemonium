from typing import Dict

import ray
import torch
from ray import tune

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons import ControlDemon, PredictionDemon
from pandemonium.demons.control import CategoricalQ
from pandemonium.envs.minigrid import MinigridDisplay
from pandemonium.experience import ReplayBuffer
from pandemonium.implementations.rainbow import DQN
from pandemonium.policies.discrete import Greedy
from pandemonium.utilities.schedules import LinearSchedule

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
EXPERIMENT_NAME = 'C51'
RESULT_DIR = EXPERIMENT_DIR / 'tune'


def create_demons(config, env, φ, μ) -> Horde:
    replay_cls = ReplayBuffer.by_name(config['replay_name'])
    control_demon = DQN(
        gvf=GVF(
            target_policy=Greedy(
                feature_dim=φ.feature_dim,
                action_space=env.action_space
            ),
            cumulant=Fitness(env),
            continuation=ConstantContinuation(config['gamma'])
        ),
        feature=φ,
        behavior_policy=μ,
        replay_buffer=replay_cls(**config['replay_cfg']),
        target_update_freq=config['target_update_freq'],
        double=config['double'],
        duelling=config['duelling'],
        num_atoms=config['num_atoms'],
        v_min=config.get('v_min'),
        v_max=config.get('v_max'),
    )
    return Horde(
        demons=[control_demon],
        device=device
    )


def eval_fn(trainer: Loop, eval_workers) -> Dict:
    """

    Called every `evaluation_interval` to run the current version of the
    agent in the the evaluation environment for one episode.

    Works for envs with fairly small, enumerable state space like gridworlds.

    Parameters
    ----------
    trainer
    eval_workers

    Returns
    -------

    """
    cfg = trainer.config['evaluation_config']
    env = cfg['eval_env'](cfg['eval_env_config'])

    display = MinigridDisplay(env)

    iteration = trainer.iteration

    # Visualize value functions of each demon
    for demon in trainer.agent.horde.demons.values():

        if isinstance(demon, ControlDemon):
            # fig = display.plot_option_values(
            #     figure_name=f'iteration {iteration}',
            #     demon=demon,
            # )
            # fig = display.plot_option_values_separate(
            #     figure_name=f'iteration {iteration}',
            #     demon=demon,
            # )
            # display.save_figure(fig, f'{trainer.logdir}/{iteration}_qf')

            if isinstance(demon, CategoricalQ) and hasattr(demon, 'num_atoms'):
                fig = display.plot_option_value_distributions(
                    figure_name=f'iteration {iteration}',
                    demon=demon,
                )
                print(f'saving @ {trainer.logdir}/{iteration}_zf')
                display.save_figure(fig, f'{trainer.logdir}/{iteration}_zf')

        elif isinstance(demon, PredictionDemon):
            pass

    return {'dummy': None}


total_steps = int(1e5)

if __name__ == "__main__":
    ray.init(local_mode=True)
    analysis = tune.run(
        Loop,
        name=EXPERIMENT_NAME,
        stop={
            "timesteps_total": total_steps,
        },
        config={
            # Model a.k.a. Feature Extractor
            'feature_name': 'identity',
            'feature_cfg': {},
            # "feature_name": 'conv_body',
            # "feature_cfg": {
            #     'feature_dim': 64,
            #     'channels': (8, 16),
            #     'kernels': (2, 2),
            #     'strides': (1, 1),
            # },

            # Policy
            'policy_name': 'egreedy',
            'policy_cfg': {
                'param': LinearSchedule(
                    schedule_timesteps=total_steps // 2,
                    final_p=0.1, initial_p=1,
                    framework='torch'
                )
            },

            # Replay buffer
            # 'replay_name': 'er',
            # 'replay_cfg': {
            #     'size': int(1e5),
            #     'batch_size': 10,
            # },
            'replay_name': 'per',
            'replay_cfg': {
                'size': int(1e3),
                'batch_size': 10,  # TODO: align with rollout length?
                # Since learning happens on a trajectory of size `batch_size`
                # we want it to be relatively small for n-step returns
                # At the same time, we can still collect the experience in
                # larger chunks
                'alpha': 0.6,
                'beta': LinearSchedule(schedule_timesteps=total_steps,
                                       final_p=0.1,
                                       initial_p=1, framework='torch'),
                'epsilon': 1e-6
            },

            # Architecture
            'gamma': 0.99,
            'target_update_freq': 100,
            'double': tune.grid_search([False]),
            'duelling': tune.grid_search([False]),
            "num_atoms": tune.grid_search([1]),
            # "v_min": 0,
            # "v_max": 1,

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            # optimizer.step performed in the trainer_template is same as agent.learn and includes exp collection and sgd
            # try to see how to write horde.learn as a SyncSampleOptimizer in ray

            # === RLLib params ===
            "env": "MiniGrid-EmptyEnv-ImgOnly-v0",
            "env_config": {
                'size': 10
            },
            "rollout_fragment_length": 10,

            # --- Evaluation ---
            # "evaluation_interval": 3,  # per training iteration
            # "custom_eval_function": eval_fn,
            # "evaluation_num_episodes": 1,
            # "evaluation_config": {
            #     'eval_env': env_creator,
            #     'eval_env_config': {},
            # },

            # used as batch size for exp collector and ER buffer
            # "train_batch_size": 32,
            "use_pytorch": True,
            # HACK to get the evaluation through
            "model": {
                'conv_filters': [
                    [8, [2, 2], 1],
                    [16, [2, 2], 1],
                    [32, [2, 2], 1],
                ],
                'fcnet_hiddens': [256]
            }
        },
        num_samples=1,
        local_dir=RESULT_DIR,
        # checkpoint_freq=1000,  # in training iterations
        # checkpoint_at_end=True,
        fail_fast=False,
        verbose=1,
        # resume='PROMPT',
    )
