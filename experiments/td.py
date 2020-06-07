from functools import partial
from typing import Dict

import ray
import torch
from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons import ControlDemon, PredictionDemon
from pandemonium.demons.control import CategoricalQ
from pandemonium.demons.online_td import SARSA, QLearning
from pandemonium.envs.minigrid import MinigridDisplay, EmptyEnv
from pandemonium.envs.wrappers import (add_wrappers, Torch,
                                       OneHotObsWrapper)
from pandemonium.policies.discrete import Egreedy
from pandemonium.utilities.schedules import ConstantSchedule
from ray import tune
from ray.tune import register_env

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
EXPERIMENT_NAME = 'DistributionalTD'
RESULT_DIR = EXPERIMENT_DIR / 'tune'


def env_creator(env_config):
    # TODO: Ignore config for now until all the envs are properly registered
    envs = [
        EmptyEnv(size=env_config['size']),
        # FourRooms(),
        # DoorKeyEnv(size=env_config['size']),
        # MultiRoomEnv(minNumRooms=4, maxNumRooms=4),
        # CrossingEnv(),
    ]
    wrappers = [
        # Non-observation wrappers
        # SimplifyActionSpace,

        # Observation wrappers
        FullyObsWrapper,
        ImgObsWrapper,
        OneHotObsWrapper,
        Torch,
    ]
    env = add_wrappers(base_env=envs[0], wrappers=wrappers)
    env.unwrapped.max_steps = float('inf')
    return env


register_env("env", env_creator)


def create_demons(config, env, φ, μ) -> Horde:
    π = Egreedy(epsilon=ConstantSchedule(0., framework='torch'),
                feature_dim=φ.feature_dim,
                action_space=env.action_space)

    print(φ.feature_dim, env.action_space.n)

    aqf = torch.nn.Linear(φ.feature_dim, env.action_space.n, bias=False)
    torch.nn.init.zeros_(aqf.weight)
    # torch.nn.init.zeros_(aqf.bias)

    control_demon = SARSA(
        gvf=GVF(target_policy=π,
                cumulant=Fitness(env),
                continuation=ConstantContinuation(config['gamma'])),
        feature=φ,
        behavior_policy=μ,
        aqf=aqf,
        trace_decay=config['trace_decay']
        # num_atoms=config['num_atoms'],
        # v_min=config.get('v_min'),
        # v_max=config.get('v_max'),
    )
    control_demon.μ.act = partial(control_demon.μ.act, q_fn=aqf)

    demon_weights = torch.tensor([1.]).to(device)
    return Horde(
        demons=[control_demon],
        aggregation_fn=lambda losses: demon_weights.dot(losses),
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
    env = cfg['eval_env'](trainer.config['env_config'])

    display = MinigridDisplay(env)

    iteration = trainer.iteration

    # Visualize value functions of each demon
    for demon in trainer.agent.horde.demons.values():

        if isinstance(demon, ControlDemon):
            # fig = display.plot_option_values(
            #     figure_name=f'iteration {iteration}',
            #     demon=demon,
            # )
            fig = display.plot_option_values_separate(
                figure_name=f'iteration {iteration}',
                demon=demon,
            )
            display.save_figure(fig, f'{trainer.logdir}/{iteration}_qf')

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
room_size = tune.grid_search([5, 8, 10])

if __name__ == "__main__":
    ray.init(local_mode=False)
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
                'param': ConstantSchedule(value=0.1, framework='torch')
                # 'param': LinearSchedule(
                #     schedule_timesteps=total_steps // 2,
                #     final_p=0.1, initial_p=1,
                #     framework='torch'
                # )
            },

            # Architecture
            'gamma': 1.,
            'trace_decay': tune.grid_search([0, 0.5, 0.9, 1]),
            # 'trace_decay': 0,
            # "num_atoms": tune.grid_search([2]),
            # "v_min": 0,
            # "v_max": 1,

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            # optimizer.step performed in the trainer_template is same as agent.learn and includes exp collection and sgd
            # try to see how to write horde.learn as a SyncSampleOptimizer in ray

            # === RLLib params ===
            "env": "env",
            "env_config": {
                'size': room_size
            },
            "rollout_fragment_length": 1,

            # --- Evaluation ---
            "evaluation_interval": 1000,  # per training iteration
            "custom_eval_function": eval_fn,
            "evaluation_num_episodes": 1,
            "evaluation_config": {
                'eval_env': env_creator,
                'eval_env_config': {},
            },

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
