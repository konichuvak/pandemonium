from functools import partial

import ray
import torch
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from ray import tune
from ray.tune import register_env

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.implementations import OnlineSARSA, OnlineQLearning
from pandemonium.envs.minigrid import EmptyEnv
from pandemonium.envs.wrappers import (add_wrappers, Torch, OneHotObsWrapper,
                                       SimplifyActionSpace)
from pandemonium.utilities.schedules import LinearSchedule

device = torch.device('cpu')
EXPERIMENT_NAME = 'ttt'
RESULT_DIR = EXPERIMENT_DIR / 'tune'


def env_creator(env_config):
    # TODO: pre-register this setup so that it can be reused across algos
    env = EmptyEnv(size=env_config['size'])
    wrappers = [
        # Non-observation wrappers
        SimplifyActionSpace,

        # Observation wrappers
        FullyObsWrapper,
        ImgObsWrapper,
        OneHotObsWrapper,
        Torch,
    ]
    env = add_wrappers(base_env=env, wrappers=wrappers)
    env.unwrapped.max_steps = float('inf')
    return env


register_env("env", env_creator)


def create_demons(config, env, φ, μ) -> Horde:
    aqf = torch.nn.Linear(φ.feature_dim, env.action_space.n, bias=False)
    torch.nn.init.zeros_(aqf.weight)

    control_demon = OnlineSARSA(
        gvf=GVF(
            target_policy=μ,
            cumulant=Fitness(env),
            continuation=ConstantContinuation(config['gamma'])
        ),
        feature=φ,
        behavior_policy=μ,
        aqf=aqf,
        trace_decay=config['trace_decay'],
    )
    control_demon.μ.act = partial(control_demon.μ.act, q_fn=aqf)

    demon_weights = torch.tensor([1.]).to(device)
    return Horde(
        demons=[control_demon],
        aggregation_fn=lambda losses: demon_weights.dot(losses),
        device=device
    )


total_steps = int(1e5)
room_size = 5
# room_size = tune.grid_search([5, 8, 10])

if __name__ == "__main__":
    ray.init(local_mode=True)
    analysis = tune.run(
        Loop,
        name=EXPERIMENT_NAME,
        num_samples=3,  # number of seeds
        stop={"timesteps_total": total_steps},
        config={
            # Model a.k.a. Feature Extractor
            'feature_name': 'identity',
            'feature_cfg': {},

            # Policy
            'policy_name': 'egreedy',
            'policy_cfg': {
                'param': LinearSchedule(
                    schedule_timesteps=total_steps // 2,
                    final_p=0, initial_p=1,
                    framework='torch'
                )
            },

            # Architecture
            'gamma': 1.,
            'trace_decay': 0.9,
            # 'trace_decay': tune.grid_search(torch.arange(0, 1.1, 0.1).tolist()),

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,

            # === RLLib params ===
            "use_pytorch": True,
            "env": "env",
            "env_config": {
                'size': room_size
            },
            "rollout_fragment_length": 1,

            # --- Evaluation ---
            # "evaluation_interval": 1000,  # per training iteration
            # from experiments.tools.evaluation import eval_fn
            # "custom_eval_function": eval_fn,
            # "evaluation_num_episodes": 1,
            # "evaluation_config": {
            #     'eval_env': env_creator,
            #     'eval_env_config': {},
            # },
            #
            # # FIXME: hack to get the evaluation through
            # "model": {
            #     'conv_filters': [
            #         [8, [2, 2], 1],
            #         [16, [2, 2], 1],
            #         [32, [2, 2], 1],
            #     ],
            #     'fcnet_hiddens': [256]
            # }
        },
        local_dir=RESULT_DIR,
        # checkpoint_freq=1000,  # in training iterations
        # checkpoint_at_end=True,
        verbose=1,
        # resume='PROMPT',
    )
