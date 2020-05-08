from functools import reduce

import ray
import torch
from gym_minigrid.envs import EmptyEnv
from gym_minigrid.wrappers import ImgObsWrapper
from ray import tune
from ray.tune import register_env

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.control import DQN
from pandemonium.envs.wrappers import Torch
from pandemonium.experience import ReplayBuffer
from pandemonium.policies.discrete import Egreedy
from pandemonium.utilities.schedules import ConstantSchedule
from pandemonium.utilities.schedules import LinearSchedule

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def env_creator():
    # TODO: Ignore config for now until all the envs are properly registered
    envs = [
        EmptyEnv(size=10),
        # FourRooms(),
        # DoorKeyEnv(size=7),
        # MultiRoomEnv(minNumRooms=4, maxNumRooms=4),
        # CrossingEnv(),
    ]
    wrappers = [
        # Non-observation wrappers
        # SimplifyActionSpace,

        # Observation wrappers
        # FullyObsWrapper,
        ImgObsWrapper,
        # OneHotObsWrapper,
        # FlatObsWrapper,
        lambda e: Torch(e, device=device)
    ]
    env = reduce(lambda e, wrapper: wrapper(e), wrappers, envs[0])
    env.unwrapped.max_steps = float('inf')
    return env


register_env("DQN_env", lambda config: env_creator())


def create_demons(config, env, φ, μ) -> Horde:
    π = Egreedy(epsilon=ConstantSchedule(0., framework='torch'),
                feature_dim=φ.feature_dim,
                action_space=env.action_space)

    replay_cls = ReplayBuffer.by_name(config['replay_name'])
    control_demon = DQN(
        gvf=GVF(target_policy=π,
                cumulant=Fitness(env),
                continuation=ConstantContinuation(config['gamma'])),
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
    demon_weights = torch.tensor([1.]).to(device)
    return Horde(
        demons=[control_demon],
        aggregation_fn=lambda losses: demon_weights.dot(losses),
        device=device
    )


total_steps = int(1e5)

if __name__ == "__main__":
    ray.init(local_mode=False)
    analysis = tune.run(
        Loop,
        name='DQN_test',
        stop={
            "timesteps_total": total_steps,
        },
        config={
            # Model a.k.a. Feature Extractor
            # 'feature_name': 'identity',
            # 'feature_cfg': {},
            "feature_name": 'conv_body',
            "feature_cfg": {
                'feature_dim': 64,
                'channels': (8, 16),
                'kernels': (2, 2),
                'strides': (1, 1),
            },

            # Policy
            'policy_name': tune.grid_search(['egreedy', 'softmax']),
            'policy_cfg': {
                'param': LinearSchedule(
                    schedule_timesteps=total_steps,
                    final_p=0.1, initial_p=1,
                    framework='torch'
                )
            },

            # Replay buffer
            'replay_name': 'er',
            'replay_cfg': {
                'size': tune.grid_search([int(1e3), int(1e4)]),
                'batch_size': tune.grid_search([10, 20])
            },
            # 'replay_name': 'per',
            # 'replay_cfg': {
            #     'size': int(1e3),
            #     'batch_size': 10,  # TODO: align with rollout length?
            #     # Since learning happens on a trajectory of size `batch_size`
            #     # we want it to be relatively small for n-step returns
            #     # At the same time, we can still collect the experience in
            #     # larger chunks
            #     'alpha': 0.6,
            #     'beta': LinearSchedule(schedule_timesteps=int(1e5), final_p=0.1,
            #                            initial_p=1, framework='torch'),
            #     'epsilon': 1e-6
            # },

            # Architecture
            'gamma': tune.grid_search([0.9, 0.99]),
            'target_update_freq': tune.grid_search([100, 1000]),
            'double': tune.grid_search([False, True]),
            'duelling': tune.grid_search([False, True]),
            "num_atoms": tune.grid_search([1, 10, 51]),
            "v_min": -10.0,
            "v_max": 10.0,

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            # optimizer.step performed in the trainer_template is same as agent.learn and includes exp collection and sgd
            # try to see how to write horde.learn as a SyncSampleOptimizer in ray

            # === RLLib params ===
            "use_pytorch": True,
            "env": "DQN_env",
            "env_config": {},
            "rollout_fragment_length": 10,
            # used as batch size for exp collector and ER buffer
            # "train_batch_size": 32,
        },
        num_samples=1,
        local_dir=EXPERIMENT_DIR,
        # checkpoint_freq=1000,  # in training iterations
        # checkpoint_at_end=True,
        fail_fast=False,
        verbose=1,
        # resume='PROMPT',
    )
