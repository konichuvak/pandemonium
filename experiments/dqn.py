from functools import partial, reduce

import ray
import torch
from gym_minigrid.envs import DoorKeyEnv
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
        # EmptyEnv(size=10),
        # FourRooms(),
        DoorKeyEnv(size=7),
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


def create_demons(config, env, feature_extractor, policy) -> Horde:
    π = Egreedy(epsilon=ConstantSchedule(0., framework='torch'),
                feature_dim=feature_extractor.feature_dim,
                action_space=env.action_space)

    # Value-based policies require value function to evaluate actions
    aqf = torch.nn.Linear(feature_extractor.feature_dim, env.action_space.n)
    policy.act = partial(policy.act, vf=aqf)

    replay_cls = ReplayBuffer.by_name(config['replay_name'])
    control_demon = DQN(
        gvf=GVF(target_policy=π,
                cumulant=Fitness(env),
                continuation=ConstantContinuation(.9)),
        aqf=aqf,
        avf=torch.nn.Linear(feature_extractor.feature_dim, 1),
        feature=feature_extractor,
        behavior_policy=policy,
        replay_buffer=replay_cls(**config['replay_cfg']),
        target_update_freq=config['target_update_freq'],
        double=config['double'],
        duelling=config['duelling'],
    )
    horde = Horde(
        demons=[control_demon],
        aggregation_fn=lambda losses: torch.tensor([1.], device=device),
        device=device
    )

    return horde


if __name__ == "__main__":
    ray.init(local_mode=False)
    analysis = tune.run(
        Loop,
        name='DQN',
        # stop={
        #     "episodes_total": 10000,
        # },
        config={
            # Model a.k.a. Feature Extractor
            "feature_name": 'conv_body',
            "feature_cfg": {
                'feature_dim': 512,
                'channels': (8, 16, 32),
                'kernels': (2, 2, 2),
                'strides': (1, 1, 1),
            },

            # Policy
            'policy_name': 'softmax',
            'policy_cfg': {
                'temperature': LinearSchedule(schedule_timesteps=int(1e7),
                                              final_p=0.1,
                                              initial_p=1, framework='torch')
            },

            # Replay buffer
            'replay_name': 'per',
            'replay_cfg': {
                'size': int(1e6),
                'batch_size': 32,
                'alpha': 0.6,
                'beta': LinearSchedule(schedule_timesteps=int(1e7), final_p=0.1,
                                       initial_p=1, framework='torch'),
                'epsilon': 1e-6
            },

            # Architecture
            'target_update_freq': 100,
            'double': tune.grid_search([False, True]),
            'duelling': tune.grid_search([False, True]),

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            # optimizer.step performed in the trainer_template is same as agent.learn and includes exp collection and sgd
            # try to see how to write horde.learn as a SyncSampleOptimizer in ray

            # === RLLib params ===
            "use_pytorch": True,
            "env": "DQN_env",
            "env_config": {},
            "rollout_fragment_length": tune.grid_search([64, 128]),
            # used as batch size for exp collector and ER buffer
            # "train_batch_size": 32,
        },
        num_samples=1,
        local_dir=EXPERIMENT_DIR,
        checkpoint_freq=1000,  # in training iterations
        checkpoint_at_end=True,
        fail_fast=True,
        verbose=1,
        # resume='PROMPT',
    )
