from functools import reduce

import ray
import torch
from gym_minigrid.envs import MultiRoomEnv
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from ray import tune
from ray.tune import register_env
from torch.nn.functional import mse_loss

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons import LinearDemon
from pandemonium.demons.control import TDAC
from pandemonium.demons.offline_td import TDn
from pandemonium.envs.wrappers import Torch
from pandemonium.policies.discrete import Egreedy
from pandemonium.utilities.schedules import ConstantSchedule

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def env_creator():
    # TODO: Ignore config for now until all the envs are properly registered
    envs = [
        # EmptyEnv(size=10),
        # FourRooms(),
        # DoorKeyEnv(size=7),
        MultiRoomEnv(minNumRooms=4, maxNumRooms=4),
        # CrossingEnv(),
    ]
    wrappers = [
        # Non-observation wrappers
        # SimplifyActionSpace,

        # Observation wrappers
        FullyObsWrapper,
        ImgObsWrapper,
        # OneHotObsWrapper,
        # FlatObsWrapper,
        lambda e: Torch(e, device=device)
    ]
    env = reduce(lambda e, wrapper: wrapper(e), wrappers, envs[0])
    env.unwrapped.max_steps = float('inf')
    return env


register_env("A2C_env", lambda config: env_creator())


def create_demons(config, env, feature_extractor, policy) -> Horde:
    π = Egreedy(epsilon=ConstantSchedule(0., framework='torch'),
                feature_dim=feature_extractor.feature_dim,
                action_space=env.action_space)

    # N-step Actor-Critic agent
    class AC(TDAC, LinearDemon, TDn):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, output_dim=1)

    control_demon = AC(
        gvf=GVF(target_policy=π,
                cumulant=Fitness(env),
                continuation=ConstantContinuation(.9)),
        behavior_policy=policy,
        feature=feature_extractor,
        criterion=mse_loss
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
        name='A2C',
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
            "policy_name": 'VPG',
            "policy_cfg": {
                'entropy_coefficient': 0.01,
            },

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            # optimizer.step performed in the trainer_template is same as agent.learn and includes exp collection and sgd
            # try to see how to write horde.learn as a SyncSampleOptimizer in ray

            # === RLLib params ===
            "use_pytorch": True,
            "env": "A2C_env",
            "env_config": {},
            "rollout_fragment_length": 32,  # batch size for exp collector
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
