from functools import reduce

import torch
from gym_minigrid.envs import DoorKeyEnv
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.offline_td import TDn
from pandemonium.demons.control import TDAC
from pandemonium.envs import FourRooms, EmptyEnv
from pandemonium.envs.minigrid.wrappers import OneHotObsWrapper
from pandemonium.envs.wrappers import Torch
from pandemonium.networks.bodies import ConvBody, ConvLSTM, Identity
from pandemonium.policies.discrete import Egreedy
from pandemonium.policies.gradient import VPG
from ray.rllib.utils.schedules import ConstantSchedule
from torch import nn

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

__all__ = ['AGENT', 'ENV', 'WRAPPERS', 'BATCH_SIZE']

# ------------------------------------------------------------------------------
# Specify learning environment
# ------------------------------------------------------------------------------

envs = [
    # EmptyEnv(size=10),
    # FourRooms(),
    DoorKeyEnv(size=7),
    # MultiRoomEnv(4, 4),
    # CrossingEnv(),
]
WRAPPERS = [
    # Non-observation wrappers
    # SimplifyActionSpace,

    # Observation wrappers
    # FullyObsWrapper,
    ImgObsWrapper,
    # OneHotObsWrapper,
    # FlatObsWrapper,
    lambda e: Torch(e, device=device)
]
ENV = reduce(lambda e, wrapper: wrapper(e), WRAPPERS, envs[0])
ENV.unwrapped.max_steps = float('inf')
print(ENV)

# ------------------------------------------------------------------------------
# Specify a question of interest
# ------------------------------------------------------------------------------

target_policy = Egreedy(epsilon=ConstantSchedule(0., framework='torch'),
                        action_space=ENV.action_space)
gvf = GVF(target_policy=target_policy,
          cumulant=Fitness(ENV),
          continuation=ConstantContinuation(.9))

# ------------------------------------------------------------------------------
# Specify learners that will be answering the question
# ------------------------------------------------------------------------------

# ==================================
# Representation learning
# ==================================
obs = ENV.reset()
# feature_extractor = ConvBody(
#     *obs.shape[1:], feature_dim=2 ** 8,
#     channels=(8, 16, 32), kernels=(2, 2, 2), strides=(1, 1, 1)
# )
feature_extractor = ConvLSTM(
    256, 1, *obs.shape[1:], feature_dim=2 ** 8,
    channels=(8, 16, 32), kernels=(2, 2, 2), strides=(1, 1, 1)
)
# feature_extractor = FCBody(state_dim=obs.shape[1], hidden_units=(256,))
# feature_extractor = Identity(state_dim=obs.shape[1])

# ==================================
# Behavioral Policy
# ==================================
policy = VPG(feature_dim=feature_extractor.feature_dim,
             action_space=ENV.action_space)

# ==================================
# Learning Algorithm
# ==================================
BATCH_SIZE = 32
prediction_demons = list()


class ActorCritic(TDAC, TDn):
    pass


control_demon = ActorCritic(
    behavior_policy=policy,
    gvf=gvf,
    avf=nn.Linear(feature_extractor.feature_dim, 1),
    feature=feature_extractor,
    criterion=torch.nn.functional.mse_loss
)

demon_weights = torch.tensor([1.], device=device)
# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------


horde = Horde(
    control_demon=control_demon,
    prediction_demons=prediction_demons,
    aggregation_fn=lambda losses: demon_weights.dot(losses),
    device=device
)
AGENT = Agent(feature_extractor, policy, horde)
print(horde)
