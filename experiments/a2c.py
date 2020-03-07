from functools import reduce

import torch
from gym_minigrid.wrappers import ImgObsWrapper
from ray.rllib.utils.schedules import ConstantSchedule

from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.control import AC
from pandemonium.envs import FourRooms
from pandemonium.envs.wrappers import Torch
from pandemonium.networks.bodies import ConvBody
from pandemonium.policies.discrete import Egreedy
from pandemonium.policies.gradient import VPG

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

__all__ = ['AGENT', 'ENV', 'WRAPPERS', 'BATCH_SIZE']

# ------------------------------------------------------------------------------
# Specify learning environment
# ------------------------------------------------------------------------------

envs = [
    # EmptyEnv(size=10),
    FourRooms(),
    # DoorKeyEnv(size=7),
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

target_policy = Egreedy(epsilon=ConstantSchedule(0.),
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
feature_extractor = ConvBody(d=3, w=7, h=7, feature_dim=2 ** 8)
# feature_extractor = FCBody(state_dim=obs.shape[0], hidden_units=(256,))
# feature_extractor = Identity(state_dim=obs.shape[0])

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
control_demon = AC(gvf=gvf, actor=policy, feature=feature_extractor)

demon_weights = torch.tensor([1.], device=device)
# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------


horde = Horde(
    control_demon=control_demon,
    prediction_demons=prediction_demons,
    aggregation_fn=lambda losses: demon_weights.dot(losses)
)
AGENT = Agent(feature_extractor=feature_extractor, horde=horde)
print(horde)
