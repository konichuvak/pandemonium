from functools import reduce

import torch
from gym_minigrid.envs import EmptyEnv
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.control import DQN
from pandemonium.envs import FourRooms
from pandemonium.envs.wrappers import Torch, OneHotObsWrapper
from pandemonium.networks.bodies import ConvBody, Identity
from pandemonium.policies.discrete import Egreedy
from pandemonium.utilities.replay import Replay
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

__all__ = ['AGENT', 'ENV', 'WRAPPERS', 'BATCH_SIZE']

# ------------------------------------------------------------------------------
# Specify learning environment
# ------------------------------------------------------------------------------

envs = [
    EmptyEnv(size=10),
    # FourRooms(),
    # DoorKeyEnv(size=7),
    # MultiRoomEnv(4, 4),
    # CrossingEnv(),
]
WRAPPERS = [
    # Non-observation wrappers
    # SimplifyActionSpace,

    # Observation wrappers
    FullyObsWrapper,
    # ImgObsWrapper,
    OneHotObsWrapper,
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
# feature_extractor = ConvBody(d=3, w=7, h=7, feature_dim=2 ** 8)
# feature_extractor = FCBody(state_dim=obs.shape[0], hidden_units=(256,))
feature_extractor = Identity(state_dim=obs.shape[0])

# ==================================
# Behavioral Policy
# ==================================

# TODO: tie the warmup period withe the annealed exploration period
policy = Egreedy(
    epsilon=LinearSchedule(schedule_timesteps=50000, final_p=0.1),
    action_space=ENV.action_space
)

# ==================================
# Learning Algorithm
# ==================================
BATCH_SIZE = 32
prediction_demons = list()

replay = Replay(memory_size=100000, batch_size=BATCH_SIZE)
control_demon = DQN(
    gvf=gvf,
    feature=feature_extractor,
    behavior_policy=policy,
    replay_buffer=replay,
    target_update_freq=200,
    warm_up_period=100,
)

# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------


horde = Horde(
    control_demon=control_demon,
    prediction_demons=prediction_demons,
    aggregation_fn=lambda loss: loss,
)
AGENT = Agent(feature_extractor=feature_extractor, horde=horde)
print(horde)
