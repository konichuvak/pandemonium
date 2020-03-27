from functools import reduce, partial

import torch
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule

from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.control import DQN
from pandemonium.envs import EmptyEnv
from pandemonium.envs.minigrid.wrappers import OneHotObsWrapper
from pandemonium.envs.wrappers import Torch
from pandemonium.experience import PER, ER
from pandemonium.networks.bodies import ConvBody, Identity
from pandemonium.policies.discrete import Egreedy

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
feature_extractor = ConvBody(
    *obs.shape[1:], feature_dim=2 ** 8,
    channels=(8, 16, 32), kernels=(2, 2, 2), strides=(1, 1, 1)
)
# feature_extractor = FCBody(state_dim=obs.shape[1], hidden_units=(256,))
# feature_extractor = Identity(state_dim=obs.shape[1])

# ==================================
# Behavioral Policy
# ==================================

# TODO: tie the warmup period withe the annealed exploration period
schedule_steps = int(10e5)
policy = Egreedy(
    epsilon=LinearSchedule(schedule_timesteps=schedule_steps,
                           final_p=0.1, initial_p=1, framework='torch'),
    action_space=ENV.action_space
)
aqf = torch.nn.Linear(feature_extractor.feature_dim, ENV.action_space.n)
policy.act = partial(policy.act, vf=aqf)

# ==================================
# Learning Algorithm
# ==================================
BATCH_SIZE = 32
# replay = PER(
#     size=100000,
#     batch_size=BATCH_SIZE,
#     alpha=0.95,
#     beta=LinearSchedule(schedule_timesteps=schedule_steps,
#                         initial_p=1, final_p=0.1, framework='torch'),
#     epsilon=1e-6
# )

replay = ER(100000, BATCH_SIZE)

prediction_demons = list()

control_demon = DQN(
    gvf=gvf,
    aqf=aqf,
    avf=torch.nn.Linear(feature_extractor.feature_dim, 1),
    feature=feature_extractor,
    behavior_policy=policy,
    replay_buffer=replay,
    target_update_freq=200,
    # warm_up_period=100,
    warm_up_period=replay.capacity // replay.batch_size,
    double=True,
    duelling=True,
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
