from functools import reduce, partial

import torch
from gym_minigrid.envs import DoorKeyEnv
from gym_minigrid.wrappers import ImgObsWrapper
from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.control import DQN
from pandemonium.envs.wrappers import Torch
from pandemonium.experience import PER, ER
from pandemonium.networks.bodies import ConvBody
from pandemonium.policies.discrete import Egreedy, SoftmaxPolicy
from pandemonium.utilities.schedules import ConstantSchedule, LinearSchedule

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
schedule_steps = int(1e7)
# policy = Egreedy(
#     epsilon=LinearSchedule(schedule_timesteps=schedule_steps,
#                            final_p=0.1, initial_p=1, framework='torch'),
#     action_space=ENV.action_space
# )
policy = SoftmaxPolicy(
    temperature=LinearSchedule(schedule_timesteps=schedule_steps,
                               final_p=0.1, initial_p=1, framework='torch'),
    action_space=ENV.action_space
)

aqf = torch.nn.Linear(feature_extractor.feature_dim, ENV.action_space.n)
policy.act = partial(policy.act, vf=aqf)

# ==================================
# Learning Algorithm
# ==================================
BATCH_SIZE = 32
REPLAY_SIZE = int(1e6)
replay = PER(
    size=REPLAY_SIZE,
    batch_size=BATCH_SIZE,
    alpha=0.6,
    # beta=ConstantSchedule(0.4, framework='torch'),
    beta=LinearSchedule(schedule_timesteps=schedule_steps,
                        initial_p=1, final_p=0.1, framework='torch'),
    epsilon=1e-6
)

# replay = ER(REPLAY_SIZE, BATCH_SIZE)

control_demon = DQN(
    gvf=gvf,
    aqf=aqf,
    avf=torch.nn.Linear(feature_extractor.feature_dim, 1),
    feature=feature_extractor,
    behavior_policy=policy,
    replay_buffer=replay,
    target_update_freq=100,
    double=True,
    duelling=True,
)

demon_weights = torch.tensor([1.], device=device)
# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------


horde = Horde(
    demons=[control_demon],
    aggregation_fn=lambda losses: demon_weights.dot(losses),
    device=device
)
AGENT = Agent(feature_extractor, policy, horde)

# Sanity checks
assert schedule_steps > REPLAY_SIZE
