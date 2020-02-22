from copy import deepcopy
from functools import reduce

import numpy as np
import torch
from gym.core import ObservationWrapper
from gym_minigrid.envs import EmptyEnv, DoorKeyEnv, MultiRoomEnv
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from ray.rllib.utils.schedules import LinearSchedule, ConstantSchedule
from tqdm import tqdm

from pandemonium.agent import Agent
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons import Horde
from pandemonium.demons.control import Q1, Sarsa1
from pandemonium.envs import FourRooms
from pandemonium.envs.utils import generate_all_states
from pandemonium.envs.wrappers import (Torch, OneHotObsWrapper,
                                       SimplifyActionSpace)
from pandemonium.gvf import GVF
from pandemonium.networks.bodies import ConvBody, Identity
from pandemonium.policies.discrete import Egreedy
from pandemonium.utilities.visualization.plotter import PlotterOneHot
from pandemonium.utilities.replay import Replay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------
# Specify learning environment
# ------------------------------------------------------------------------------

envs = [
    EmptyEnv(size=8),
    # FourRooms(),
    # DoorKeyEnv(size=8),
    # MultiRoomEnv(4, 4)
]
wrappers = [
    # Non-observation wrappers
    SimplifyActionSpace,

    # Observation wrappers
    # FullyObsWrapper,
    ImgObsWrapper,
    OneHotObsWrapper,
    lambda e: Torch(e, device=device)
]
env = reduce(lambda e, wrapper: wrapper(e), wrappers, envs[0])
env.unwrapped.max_steps = float('inf')
test_env = deepcopy(env)
print(env)

# ------------------------------------------------------------------------------
# Specify a question of interest
# ------------------------------------------------------------------------------

target_policy = Egreedy(epsilon=ConstantSchedule(0.01),
                        action_space=env.action_space)
gvf = GVF(target_policy=target_policy,
          cumulant=Fitness(env),
          termination=ConstantContinuation(0.9))

# ------------------------------------------------------------------------------
# Specify learners that will be answering the question
# ------------------------------------------------------------------------------

# ==================================
# Representation learning
# ==================================
obs = env.reset()
# feature_extractor = ConvBody(d=3,
#                              w=7,
#                              h=7,
#                              # w=env.unwrapped.width,
#                              # h=env.unwrapped.height,
#                              feature_dim=256)
# feature_extractor = FCBody(state_dim=obs.shape[0], hidden_units=(256,))
feature_extractor = Identity(state_dim=obs.shape[0])

# ==================================
# Behavioral Policy
# ==================================
# policy = Random(action_space=env.action_space)
exploration = LinearSchedule(schedule_timesteps=10000, final_p=0.01)
policy = Egreedy(epsilon=exploration, action_space=env.action_space)

# ==================================
# Learning Algorithm
# ==================================

# demon = Sarsa1(gvf=gvf,
#            feature=feature_extractor,
#            behavior_policy=policy,
#            device=device)

replay = Replay(memory_size=1e5, batch_size=32)
demon = Q1(gvf=gvf,
           feature=feature_extractor,
           behavior_policy=policy,
           replay_buffer=replay,
           device=device)

# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------

# NOTE: How is agent different from a Demon?
#   Together with the environment, agent produces a steam of (x, A) data,
#   from which the GVFs in question are estimated by demons

agent = Agent(policy=policy,
              feature_extractor=feature_extractor,
              horde=Horde(control_demon=demon, prediction_demons=[]))

# ------------------------------------------------------------------------------
# Monitoring tools
# ------------------------------------------------------------------------------

plotter = PlotterOneHot(env)

# Generate all possible states
states = generate_all_states(test_env, wrappers)
states = torch.stack([s[0] for s in states]).squeeze()

# ------------------------------------------------------------------------------
# Learning and evaluation loop
# ------------------------------------------------------------------------------

episodes = 500
total_steps = steps = 0
pbar = tqdm(range(episodes),
            leave=False, position=1,
            desc=f'S {steps:5}, T {total_steps:8}')
for e in pbar:
    steps = agent.interact(env, render=False)
    total_steps += steps

    eps = agent.horde.control_demon.μ._epsilon.value(total_steps)
    pbar.set_description(desc=f'S {steps:5} | '
                              f'T {total_steps:7} | '
                              f'ε {round(eps, 3):5}',
                         refresh=False)

    if e % 25 == 0:
        v = agent.horde.control_demon.predict(states)
        v = v.mean(1).view(4, env.height, env.width)
        v = v.cpu().detach().numpy()
        plotter.plot_option_value_function(f'episode {e} Q', v, 'vf')
