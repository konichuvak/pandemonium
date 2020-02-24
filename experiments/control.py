from copy import deepcopy
from functools import reduce

import torch
from gym_minigrid.envs import EmptyEnv, DoorKeyEnv, MultiRoomEnv
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from ray.rllib.utils.schedules import LinearSchedule, ConstantSchedule
from tqdm import tqdm

from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.control import DQN, Sarsa, AC
from pandemonium.envs import FourRooms
from pandemonium.envs.utils import generate_all_states
from pandemonium.envs.wrappers import Torch, OneHotObsWrapper
from pandemonium.networks.bodies import ConvBody, FCBody, Identity
from pandemonium.policies.discrete import Egreedy
from pandemonium.policies.gradient import VPG
from pandemonium.utilities.visualization.plotter import Plotter
from pandemonium.utilities.replay import Replay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# ------------------------------------------------------------------------------
# Specify learning environment
# ------------------------------------------------------------------------------

envs = [
    EmptyEnv(size=6),
    # FourRooms(),
    # DoorKeyEnv(size=10),
    # MultiRoomEnv(4, 4)
]
wrappers = [
    # Non-observation wrappers
    # SimplifyActionSpace,

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
# feature_extractor = ConvBody(d=3, w=7, h=7, feature_dim=2**10)
# feature_extractor = FCBody(state_dim=obs.shape[0], hidden_units=(256,))
feature_extractor = Identity(state_dim=obs.shape[0])

# ==================================
# Behavioral Policy
# ==================================
# policy = Random(action_space=env.action_space)

# exploration = LinearSchedule(schedule_timesteps=20000, final_p=0.1)
# policy = Egreedy(epsilon=exploration, action_space=env.action_space)

policy = VPG(feature_dim=feature_extractor.feature_dim, action_space=env.action_space)

# ==================================
# Learning Algorithm
# ==================================
BATCH_SIZE = 32
prediction_demons = list()

# control_demon = Sarsa(gvf=gvf,
#            feature=feature_extractor,
#            behavior_policy=policy,
#            device=device)

# replay = Replay(memory_size=1e5, batch_size=32)
# control_demon = DQN(gvf=gvf,
#            feature=feature_extractor,
#            behavior_policy=policy,
#            replay_buffer=replay,
#            device=device)

control_demon = AC(gvf=gvf, actor=policy, feature=feature_extractor, device=device)

horde = Horde(control_demon=control_demon, prediction_demons=prediction_demons)
print(horde)

# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------

# NOTE: How is agent different from a Demon?
#   Together with the environment, agent produces a steam of (x, A) data,
#   from which the GVFs in question are estimated by demons

# NOTE: How is agent different from a Horde?
#   ??? It is not???

agent = Agent(feature_extractor=feature_extractor, horde=horde)

# ------------------------------------------------------------------------------
# Monitoring tools
# ------------------------------------------------------------------------------

plotter = Plotter(env)

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
    steps = agent.interact(BATCH_SIZE=BATCH_SIZE, env=env, render=False)
    total_steps += steps

    # eps = agent.horde.control_demon.μ._epsilon.value(total_steps)
    pbar.set_description(desc=f'S {steps:5} | '
                              f'T {total_steps:7} | ',
                              # f'ε {round(eps, 3):5}',
                         refresh=False)

    # Record value function
    if e % 10 == 0:
        q = agent.horde.control_demon.behavior_policy(states).probs
        q = q.transpose(0, 1).view(q.shape[1], 4, env.height, env.width)
        q = q.cpu().detach().numpy()
        plotter.save_figure(plotter.plot_value_function, save_path='vf',
                            figure_name=f'episode {e}',
                            value_tensor=q)

