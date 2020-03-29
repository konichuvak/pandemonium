from functools import reduce
from typing import Dict

import torch
from gym_minigrid.envs import DoorKeyEnv
from gym_minigrid.wrappers import ImgObsWrapper
from plotly.graph_objects import Figure
from pandemonium.utilities.schedules import ConstantSchedule

from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.control import OC
from pandemonium.envs.minigrid.wrappers import Torch
from pandemonium.networks.bodies import ConvBody
from pandemonium.policies.discrete import Egreedy, EgreedyOverOptions
from pandemonium.utilities.spaces import create_option_space

__all__ = ['AGENT', 'ENV', 'WRAPPERS', 'BATCH_SIZE', 'viz']

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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

target_policy = Egreedy(epsilon=ConstantSchedule(0.01),
                        action_space=ENV.action_space)
gvf = GVF(target_policy=target_policy,
          cumulant=Fitness(ENV),
          continuation=ConstantContinuation(0.9))

# ------------------------------------------------------------------------------
# Specify learners that will be answering the question
# ------------------------------------------------------------------------------

# ================================
# Representation learning (shared)
# ================================
obs = ENV.reset()
feature_extractor = ConvBody(d=3, w=7, h=7, feature_dim=2 ** 8)
# feature_extractor = FCBody(state_dim=obs.shape[0], hidden_units=(256,))
# feature_extractor = Identity(state_dim=obs.shape[0])

# ==========================
# Behavioral Policy (shared)
# ==========================

option_space = create_option_space(
    n=2, action_space=ENV.action_space,
    feature_dim=feature_extractor.feature_dim
)
policy = EgreedyOverOptions(
    # epsilon=LinearSchedule(schedule_timesteps=20000, final_p=0.1),
    epsilon=ConstantSchedule(0.1),
    option_space=option_space
)
# ==================================
# Learning Algorithm (demon)
# ==================================

# TODO: can we can think about each option as a target policy of each individual control demon?

# TODO: do we need continuation and initiation functions at all?
#  Why not just augment the action space of each policy with initiation and
#  termination actions. Then the criterion for initiating a policy would be
#  based on whether the prob of termination of the current policy > prob of
#  initiating for some other policy

BATCH_SIZE = 32
prediction_demons = list()

control_demon = OC(
    gvf=gvf,
    actor=policy,
    feature=feature_extractor,
    target_update_freq=200
)

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


# ------------------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------------------

def viz(episode, states: torch.Tensor, plotter) -> Dict[str, Figure]:
    figures = dict()

    grid_shape = (4, ENV.height, ENV.width)
    n = len(control_demon.μ.option_space)

    # Prepare tensors to be visualized
    π = torch.empty((n, ENV.action_space.n, *grid_shape))
    β = torch.empty((n, *grid_shape))

    x = control_demon.feature(states)
    q = control_demon.μ.dist(x, control_demon.value_head).probs
    q = q.transpose(0, 1).view(option_space, *grid_shape)
    for i, option in control_demon.μ.option_space.options.items():
        π[i] = option.policy.dist(x).probs.transpose(0, 1).view(π.shape[1:])
        β[i] = option.continuation(x).squeeze().view(grid_shape)

    # Plot value function
    fig = plotter.plot_option_value_function(
        figure_name=f'episode {episode}',
        q=q.cpu().detach().numpy(),
        option_ids=tuple(control_demon.μ.option_space.options)
    )
    figures[f'vf_{episode}'] = fig

    # Plot continuation functions of options
    fig = plotter.plot_option_continuation(
        figure_name=f'episode {episode}',
        beta=β.cpu().detach().numpy(),
        option_ids=tuple(control_demon.μ.option_space.options)
    )
    figures[f'β_{episode}'] = fig

    # Plot options' policies
    figs = plotter.plot_option_action_values(
        figure_name=f'episode {episode}',
        pi=π.cpu().detach().numpy(),
    )
    for i, fig in enumerate(figs):
        figures[f'pi_{episode}_o{i}'] = fig

    return figures
