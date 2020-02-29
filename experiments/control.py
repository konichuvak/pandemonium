from copy import deepcopy
from datetime import datetime
from functools import reduce

import tensorboardX
import torch
from gym_minigrid.envs import EmptyEnv, DoorKeyEnv
from gym_minigrid.wrappers import ImgObsWrapper
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule
from tqdm import tqdm

from experiments import EXPERIMENT_DIR, RLogger
from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.control import AC, OC
from pandemonium.envs.utils import generate_all_states
from pandemonium.envs.wrappers import Torch
from pandemonium.experience import Transition, Trajectory
from pandemonium.networks.bodies import ConvBody
from pandemonium.policies.discrete import Egreedy, EgreedyOverOptions
from pandemonium.policies.gradient import VPG
from pandemonium.utilities.visualization.plotter import Plotter
from pandemonium.utilities.spaces import create_option_space

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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
feature_extractor = ConvBody(d=3, w=7, h=7, feature_dim=2 ** 8)
# feature_extractor = FCBody(state_dim=obs.shape[0], hidden_units=(256,))
# feature_extractor = Identity(state_dim=obs.shape[0])

# ==================================
# Behavioral Policy
# ==================================
# policy = Random(action_space=env.action_space)

# exploration = LinearSchedule(schedule_timesteps=20000, final_p=0.1)
# policy = Egreedy(epsilon=exploration, action_space=env.action_space)

# policy = VPG(feature_dim=feature_extractor.feature_dim,
#              action_space=env.action_space)
#
option_space = create_option_space(
    n=2, action_space=env.action_space,
    feature_dim=feature_extractor.feature_dim
)
policy = EgreedyOverOptions(
    # epsilon=LinearSchedule(schedule_timesteps=20000, final_p=0.1),
    epsilon=ConstantSchedule(0.1),
    option_space=option_space
)
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

# control_demon = AC(gvf=gvf, actor=policy, feature=feature_extractor, device=device)
#
control_demon = OC(gvf=gvf, actor=policy,
                   feature=feature_extractor, device=device)

horde = Horde(control_demon=control_demon, prediction_demons=prediction_demons)
print(horde)

# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------

agent = Agent(feature_extractor=feature_extractor, horde=horde)

# ------------------------------------------------------------------------------
# Monitoring tools
# ------------------------------------------------------------------------------

EXPERIMENT_PATH = EXPERIMENT_DIR / str(datetime.now().replace(microsecond=0))
EXPERIMENT_PATH.mkdir()

logger = RLogger()
tb_writer = tensorboardX.SummaryWriter(EXPERIMENT_PATH)

plotter = Plotter(env)

# Generate all possible states to query value function for
states = generate_all_states(test_env, wrappers)
states = torch.stack([s[0] for s in states]).squeeze()

# ------------------------------------------------------------------------------
# Learning and evaluation loop
# ------------------------------------------------------------------------------


def trajectory_stats(traj: Trajectory):

    if not isinstance(traj, Trajectory):
        raise TypeError(type(traj))

    stats = dict()

    # Action frequencies
    pass

    # External reward
    stats.update({
        'max_reward': traj.r.max().item(),
        'min_reward': traj.r.min().item(),
        'mean_reward': traj.r.mean().item(),
        'std_reward': traj.r.std().item(),
    })
    return stats


def gen_pbar(stats):
    rounding = 4
    metric_names = {
        'entropy': ('H', rounding+2),
        'policy_grad': ('π_∇', rounding+3),
        'policy_loss': ('π_loss', rounding+3),
        'value_loss': ('v_loss', rounding+3),
        'loss': ('loss', rounding+3),
    }
    metrics = ''
    for metric, value in stats.items():
        metric, decimals = metric_names.get(metric, (metric, 5))
        metrics += f'{metric} {round(value, rounding):{decimals}} | '
    return metrics


# Make this an
total_steps = total_time = total_updates = 0

for episode in range(500 + 1):
    for logs in agent.interact(BATCH_SIZE=BATCH_SIZE, env=env):

        done = logs.pop('done')
        if done:
            total_steps += logs['episode_steps']
            total_time += logs['episode_time']
            total_updates += logs['episode_updates']

            # Record per-episode averages
            pass

            # Record value function
            if episode % 1 == 0 and episode:

                grid_shape = (4, env.height, env.width)

                # Visualize value function
                x = control_demon.feature(states)
                q = control_demon.μ.dist(x, agent.horde.control_demon.value_head).probs
                q = q.transpose(0, 1).view(q.shape[1], 4, env.height, env.width)
                fig = plotter.plot_option_value_function(
                    figure_name=f'episode {episode}',
                    q=q.cpu().detach().numpy(),
                    option_ids=tuple(control_demon.μ.option_space.options)
                )
                plotter.save_figure(fig, save_path=f'{EXPERIMENT_PATH}/vf{episode}')

                # Visualize individual policies and continuations of options
                n = len(control_demon.μ.option_space)
                pi = torch.empty((n, env.action_space.n, *grid_shape))
                beta = torch.empty((n, *grid_shape))
                for option_id, option in control_demon.μ.option_space.options.items():
                    pi[option_id] = option.policy.dist(x).probs.transpose(0, 1).view(env.action_space.n, *grid_shape)
                    beta[option_id] = option.continuation(x).squeeze().view(grid_shape)

                fig = plotter.plot_option_continuation(
                    figure_name=f'episode {episode}',
                    beta=beta.cpu().detach().numpy(),
                    option_ids=tuple(control_demon.μ.option_space.options)
                )
                plotter.save_figure(fig, save_path=f'{EXPERIMENT_PATH}/beta{episode}')

                figures = plotter.plot_option_action_values(
                    figure_name=f'episode {episode}',
                    pi=pi.cpu().detach().numpy(),
                )
                for i, fig in enumerate(figures):
                    plotter.save_figure(fig, save_path=f'{EXPERIMENT_PATH}/pi{episode}_o{i}')

        else:
            # Generate progress bar
            step = total_steps+logs['episode_steps']
            logs.update(**logs.pop(id(control_demon)))
            logs.update(**trajectory_stats(logs.pop('trajectory')))

            desc = f"E {episode:3} | STEP {step:7} | {gen_pbar(logs)} | TIME {total_time + logs['episode_time']:5}"
            logger.info(desc)

            exclude_from_tb = {
                'episode_time',
                'episode_steps',
                'episode_updates',
            }
            for field, value in logs.items():
                if field not in exclude_from_tb:
                    tb_writer.add_scalar(f'info/{field}', value, step)





