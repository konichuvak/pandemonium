from datetime import datetime

import tensorboardX

from pandemonium.experience import Trajectory
from pandemonium.utilities.visualization import Plotter

from experiments import EXPERIMENT_DIR, RLogger

# from experiments.option_critic import *
# from experiments.a2c import *
from experiments.unreal import *


def gen_pbar(stats):
    rounding = 4
    metric_names = {
        'entropy': ('H', rounding + 2),
        'policy_grad': ('π_∇', rounding + 3),
        'policy_loss': ('π_loss', rounding + 3),
        'value_loss': ('v_loss', rounding + 3),
        'loss': ('loss', rounding + 3),
    }
    metrics = ''
    for metric, value in stats.items():
        metric, decimals = metric_names.get(metric, (metric, 5))
        metrics += f'{metric} {round(value, rounding):{decimals}} | '
    return metrics


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


# ------------------------------------------------------------------------------
# Monitoring and plotting tools set up
# ------------------------------------------------------------------------------

EXPERIMENT_PATH = EXPERIMENT_DIR / str(datetime.now().replace(microsecond=0))
EXPERIMENT_PATH.mkdir()

logger = RLogger()
tb_writer = tensorboardX.SummaryWriter(EXPERIMENT_PATH)

plotter = Plotter(ENV)

# Generate all possible states to query value functions for
# test_env = deepcopy(ENV)
# states = generate_all_states(test_env, WRAPPERS)
# states = torch.stack([s[0] for s in states]).squeeze()

# ------------------------------------------------------------------------------
# Learning and evaluation loop
# ------------------------------------------------------------------------------

total_steps = total_time = total_updates = 0

for episode in range(500 + 1):
    for logs in AGENT.interact(BATCH_SIZE=BATCH_SIZE, env=ENV):

        done = logs.pop('done')
        if done:
            total_steps += logs['episode_steps']
            total_time += logs['episode_time']
            total_updates += logs['episode_updates']

            # Record per-episode averages
            pass

            # Record value function
            # if episode % 25 == 0 and episode:
            #     viz(episode, states, plotter)

        else:
            # Generate progress bar
            step = total_steps + logs.pop('episode_steps')
            logs.update(**trajectory_stats(logs.pop('trajectory')))

            # {gen_pbar(logs)}
            desc = f"E {episode:3} | STEP {step:7} | TIME {total_time + logs.pop('episode_time'):5}"
            logger.info(desc)

            # Log a computational graph
            graph = logs.pop('graph', None)
            if graph is not None:
                graph.render(f'{EXPERIMENT_PATH}/graph')

            # Tensorboard logging
            exclude_from_tb = {
                'episode_updates',
            }
            for field, value in logs.items():
                if field not in exclude_from_tb:
                    tb_writer.add_scalar(f'info/{field}', value, step)
