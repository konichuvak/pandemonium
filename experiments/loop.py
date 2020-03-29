import os
from collections import deque
from datetime import datetime

import numpy as np
import tensorboardX
import torch
from experiments import EXPERIMENT_DIR, RLogger
# from experiments.option_critic import *
# from experiments.unreal import *
from experiments.a2c import *
# from experiments.dqn import *
from pandemonium.experience import Trajectory

# Ensure that we are not using more than one core per experiment
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def gen_pbar(stats):
    rounding = 4
    metric_names = {
        'entropy': ('H', rounding + 2),
        'policy_grad': ('π_∇', rounding + 3),
        'policy_loss': ('π_loss', rounding + 3),
        'value_loss': ('v_loss', rounding + 3),
        'loss': ('loss', rounding + 3),
        'epsilon': ('ε', rounding + 3),
    }
    metrics = ''
    for metric, value in stats.items():
        metric, decimals = metric_names.get(metric, (metric, 5))
        metrics += f'{metric} {round(value, rounding):{decimals}} | '
    return metrics


interval = BATCH_SIZE * 100
episodic_metrics = {
    'reward_per_episode': 0,
}
reward_tracker = deque([], maxlen=interval)


def trajectory_stats(traj: Trajectory):
    if not isinstance(traj, Trajectory):
        raise TypeError(type(traj))

    stats = dict()

    # TODO: Action / option frequencies

    # TODO: gradient norm

    # External reward
    episodic_metrics['reward_per_episode'] += traj.r.sum().item()
    for r in list(traj.r.cpu().detach().numpy()):
        reward_tracker.append(r)

    # TODO: Discounts

    for metric, values in traj.info.items():
        if metric == 'epsilon':
            stats.update({'epsilon': values[-1]})
        elif metric == 'temperature':
            stats.update({'temperature': values[-1]})
        elif metric == 'entropy':
            stats.update({'entropy': values.mean().item()})

    stats.update({
        'max_reward': np.max(reward_tracker),
        'min_reward': np.min(reward_tracker),
        'mean_reward': np.mean(reward_tracker),
        'std_reward': np.std(reward_tracker),
    })
    return stats


# ------------------------------------------------------------------------------
# Monitoring and plotting tools set up
# ------------------------------------------------------------------------------

EXPERIMENT_PATH = EXPERIMENT_DIR / str(datetime.now().replace(microsecond=0))
EXPERIMENT_PATH.mkdir()
PARAMETER_DIR = EXPERIMENT_PATH / 'weights'
PARAMETER_DIR.mkdir()

# # Load the weights
# experiment_id = '2020-03-19 14:58:10'
# weight_name = '1600.pt'
# AGENT.horde.load_state_dict(
#     state_dict=torch.load(
#         f=EXPERIMENT_DIR / experiment_id / 'weights' / weight_name,
#         map_location=device,
#     ),
#     strict=True
# )

# Set up the logger
logger = RLogger(level=10,
                 file_logger=(10, f'{EXPERIMENT_PATH}/experiment.log'))
logger.critical(str(ENV))
logger.critical(repr(AGENT.horde))

# Tensorboard set up
tb_writer = tensorboardX.SummaryWriter(EXPERIMENT_PATH)
tb_writers = dict()
for d, demon in AGENT.horde.demons.items():
    tb_writers[f'{d}{id(demon)}'] = tensorboardX.SummaryWriter(
        EXPERIMENT_PATH / f'{demon}{id(demon)}')

# Generate all possible states to query value functions for
# plotter = Plotter(ENV)
# test_env = deepcopy(ENV)
# states = generate_all_states(test_env, WRAPPERS)
# states = torch.stack([s[0] for s in states]).squeeze()

# ------------------------------------------------------------------------------
# Learning and evaluation loop
# ------------------------------------------------------------------------------

SAVE_EVERY = 200
total_steps = total_time = total_updates = 0

for episode in range(10000 + 1):

    # Save the weights
    if episode and episode % SAVE_EVERY == 0:
        torch.save(AGENT.horde.state_dict(), PARAMETER_DIR / f'{episode}.pt')

    # Visualize this episode
    # if episode % 50 == 0 and episode:
    # viz()

    # Play
    for logs in AGENT.interact(BATCH_SIZE=BATCH_SIZE, env=ENV):

        done = logs.pop('done')
        if done:
            total_steps += logs['episode_steps']
            total_time += logs['episode_time']
            total_updates += logs['episode_updates']
            episodic_metrics['steps_per_episode'] = logs['episode_steps']
            episodic_metrics['episode_count'] = episode

            # Update per-episode counters
            for field, value in episodic_metrics.items():
                tb_writer.add_scalar(f'info/{field}', value, total_steps)

            episodic_metrics = {
                'reward_per_episode': 0,
            }

            # Record per-episode averages
            pass
        else:
            # Generate progress bar
            step = total_steps + logs.pop('episode_steps')
            logs.update(**trajectory_stats(logs.pop('trajectory')))

            # if step % (BATCH_SIZE * 10) == 0:
            desc = f"E {episode:3} | STEP {step:7} | " \
                   f"TIME {total_time + logs.pop('episode_time'):5} | " \
                   f"{EXPERIMENT_PATH} | " \
                   f"{ENV.unwrapped.__class__.__name__} | " \
                   f"{AGENT.horde}"
            logger.info(desc)

            # Create a schematic of computational graph
            # if episode % 3 == 0:
            #     graph = make_dot(logs['total_loss'],
            #                      params=dict(AGENT.horde.named_parameters()))
            #     graph.render(f'{EXPERIMENT_PATH}/graph')

            # Tensorboard logging
            exclude_from_tb = {
                'episode_updates',
            }

            for field, value in logs.items():
                if field in exclude_from_tb:
                    continue
                if field in tb_writers:
                    for f, v in value.items():
                        if isinstance(v, float) or (
                                torch.is_tensor(v) and len(v.shape) == 0):
                            tb_writers[field].add_scalar(f'info/{f}', v, step)
                else:
                    tb_writer.add_scalar(f'info/{field}', value, step)
