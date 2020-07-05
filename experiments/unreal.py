from collections import deque

import ray
import torch
from ray import tune
from tqdm import tqdm

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium.experience import Transition, Trajectory
from pandemonium.implementations.unreal import create_demons


# ------------------------------------------------------------------------------
# Evaluation setup

# FPS = 60
# display_env = Torch(
#     env=DeepmindLabEnv(
#         level='seekavoid_arena_01',
#         width=600,
#         height=600,
#         fps=FPS,
#         display_size=(900, 600),
#         render=True,
#     ),
#     device=device
# )


def viz():
    import pygame
    clock = pygame.time.Clock()

    display = display_env.display
    display_env.reset()
    s0 = ENV.reset()
    x0 = feature_extractor(s0)
    v = control_demon(x0)

    # Simple circular buffers for displaying value trace and reward predictions
    # states = deque([s0], maxlen=rp.sequence_size)
    values = deque([v], maxlen=100)

    for _ in tqdm(range(300)):
        # Step in the actual environment with the AC demon
        a, policy_info = policy(x0)
        s1, reward, done, info = ENV.step(a)
        x1 = feature_extractor(s1)
        t = Transition(s0, a, reward, s1, done, x0, x1, info=info)
        traj = Trajectory.from_transitions([t])

        # Step in the hi-res env for display purposes
        display_env.step(a)

        # Reset canvas
        display.surface.fill((0, 0, 0))

        # Display agent's observation
        display.show_image(display_env.last_obs)

        # Display rolling value of states
        values.append(control_demon.predict(x1))
        display.show_value(torch.tensor(list(values)).cpu().detach().numpy())

        # Display reward prediction bar
        # states.append(s0)
        # if len(states) != states.maxlen:
        #     continue
        # x = rp.feature(torch.cat(list(states)))
        # v = rp.predict(x.view(1, -1)).softmax(1)
        # v = v.squeeze().cpu().detach().numpy()
        # display.show_reward_prediction(reward=reward, rp_c=v)

        # Display pixel change
        z = pc.gvf.cumulant(traj).squeeze()
        x = pc.feature(traj.s0)
        v = pc.predict_q(x, target=False)[[0], traj.a]
        v = v.squeeze().cpu().detach().numpy()
        z = z.squeeze().cpu().detach().numpy()
        display.show_pixel_change(z, display.obs_shape[0], 0, 3.0, "PC")
        display.show_pixel_change(v, display.obs_shape[0] + 100, 0, 0.4, "PC Q")

        # Update surface, states and tick
        pygame.display.update()
        s0, x0 = s1, x1
        clock.tick(FPS)


# scheduler = PopulationBasedTraining(
#     time_attr="training_iteration",
#     metric="episode_reward",
#     mode="max",
#     perturbation_interval=5,
#     hyperparam_mutations={
#         "lr": lambda: np.random.uniform(0.0001, 1),
#         "momentum": [0.8, 0.9, 0.99],
#     })


if __name__ == "__main__":
    ray.init(local_mode=True)
    analysis = tune.run(
        Loop,
        # scheduler=scheduler,
        name='UNREAL',
        # num_samples=3,
        # stop={"episodes_total": 10000},
        local_dir=EXPERIMENT_DIR,
        checkpoint_freq=1000,  # in training iterations
        checkpoint_at_end=True,
        config={
            "env": "DeepmindLabEnv",
            "env_config": {
                'level': 'seekavoid_arena_01',
                # 'level': 'nav_maze_static_01',
            },
            'gamma': tune.grid_search([0.9]),
            "rollout_fragment_length": 20,

            # Feature extractor
            "encoder": 'nature_cnn_3',

            # Policy
            "policy_name": 'VPG',
            "policy_cfg": {'entropy_coefficient': 0.01},

            # Replay buffer
            'buffer_size': 100,
            'target_update_freq': tune.grid_search([100]),

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            'ac_weight': 1.,
            'vr_weight': tune.grid_search([1.]),
            'pc_weight': tune.grid_search([1.]),
            'rp_weight': tune.grid_search([1.]),
        }
    )
