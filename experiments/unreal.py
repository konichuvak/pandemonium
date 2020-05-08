from collections import deque

import ray
import torch
from ray import tune
from ray.tune import register_env
from ray.tune.trial import Trial
from tqdm import tqdm

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness, PixelChange
from pandemonium.demons import LinearDemon
from pandemonium.demons.control import PixelControl, TDAC
from pandemonium.demons.offline_td import TDn
from pandemonium.demons.prediction import ValueReplay, RewardPrediction
from pandemonium.envs import DeepmindLabEnv
from pandemonium.envs.wrappers import Torch
from pandemonium.experience import Transition, Trajectory
from pandemonium.experience.buffers import ER, SkewedER
from pandemonium.policies.discrete import Egreedy
from pandemonium.utilities.schedules import ConstantSchedule

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

register_env("DMLab", lambda config: Torch(DeepmindLabEnv(**config),
                                           device=device))

env_config = {
    # 'level': 'seekavoid_arena_01',
    'level': 'nav_maze_static_01',
}

model_cfg = tune.grid_search(['shallow', 'deep'])

replay_params = {
    'buffer_size': 2000,
    'target_update_freq': tune.grid_search([100, 500]),
}

policy_cfg = {'entropy_coefficient': 0.01}

# Horde vs Ray optimizer?
optimizer_params = {
    'ac_weight': 1,
    'vr_weight': 1,
    'pc_weight': 1,
    'rp_weight': 1,
    # 'vr_weight': tune.grid_search([0, 1]),
    # 'pc_weight': tune.grid_search([0, 1]),
    # 'rp_weight': tune.grid_search([0, 1]),
}


def create_demons(config, env, feature_extractor, policy) -> Horde:
    demons = list()

    # Target policy is greedy
    π = Egreedy(epsilon=ConstantSchedule(0., framework='torch'),
                feature_dim=feature_extractor.feature_dim,
                action_space=env.action_space)

    # ==========================================================================
    # Main task performed by control demon
    # ==========================================================================

    # The main task is to optimize for the extrinsic reward
    optimal_control = GVF(
        target_policy=π,
        cumulant=Fitness(env),
        continuation=ConstantContinuation(config['gamma'])
    )

    # Main N-step Actor-Critic agent
    class AC(TDAC, LinearDemon, TDn):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, output_dim=1)

    demons.append(AC(gvf=optimal_control,
                     behavior_policy=policy,
                     feature=feature_extractor))

    # ==========================================================================
    # Auxiliary tasks performed by a mix of prediction and control demons
    # ==========================================================================

    # Create a shared Experience Replay between PC and VR demons
    replay = None
    if config['pc_weight'] or config['vr_weight']:
        replay = ER(size=config['buffer_size'],
                    batch_size=config['rollout_fragment_length'])

    # --------------------------------------------------------------------------
    # Tracks the color intensity over patches of pixels in the image
    # --------------------------------------------------------------------------
    if config['pc_weight']:
        demons.append(PixelControl(
            gvf=GVF(
                target_policy=π,
                cumulant=PixelChange(),
                continuation=ConstantContinuation(config['gamma']),
            ),
            feature=feature_extractor,
            behavior_policy=policy,
            replay_buffer=replay,  # shared with value replay demon
            target_update_freq=config['target_update_freq'],
        ))

    # --------------------------------------------------------------------------
    # The objective of the value replay is to speed up & stabilize A3C agent
    # --------------------------------------------------------------------------
    if config['vr_weight']:
        demons.append(ValueReplay(gvf=optimal_control,
                                  feature=feature_extractor,
                                  behavior_policy=policy,
                                  replay_buffer=replay))

    # --------------------------------------------------------------------------
    # Auxiliary task of maximizing un-discounted n-step return
    # --------------------------------------------------------------------------
    if config['rp_weight']:
        demons.append(RewardPrediction(
            gvf=GVF(
                target_policy=policy,
                cumulant=Fitness(env),
                continuation=ConstantContinuation(0.),
            ),
            feature=feature_extractor,
            behavior_policy=policy,
            replay_buffer=SkewedER(config['buffer_size'],
                                   config['rollout_fragment_length'])
        ))

    # ==========================================================================
    # Combine demons into a horde
    # ==========================================================================

    weights = [
        config.get('ac_weight'),
        config.get('pc_weight'),
        config.get('vr_weight'),
        config.get('rp_weight')
    ]
    demon_weights = torch.tensor(
        data=[w for w in weights if w],
        dtype=torch.float
    ).to(device)

    horde = Horde(
        demons=demons,
        aggregation_fn=lambda losses: demon_weights.dot(losses),
        device=device
    )

    return horde


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


def trial_the_creator(trial: Trial):
    return str(trial)


if __name__ == "__main__":
    ray.init(local_mode=False)
    analysis = tune.run(
        Loop,
        # scheduler=scheduler,
        name='UNREAL',
        # stop={
        #     "episodes_total": 10000,
        # },
        config={
            # Model a.k.a. Feature Extractor
            "feature_name": 'conv_body',
            "feature_cfg": model_cfg,

            # Policy
            "policy_name": 'VPG',
            "policy_cfg": policy_cfg,

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            **replay_params,
            **optimizer_params,
            # optimizer.step performed in the trainer_template is same as agent.learn and includes exp collection and sgd
            # try to see how to write horde.learn as a SyncSampleOptimizer in ray

            'gamma': tune.grid_search([0.9, 0.99]),

            # === RLLib params ===
            "use_pytorch": True,
            "env": "DMLab",
            "env_config": env_config,
            "rollout_fragment_length": 20,  # as per original paper
            # used as batch size for exp collector and ER buffer
            # "train_batch_size": 32,
        },
        trial_name_creator=trial_the_creator,
        num_samples=1,
        local_dir=EXPERIMENT_DIR,
        checkpoint_freq=1000,  # in training iterations
        checkpoint_at_end=True,
        fail_fast=True,
        verbose=1,
        # resume='PROMPT',
    )
