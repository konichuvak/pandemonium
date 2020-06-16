from functools import reduce

import ray
import torch
from gym_minigrid.envs import EmptyEnv
from gym_minigrid.wrappers import ImgObsWrapper
from ray import tune
from ray.tune import register_env
from torch.nn.functional import mse_loss

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.envs.minigrid import MinigridDisplay
from pandemonium.envs.wrappers import Torch
from pandemonium.implementations import AC
from pandemonium.policies.discrete import Greedy

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
EXPERIMENT_NAME = 'A2C'


def env_creator(env_config):
    # TODO: Ignore config for now until all the envs are properly registered
    envs = [
        EmptyEnv(size=10),
        # FourRooms(),
        # DoorKeyEnv(size=7),
        # MultiRoomEnv(minNumRooms=4, maxNumRooms=4),
        # DeepmindLabEnv(level='seekavoid_arena_01')
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
    return env


register_env("A2C_env", env_creator)


def create_demons(config, env, feature_extractor, policy) -> Horde:
    control_demon = AC(
        gvf=GVF(
            target_policy=Greedy(
                feature_dim=feature_extractor.feature_dim,
                action_space=env.action_space
            ),
            cumulant=Fitness(env),
            continuation=ConstantContinuation(config['gamma'])),
        behavior_policy=policy,
        feature=feature_extractor,
        criterion=mse_loss
    )

    demon_weights = torch.tensor([1.]).to(device)
    horde = Horde(
        demons=[control_demon],
        aggregation_fn=lambda losses: demon_weights.dot(losses),
        device=device
    )

    return horde


EVAL_ENV = env_creator(dict())


def eval_fn(trainer: Loop, eval_workers):
    # cfg = trainer.config
    # env = trainer.env_creator(cfg['env_config'])
    env = EVAL_ENV
    display = MinigridDisplay(env, [])
    states = torch.stack([s[0] for s in display.all_states]).squeeze()

    episode = 1
    # Visualize action-value function of each demon
    for demon in trainer.agent.horde.demons.values():
        x = demon.feature(states)
        v = demon.predict(x)
        v = v.transpose(0, 1).view(4, env.height - 2, env.width - 2)
        v = torch.nn.ConstantPad2d(1, 0)(v)
        v = v.cpu().detach().numpy()

        fig = display.plot_state_values(
            figure_name=f'episode {episode}',
            v=v,
        )
        display.save_figure(fig,
                            save_path=f'{EXPERIMENT_DIR}/{EXPERIMENT_NAME}/vf{episode}',
                            auto_open=False)

    return {'dummy': None}


if __name__ == "__main__":
    ray.init(local_mode=False)
    analysis = tune.run(
        Loop,
        name=EXPERIMENT_NAME,
        stop={"timesteps_total": int(1e5)},
        config={
            # Model a.k.a. Feature Extractor
            "feature_name": 'conv_body',
            "feature_cfg": {
                'feature_dim': 256,
                'channels': (8, 16),
                'kernels': (2, 2),
                'strides': (1, 1),
            },

            # Policy
            "policy_name": 'VPG',
            "policy_cfg": {
                'entropy_coefficient': 0.01,
            },

            'gamma': 0.9,

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            # optimizer.step performed in the trainer_template is same as agent.learn and includes exp collection and sgd
            # try to see how to write horde.learn as a SyncSampleOptimizer in ray

            # === RLLib params ===
            "use_pytorch": True,
            "env": "A2C_env",
            "env_config": {},
            "rollout_fragment_length": 16,  # batch size for exp collector
            # "train_batch_size": 32,

            # # --- Evaluation ---
            # "evaluation_interval": 100,  # per training iteration
            # "custom_eval_function": eval_fn,
            # "evaluation_num_episodes": 1,
            # "evaluation_config": {
            #     "env_config": {},
            # },
            # # HACK to get the evaluation through
            # "model": {
            #     'conv_filters': [
            #         [8, [2, 2], 1],
            #         [16, [2, 2], 1],
            #         [32, [2, 2], 1],
            #     ],
            #     'fcnet_hiddens': [256]
            # }

        },
        num_samples=1,
        local_dir=EXPERIMENT_DIR,
        checkpoint_freq=1000,  # in training iterations
        checkpoint_at_end=True,
        fail_fast=True,
        verbose=1,
        # resume='PROMPT',
    )
