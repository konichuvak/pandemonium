import ray
from ray import tune

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium.implementations.a2c import create_horde

EXPERIMENT_NAME = 'AC'
total_steps = int(1e5)

if __name__ == "__main__":
    ray.init(local_mode=False)
    analysis = tune.run(
        Loop,
        name=EXPERIMENT_NAME,
        stop={"timesteps_total": total_steps},
        config={
            "env": "MiniGrid-EmptyEnv-ImgOnly-v0",
            "env_config": {'size': 10},
            'encoder': 'image',
            "rollout_fragment_length": 16,  # batch size for exp collector

            "policy_name": 'VPG',
            "policy_cfg": {'entropy_coefficient': tune.grid_search([0.01])},
            'gamma': tune.grid_search([0.9]),
            'trace_decay': tune.grid_search([0.5]),
            "horde_fn": create_horde,

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
        # num_samples=1,
        local_dir=EXPERIMENT_DIR,
        # checkpoint_freq=1000,  # in training iterations
        # checkpoint_at_end=True,
        fail_fast=True,
        verbose=1,
        # resume='PROMPT',
    )
