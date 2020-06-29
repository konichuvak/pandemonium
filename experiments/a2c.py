import ray
from ray import tune

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium.implementations.a2c import create_demons

EXPERIMENT_NAME = 'A2C'
total_steps = int(1e5)

if __name__ == "__main__":
    ray.init(local_mode=False)
    analysis = tune.run(
        Loop,
        name=EXPERIMENT_NAME,
        stop={
            "timesteps_total": total_steps
        },
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

            'gamma': tune.grid_search([0.9]),
            'trace_decay': tune.grid_search([0.5]),

            # Optimizer a.k.a. Horde
            "horde_fn": create_demons,
            # optimizer.step performed in the trainer_template is same as agent.learn and includes exp collection and sgd
            # try to see how to write horde.learn as a SyncSampleOptimizer in ray

            # === RLLib params ===
            "use_pytorch": True,
            "env": "MiniGrid-EmptyEnv-ImgOnly-v0",
            "env_config": {
                'size': 10
            },
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
