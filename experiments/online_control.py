import ray
import torch
from ray import tune

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium.implementations.q_learning import create_horde
from pandemonium.utilities.schedules import LinearSchedule

EXPERIMENT_NAME = 'OnlineControl'

total_steps = int(1e5)

if __name__ == "__main__":
    ray.init(local_mode=True)
    analysis = tune.run(
        Loop,
        name=EXPERIMENT_NAME,
        # num_samples=3,  # number of seeds
        local_dir=EXPERIMENT_DIR,
        verbose=1,
        stop={"timesteps_total": total_steps},
        config={
            # Model a.k.a. Feature Extractor
            'feature_name': 'identity',
            'feature_cfg': {},

            # Policy
            'policy_name': 'egreedy',
            'policy_cfg': {
                'param': LinearSchedule(
                    schedule_timesteps=total_steps // 2,
                    final_p=0.001, initial_p=1,
                )
            },

            # Architecture
            'gamma': 1.,
            'trace_decay': tune.grid_search(torch.arange(0, 1.1, 0.1).tolist()),

            # Optimizer a.k.a. Horde
            "horde_fn": create_horde,

            # === RLLib params ===
            "use_pytorch": True,
            "env": "MiniGrid-EmptyEnv6x6-Simple-v0",
            "rollout_fragment_length": 1,

            # --- Evaluation ---
            # "evaluation_interval": 1000,  # per training iteration
            # from experiments.tools.evaluation import eval_fn
            # "custom_eval_function": eval_fn,
            # "evaluation_num_episodes": 1,
            # "evaluation_config": {
            #     'eval_env': env_creator,
            #     'eval_env_config': {},
            # },
            #
            # # FIXME: hack to get the evaluation through
            # "model": {
            #     'conv_filters': [
            #         [8, [2, 2], 1],
            #         [16, [2, 2], 1],
            #         [32, [2, 2], 1],
            #     ],
            #     'fcnet_hiddens': [256]
            # }
        }
    )
