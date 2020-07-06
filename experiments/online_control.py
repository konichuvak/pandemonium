import ray
import torch
from ray import tune

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium.implementations.q_learning import create_horde
from pandemonium.utilities.schedules import LinearSchedule

device = torch.device('cpu')
EXPERIMENT_NAME = 'ttt'
RESULT_DIR = EXPERIMENT_DIR / 'tune'

total_steps = int(1e5)
room_size = tune.grid_search([5, 8, 10])

if __name__ == "__main__":
    ray.init(local_mode=True)
    analysis = tune.run(
        Loop,
        name=EXPERIMENT_NAME,
        # num_samples=3,  # number of seeds
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
                    framework='torch'
                )
            },

            # Architecture
            'gamma': 1.,
            'trace_decay': tune.grid_search(torch.arange(0, 1.1, 0.1).tolist()),

            # Optimizer a.k.a. Horde
            "horde_fn": create_horde,

            # === RLLib params ===
            "use_pytorch": True,
            "env": "MiniGrid-EmptyEnv-Simple-v0",
            "env_config": {
                'size': room_size
            },
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
        },
        local_dir=RESULT_DIR,
        trial_name_creator=lambda trial: '_'.join(str(trial).split('_')[1:]),
        # checkpoint_freq=1000,  # in training iterations
        # checkpoint_at_end=True,
        verbose=1,
        # resume='PROMPT',
    )
