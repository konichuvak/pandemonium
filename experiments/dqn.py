import ray
from ray import tune

from experiments import EXPERIMENT_DIR
from experiments.tools.evaluation import eval_fn
from experiments.trainable import Loop
from pandemonium.implementations.rainbow import create_horde
from pandemonium.utilities.schedules import LinearSchedule

EXPERIMENT_NAME = 'RAINBOW'

total_steps = int(1e5)

if __name__ == "__main__":
    ray.init(local_mode=True)
    analysis = tune.run(
        Loop,
        name=EXPERIMENT_NAME,
        local_dir=EXPERIMENT_DIR,
        # checkpoint_freq=1000,  # in training iterations
        # checkpoint_at_end=True,
        stop={"timesteps_total": total_steps},
        config={

            "env": "MiniGrid-EmptyEnv6x6-ImgOnly-v0",

            # Feature extractor for observations
            'encoder': 'image',

            # Policy
            'policy_name': 'egreedy',
            'policy_cfg': {
                'epsilon': LinearSchedule(schedule_timesteps=total_steps // 2,
                                          final_p=0.1, initial_p=1)
            },
            # 'policy_name': 'softmax',
            # 'policy_cfg': {
            #     'temperature': LinearSchedule(schedule_timesteps=total_steps,
            #                                   final_p=0.1, initial_p=1)
            # },

            # Replay buffer
            'replay_name': 'per',
            'replay_cfg': {
                'size': int(1e3),
                'batch_size': 10,  # TODO: align with rollout length?
                # Since learning happens on a trajectory of size `batch_size`
                # we want it to be relatively small for n-step returns
                # At the same time, we can still collect the experience in
                # larger chunks
                'alpha': 0.6,
                'beta': LinearSchedule(schedule_timesteps=total_steps,
                                       final_p=0.1, initial_p=1),
                'epsilon': 1e-6
            },
            # 'replay_name': 'er',
            # 'replay_cfg': {
            #     'size': int(1e5),
            #     'batch_size': 10,
            # },

            # Architecture
            'gamma': 0.99,
            'trace_decay': tune.grid_search([0.5]),
            'target_update_freq': 100,
            'double': tune.grid_search([True]),
            'duelling': tune.grid_search([True]),
            "num_atoms": tune.grid_search([1]),
            # "v_min": 0,
            # "v_max": 1,

            # Optimizer a.k.a. Horde
            "horde_fn": create_horde,

            "rollout_fragment_length": 10,

            # --- Evaluation ---
            # "evaluation_interval": 1000,  # per training iteration
            # "custom_eval_function": eval_fn,
            # "evaluation_num_episodes": 1,
            # "evaluation_config": {
            #     'eval_env': env_creator,
            #     'eval_env_config': {},
            # },

            # HACK to get the evaluation through
            # "framework": 'torch',
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
