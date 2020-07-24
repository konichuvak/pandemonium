import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from experiments import EXPERIMENT_DIR
from experiments.trainable import Trainer
from pandemonium.implementations.ride import create_horde

# TODO: unroll length vs batch size

search_space = {
    # "entropy_coefficient": [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
    # "ir_weight": [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
    # "beta": [0.2, 0.5],
    # "rollout_fragment_length": [16, 32],
    "entropy_coefficient": [0.01],
    "ir_weight": [1.0],
    "beta": [0.2],
    "rollout_fragment_length": [16, 32],
}

if __name__ == "__main__":
    ray.init(
        local_mode=False,
        # dashboard_port=8267
    )
    analysis = tune.run(
        run_or_experiment=Trainer,
        local_dir=EXPERIMENT_DIR,
        name='RIDE_pbt',
        verbose=1,
        # num_samples=1,
        stop={"timesteps_total": int(7e7)},
        scheduler=PopulationBasedTraining(
            # time_attr="training_iteration",
            # metric="episode_reward",
            metric="episodes_total",
            mode="max",
            time_attr="time_total_s",
            perturbation_interval=1,
            hyperparam_mutations=search_space,
        ),
        config={
            **{k: tune.grid_search(v) for k, v in search_space.items()},

            # Environment for the agent to interact with
            'env': "MiniGrid-MultiRoomEnvN4S5-Img-v0",
            # "env": tune.grid_search([
            #     "MiniGrid-MultiRoomEnvN6-Img-v0",
            #     "MiniGrid-MultiRoomEnvN4S5-Img-v0",
            #     "MiniGrid-MultiRoomEnvN2S4-Img-v0",
            # ]),
            # Environment discounting factor
            # 'gamma': tune.grid_search([0.99]),
            # 'gamma': 0.99,
            # Trajectory length on which the agent will learn
            # "rollout_fragment_length": 16,
            # A function that creates the Horde
            "horde_fn": create_horde,
            # Feature generator for the agent
            'encoder': 'image',
            # Policy configuration
            "policy_name": 'VPG',

            # The rate at which trace decays (1 is equivalent to n-step TD)
            # 'trace_decay': tune.grid_search([1]),
            # 'trace_decay': 1,
            # The parameter that trades forward model loss against the loss
            # of the inverse model
            # 'beta': tune.grid_search([0.2, 0.5]),
            # 'beta': 0.2,
            # Weight of the loss from intrinsic curiosity module to be used
            # when calculating the total loss
            # 'icm_weight': tune.grid_search([1]),
            # 'icm_weight': 1.,
            # Intrinsic reward weight to use when combined with extrinsic reward
            # 'ir_weight': tune.grid_search([0.1]),
            # 'ir_weight': 0.1,
        }
    )
