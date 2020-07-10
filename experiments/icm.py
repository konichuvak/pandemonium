import ray
from ray import tune

from experiments import EXPERIMENT_DIR
from experiments.trainable import Loop
from pandemonium.implementations.icm import create_horde

if __name__ == "__main__":
    ray.init(
        local_mode=False,
        dashboard_port=8268
    )
    analysis = tune.run(
        Loop,
        name='ICM',
        verbose=1,
        stop={"timesteps_total": int(1e6)},
        num_samples=6,
        local_dir=EXPERIMENT_DIR,
        config={
            "env": "MiniGrid-MultiRoomEnvN4S5-Img-v0",
            'encoder': 'image',
            "policy_name": 'VPG',
            "policy_cfg": {'entropy_coefficient': tune.grid_search([0.01])},
            'gamma': tune.grid_search([0.9]),
            'beta': tune.grid_search([0.2]),
            'trace_decay': tune.grid_search([1]),
            'icm_weight': tune.grid_search([0, 1]),
            "rollout_fragment_length": 16,  # batch size for exp collector
            "horde_fn": create_horde,
        }
    )
