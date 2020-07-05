from ray.tune import register_env
from torch import device

from pandemonium.envs import Torch
from pandemonium.envs.dm_lab.dm_env import DeepmindLabEnv

register_env("DeepmindLabEnv", lambda config: Torch(DeepmindLabEnv(**config),
                                                    device=device('cpu')))

encoder_registry = {
    'nature_cnn_3': {
        "encoder_name": 'nature_cnn',
        "encoder_cfg": {
            'feature_dim': 512,
            'channels': (16, 32, 64),
            'kernels': (8, 4, 2),
            'strides': (4, 2, 1),
            'padding': (0, 0, 0),
        },
    },
    'nature_cnn_2': {
        "encoder_name": 'nature_cnn',
        "encoder_cfg": {
            'feature_dim': 512,
            'channels': (16, 32),
            'kernels': (8, 4),
            'strides': (4, 2),
            'padding': (0, 0)
        },
    },
}
