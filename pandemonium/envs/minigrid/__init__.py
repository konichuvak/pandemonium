import inspect

import gym_minigrid
from gym_minigrid.envs import MiniGridEnv
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from ray.tune import register_env

from pandemonium.envs.minigrid.four_rooms import FourRooms
from pandemonium.envs.minigrid.plotter import MinigridDisplay
from pandemonium.envs.wrappers import (add_wrappers, Torch, OneHotObsWrapper,
                                       SimplifyActionSpace)

MINIGRIDS = {member[1] for member in inspect.getmembers(gym_minigrid.envs) if
             inspect.isclass(member[1]) and issubclass(member[1], MiniGridEnv)}
MINIGRIDS -= {MiniGridEnv}

encoder_registry = {
    'binary': {
        'encoder_name': 'identity',
        'encoder_cfg': {}
    },
    'image': {
        "encoder_name": 'nature_cnn',
        "encoder_cfg": {
            'feature_dim': 64,
            'channels': (8, 16),
            'kernels': (2, 2),
            'strides': (1, 1),
            'padding': (0, 0),
        },
    },
    'image+text': {

    }
}

wrappers = {
    "OneHotImage": [
        FullyObsWrapper,
        ImgObsWrapper,
        OneHotObsWrapper,
    ],
    "OneHotImageText": [
        FullyObsWrapper,
        OneHotObsWrapper,
    ],
    "ImgOnly": [
        ImgObsWrapper,
    ]
}

for cls in MINIGRIDS:
    name = cls.__name__
    for wrapper_name, wraps in wrappers.items():
        def env_creator(env_cls):
            def env_wrapper(env_config):
                env = env_cls(**env_config)
                env = add_wrappers(base_env=env, wrappers=wraps + [Torch])
                env.unwrapped.max_steps = float('inf')
                return env

            return env_wrapper


        register_env(name=f"MiniGrid-{name}-{wrapper_name}-v0",
                     env_creator=env_creator(cls))
