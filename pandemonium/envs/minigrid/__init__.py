from gym_minigrid.envs import *
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from ray.tune import register_env

from pandemonium.envs.minigrid.four_rooms import FourRooms
from pandemonium.envs.minigrid.plotter import MinigridDisplay
from pandemonium.envs.wrappers import (add_wrappers, Torch, OneHotObsWrapper,
                                       SimplifyActionSpace)

wrappers = {
    "Simple": [
        SimplifyActionSpace,
        FullyObsWrapper,
        ImgObsWrapper,
        OneHotObsWrapper,
    ],
    "ImgOnly": [
        ImgObsWrapper,
    ]
}

for cls in (EmptyEnv, MultiRoomEnv, FourRoomsEnv):
    name = cls.__name__
    for wrap_name, wraps in wrappers.items():
        def env_creator(env_config):
            env = cls(**env_config)
            env = add_wrappers(base_env=env, wrappers=wraps + [Torch])
            env.unwrapped.max_steps = float('inf')
            return env


        register_env(f"MiniGrid-{name}-{wrap_name}-v0", env_creator)
