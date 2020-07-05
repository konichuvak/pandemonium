from functools import reduce
from typing import List, Type

import torch
from gym import Env
from gym.core import ObservationWrapper, Wrapper

from pandemonium.envs.minigrid.wrappers import *


def add_wrappers(base_env: Env, wrappers: List[Type[Wrapper]]):
    """Returns an environment wrapped with wrappers """
    return reduce(lambda e, wrapper: wrapper(e), wrappers, base_env)


class Torch(ObservationWrapper):

    def __init__(self, *args, device: torch.device = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = 'cpu'
        # self.device = device if device is not None else torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")

    def observation(self, obs):
        t = torch.tensor(obs, device=self.device, dtype=torch.float32)
        if len(t.shape) == 3:
            t = t.permute(2, 0, 1)  # swap (h, w, c) -> (c, h, w)
        t = t.unsqueeze(0)  # add batch dim: (c, h, w) -> (1, c, h, w)
        return t


class Scaler(Torch):
    def __init__(self, coef: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coef = coef

    def observation(self, obs):
        return self.coef * obs


class ImageNormalizer(Scaler):
    def __init__(self, env):
        super().__init__(coef=1 / 255, env=env)
