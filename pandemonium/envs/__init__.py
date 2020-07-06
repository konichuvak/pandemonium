from typing import Union

from pandemonium.envs.minigrid import *
from pandemonium.envs.dm_lab.dm_env import DeepmindLabEnv

PandemoniumEnv = Union[MiniGridEnv, DeepmindLabEnv]
