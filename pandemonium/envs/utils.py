from functools import reduce
from typing import List, Type

import gym
import numpy as np
from gym import ObservationWrapper
from gym_minigrid.minigrid import MiniGridEnv


def generate_all_states(env: MiniGridEnv, wrappers: List[Type[gym.Wrapper]]):
    """ Generates all possible states for a given minigrid """
    states = list()
    for direction in range(4):
        for j in range(env.grid.height):
            for i in range(env.grid.width):
                env.grid.set(*np.array((i, j)), None)
                try:
                    env.place_agent(top=(i, j), size=(1, 1))
                except TypeError:
                    env.place_agent(i, j, force=True)
                env.unwrapped.agent_dir = direction

                # Obtain observation by sequentially applying all the wrappers
                #   on top of the original observation.
                obs = env.gen_obs()
                for i, wrapper in zip(range(len(wrappers) - 1, -1, -1),
                                      wrappers):
                    if not isinstance(wrapper(env.unwrapped),
                                      ObservationWrapper):
                        continue
                    wrapped = reduce(lambda wrap, _: wrap.env, range(i),
                                     env)
                    obs = wrapped.observation(obs)
                states.append((obs, (direction, i, j)))

    return states
