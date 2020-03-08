import json
from functools import reduce
from typing import List, Type

import gym
import numpy as np
import plotly.graph_objs as go
from gym import ObservationWrapper
from gym_minigrid.minigrid import MiniGridEnv
from plotly.utils import PlotlyJSONEncoder


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


def plotlyfig2json(fig, fpath):
    """
    Serialize a plotly figure object to JSON so it can be persisted to disk.
    Figures persisted as JSON can be rebuilt using the plotly JSON chart API:

    http://help.plot.ly/json-chart-schema/

    If `fpath` is provided, JSON is written to file.

    Modified from https://github.com/nteract/nteract/issues/1229
    """

    redata = json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder))
    relayout = json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))

    with open(fpath, 'w') as f:
        json.dump({'data': redata, 'layout': relayout}, f)


def plotlyfromjson(fpath):
    """Render a plotly figure from a json file"""
    with open(fpath, 'r') as f:
        v = json.loads(f.read())

    fig = go.Figure(data=v['data'], layout=v['layout'])
    return fig
