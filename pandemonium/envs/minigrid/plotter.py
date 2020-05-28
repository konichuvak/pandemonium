from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Union, NamedTuple, Tuple, Dict, Any
from warnings import warn

import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.subplots as tools
import torch
from gym_minigrid.minigrid import MiniGridEnv

from pandemonium.demons import Demon, ControlDemon
from pandemonium.envs.minigrid.utilities import plotlyfig2json

tools.make_subplots = partial(
    tools.make_subplots,
    horizontal_spacing=0.005,
    vertical_spacing=0.005,
    print_grid=False
)


class Action(NamedTuple):
    name: str
    symbol: str


ACTIONS = {
    0: Action('left', r"$\curvearrowleft$"),
    1: Action('right', r"$\curvearrowright$$"),
    2: Action('forward', r"$\Uparrow$"),
    3: Action('pickup', r"$$"),
    4: Action('drop', r'$$'),
    5: Action('toggle', r'$$'),
    6: Action('done', '$\cross$'),
}

ACTION_NAMES = tuple(a.name for a in ACTIONS.values())


class Directions(IntEnum):
    right = 0
    down = 1
    left = 2
    up = 3


class MinigridDisplay:

    def __init__(self, env: MiniGridEnv):
        self.env = env
        self.env_image = go.Image(z=self.env.render(mode='image'))
        self.all_states = self.generate_all_states(env)

    @staticmethod
    def generate_all_states(env: MiniGridEnv) -> torch.Tensor:
        """ Generates all possible states for a given minigrid.

        Returns
        -------
        A tensor of shape (N, ), where N = w * h - (2w + 2(h-1)).
        Note that N < w * h because we exclude boundaries of the grid.
        """
        states = list()
        for direction in range(4):
            for j in range(1, env.grid.height - 1):
                for i in range(1, env.grid.width - 1):
                    env.grid.set(*np.array((i, j)), None)
                    try:
                        env.place_agent(top=(i, j), size=(1, 1))
                    except TypeError:
                        env.place_agent(i, j, force=True)
                    env.unwrapped.agent_dir = direction
                    obs, reward, done, info = env.step(env.Actions.done)
                    states.append((obs, (direction, i, j)))
        states = torch.stack([s[0] for s in states]).squeeze()
        return states

    def plot_state_values(self,
                          figure_name: str,
                          demon: Demon) -> go.Figure:
        """ Plots state-value function as a heatmap.

        Returns
        -------
        ``go.Figure`` with DIRECTIONS heatmaps as subplots, representing
        state values.
        """
        x = demon.feature(self.all_states)
        v = demon.predict(x)
        v = v.transpose(0, 1).view(4, self.env.height - 2, self.env.width - 2)
        v = torch.nn.ConstantPad2d(1, 0)(v)
        v = v.cpu().detach().numpy()

        assert len(v.shape) == 3

        directions, w, h = v.shape

        fig = tools.make_subplots(
            cols=1, rows=directions, row_titles=[d.name for d in Directions]
        )
        fig = self.remove_tick_labels(fig)

        # Create a heatmap object
        for dir in range(directions):
            values = v[dir]
            heatmap = go.Heatmap(
                z=np.flip(values, axis=0),
                coloraxis='coloraxis',
                name=f'{Directions(dir).name}'
            )
            fig.add_trace(heatmap, col=1, row=dir + 1)

        fig.update_layout(
            title=f'{figure_name}',
            height=900, width=900,
            coloraxis={'colorscale': 'viridis'}
        )
        return fig

    def plot_option_value_distributions(self,
                                        figure_name: str,
                                        z: np.ndarray,
                                        option_ids: Tuple[str] = ACTION_NAMES):
        r"""

        Returns
        -------
        ``go.Figure`` with DIRECTIONS x ACTION_SIZE heatmaps as subplots,
        with a histogram of state-action values in each cell of a heatmap.
        """

        assert len(z.shape) == 5

        options, probs, directions, w, h = z.shape
        assert options == len(option_ids)

        for option in range(options):
            pass

    def plot_option_values_separate(self,
                                    figure_name: str,
                                    demon: ControlDemon,
                                    option_ids: Tuple[str] = ACTION_NAMES
                                    ) -> Dict[str, go.Figure]:
        x = demon.feature(self.all_states)
        q = demon.predict_q(x)
        q = q.transpose(0, 1).view(q.shape[1], 4,
                                   self.env.height - 2,
                                   self.env.width - 2)
        q = torch.nn.ConstantPad2d(1, 0)(q)
        q = q.cpu().detach().numpy()

        assert len(q.shape) == 4

        options, directions, w, h = q.shape
        assert options == len(option_ids)

        tiles = {
            Directions.up: (1, 2),
            Directions.down: (3, 2),
            Directions.left: (2, 1),
            Directions.right: (2, 3),
        }

        figures = dict()

        for option, option_id in zip(range(options), option_ids):
            fig = tools.make_subplots(cols=3, rows=3)
            fig = self.remove_tick_labels(fig)

            for dir in range(directions):
                values = q[option, dir]
                heatmap = go.Heatmap(
                    z=np.flip(values, axis=0),
                    coloraxis='coloraxis',
                    name=f'{option_id}, {Directions(dir).name}'
                )
                fig.add_trace(heatmap, *tiles[dir])

            # Add the env image in the middle
            fig.add_trace(self.env_image, 2, 2)

            fig.update_layout(
                title=f'{figure_name}',
                height=900, width=900,
                coloraxis={'colorscale': 'inferno'}
            )

            figures[option_id] = fig

        return figures

    def plot_option_values(self,
                           figure_name: str,
                           demon: ControlDemon,
                           option_ids: Tuple[str] = ACTION_NAMES
                           ) -> go.Figure:
        r""" Plots option(action)-values of an agent.

        Returns
        -------
        ``go.Figure`` with DIRECTIONS x ACTION_SIZE heatmaps as subplots,
        representing values for each option
        """

        x = demon.feature(self.all_states)
        q = demon.predict_q(x)
        q = q.transpose(0, 1).view(q.shape[1], 4,
                                   self.env.height - 2,
                                   self.env.width - 2)
        q = torch.nn.ConstantPad2d(1, 0)(q)
        q = q.cpu().detach().numpy()

        assert len(q.shape) == 4

        options, directions, w, h = q.shape
        assert options == len(option_ids)

        fig = tools.make_subplots(
            cols=options, column_titles=option_ids,
            rows=directions, row_titles=[d.name for d in Directions]
        )
        fig = self.remove_tick_labels(fig)

        # Create a heatmap object
        for option, option_id in zip(range(options), option_ids):
            for dir in range(directions):
                values = q[option, dir]
                heatmap = go.Heatmap(
                    z=np.flip(values, axis=0),
                    coloraxis='coloraxis',
                    name=f'{option_id}, {Directions(dir).name}'
                )
                fig.add_trace(heatmap, col=option + 1, row=dir + 1)

        fig.update_layout(
            title=f'{figure_name}',
            height=900, width=1900,
            coloraxis={'colorscale': 'viridis'}
        )
        return fig

    def plot_option_continuation(self,
                                 figure_name: str,
                                 beta: np.ndarray,
                                 option_ids: Tuple[str]
                                 ) -> go.Figure:
        assert len(beta.shape) == 4

        options, directions, w, h = beta.shape
        assert options == len(option_ids)

        fig = tools.make_subplots(
            cols=options, column_titles=option_ids,
            rows=directions, row_titles=[d.name for d in Directions]
        )
        fig = self.remove_tick_labels(fig)

        # Create a heatmap object
        for option, option_id in zip(range(options), option_ids):
            for dir in range(directions):
                values = beta[option, dir]
                heatmap = go.Heatmap(
                    z=np.flip(values, axis=0),
                    coloraxis='coloraxis',
                    name=f'{option_id}, {Directions(dir).name}'
                )
                fig.add_trace(heatmap, col=option + 1, row=dir + 1)

        fig.update_layout(
            title=f'{figure_name}',
            height=900, width=1900,
            coloraxis={'colorscale': 'viridis'}
        )
        return fig

    def plot_option_action_values(self,
                                  figure_name: str,
                                  pi: np.ndarray,
                                  ) -> go.Figure:
        assert len(pi.shape) == 5

        options, actions, directions, w, h = pi.shape

        figures = []
        for option in range(options):
            fig = tools.make_subplots(
                cols=actions, column_titles=ACTION_NAMES,
                rows=directions, row_titles=[d.name for d in Directions]
            )
            fig = self.remove_tick_labels(fig)

            # Create a heatmap object
            for action in range(actions):
                for dir in range(directions):
                    values = pi[option, action, dir]
                    heatmap = go.Heatmap(
                        z=np.flip(values, axis=0),
                        coloraxis='coloraxis',
                        name=f'{ACTIONS[action].name}, {Directions(dir).name}'
                    )
                    fig.add_trace(heatmap, col=action + 1, row=dir + 1)

            fig.update_layout(
                title=f'{figure_name}_option{option}',
                height=900, width=1900,
                coloraxis={'colorscale': 'viridis'}
            )
            figures.append(fig)
        return figures

    @staticmethod
    def save_figure(fig: Union[Dict[Any, go.Figure], go.Figure],
                    save_path: Union[Path, str],
                    json: bool = True,
                    auto_open: bool = False,
                    html: bool = False,
                    ):
        if json:
            plotlyfig2json(fig, fpath=f'{save_path}.json')
        if html:
            if isinstance(fig, go.Figure):
                plotly.offline.plot(fig,
                                    filename=f'{save_path}.html',
                                    include_mathjax='cdn', auto_open=auto_open)
            else:
                warn('Cannot save a collection of figures as html')

    @staticmethod
    def _normalize_matrix(m):
        vmin, vmax = np.min(m), np.max(m)
        m = (m - vmin) / (vmax - vmin)
        return m

    @staticmethod
    def remove_tick_labels(fig: go.Figure) -> go.Figure:
        """ Removes all the tick labels from heatmaps """
        axis_template = dict(showgrid=False, zeroline=False,
                             showticklabels=False, ticks='')
        for prop in fig.layout._props:
            if 'axis' in prop:
                fig['layout'][prop].update(axis_template)
        return fig

    def _get_env_grid(self):
        """ """
        full_grid = self.env.unwrapped.grid.encode()[:, :, 0]
        full_grid[full_grid == 4] = 1  # remove the doors
        full_grid[full_grid == 5] = 1  # remove the keys
        full_grid[full_grid == 6] = 1  # remove the balls
        full_grid[full_grid == 7] = 1  # remove the boxes
        full_grid[full_grid == 8] = 1  # remove the goal state
        full_grid[full_grid == 9] = 1  # remove the lava
        return full_grid
