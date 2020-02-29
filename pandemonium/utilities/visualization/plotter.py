from enum import IntEnum
from pathlib import Path
from typing import Union, NamedTuple, Tuple

import numpy as np
import plotly
import plotly.graph_objs as go
from gym_minigrid.minigrid import MiniGridEnv
from pandemonium.utilities.visualization import tools
from pandemonium.utilities.visualization.utilities import plotlyfig2json


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


class Plotter:

    def __init__(self, env: MiniGridEnv):
        self.env = env
        self.env_img = self.env.render(close=True)

    def plot_env(self) -> go.Image:
        return go.Image(z=self.env_img)

    def plot_option_value_function(self,
                                   figure_name: str,
                                   q: np.ndarray,
                                   option_ids: Tuple[str] = ACTION_NAMES
                                   ) -> go.Figure:
        """ Plots V (Q) of an agent associated with a particular mini-grid.

        Creates a figure with DIRECTIONS x ACTION_SIZE subplots.
        """
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
                fig.append_trace(heatmap, col=option + 1, row=dir + 1)

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
                fig.append_trace(heatmap, col=option + 1, row=dir + 1)

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
                    fig.append_trace(heatmap, col=action + 1, row=dir + 1)

            fig.update_layout(
                title=f'{figure_name}_option{option}',
                height=900, width=1900,
                coloraxis={'colorscale': 'viridis'}
            )
            figures.append(fig)
        return figures

    @staticmethod
    def save_figure(fig: go.Figure,
                    save_path: Union[Path, str],
                    auto_open: bool = False,
                    save_json: bool = False):
        plotly.offline.plot(fig,
                            filename=f'{save_path}.html',
                            include_mathjax='cdn', auto_open=auto_open)
        if save_json:
            plotlyfig2json(fig=fig, fpath=f'{save_path}.json')

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
