from pathlib import Path

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from experiments import LIB_ROOT
from pandemonium.envs.minigrid.plotter import ACTIONS, ACTION_NAMES
from pandemonium.envs.minigrid.utilities import plotlyfromjson

trial = 'Loop_env_0_size=5,trace_decay=0_2020-06-06_21-56-04vv88k73u'
experiment = 'DistributionalTD'

LOGDIR = Path(f'experiments/tune/{experiment}')
LOGDIR = LIB_ROOT / LOGDIR / trial

# TODO: a global dropdown to choose from all the experiments in C51 dir


CACHE = {
    file.stem.split('_')[0]: plotlyfromjson(file)
    for file in LOGDIR.glob('*_qf.json')
}
ITER_LOOKUP = dict(enumerate(sorted(CACHE)))
ACTION_LOOKUP = dict(enumerate(ACTION_NAMES))

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            children=[
                dcc.Slider(
                    id='iteration-slider',
                    min=0,
                    max=len(CACHE) - 1,
                    value=0,
                    marks=ITER_LOOKUP,
                    step=None
                )
            ],
            style={'width': '50%'}
        ),
        html.Div(
            children=[
                dcc.Slider(
                    id='action-slider',
                    min=0,
                    max=len(ACTIONS) - 1,
                    value=0,
                    marks=ACTION_LOOKUP,
                    step=None
                )
            ],
            style={'width': '50%'}
        ),
        html.Div(
            id='main-plot',
            children=[dcc.Graph(id='plot')]
        )
    ]
)


@app.callback(
    [Output('plot', 'figure')],
    [
        Input('iteration-slider', 'value'),
        Input('action-slider', 'value'),
    ],
)
def action_selection_callback(iteration, action):
    figure = CACHE.get(ITER_LOOKUP[iteration])
    if isinstance(figure, go.Figure):
        return [figure]
    elif isinstance(figure, dict):
        return [figure[ACTION_LOOKUP[action]]]

    return []


if __name__ == '__main__':
    app.run_server(debug=True)
