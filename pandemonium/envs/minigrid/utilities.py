import json
from typing import Union, Dict, Any

import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder


def plotlyfig2json(fig: Union[Dict[Any, go.Figure], go.Figure], fpath: str):
    """
    Serialize a plotly figure object to JSON so it can be persisted to disk.
    Figures persisted as JSON can be rebuilt using the plotly JSON chart API:

    http://help.plot.ly/json-chart-schema/

    If `fpath` is provided, JSON is written to file.

    Modified from https://github.com/nteract/nteract/issues/1229
    """

    if isinstance(fig, dict):
        figure = fig.copy()
        json_fig = {
            fig_name: {
                'data': json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder)),
                'layout': json.loads(
                    json.dumps(fig.layout, cls=PlotlyJSONEncoder))
            }
            for fig_name, fig in figure.items()
        }
    else:
        json_fig = {
            'data': json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder)),
            'layout': json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))
        }

    with open(fpath, 'w') as f:
        json.dump(json_fig, f)


def plotlyfromjson(fpath) -> Union[Dict[Any, go.Figure], go.Figure]:
    """Render a plotly figure from a json file"""
    with open(fpath, 'r') as f:
        fig = json.loads(f.read())

    if isinstance(fig, dict):
        figure = fig.copy()
        return {fig_name: go.Figure(data=fig['data'], layout=fig['layout'])
                for fig_name, fig in figure.items()}
    else:
        return go.Figure(data=fig['data'], layout=fig['layout'])
