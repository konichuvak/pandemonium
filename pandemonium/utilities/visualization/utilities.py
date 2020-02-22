import json

import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder


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
