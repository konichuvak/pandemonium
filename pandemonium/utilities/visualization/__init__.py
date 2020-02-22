from functools import partial

import plotly.subplots as tools

from pandemonium.utilities.visualization.plotter import PlotterOneHot

tools.make_subplots = partial(
    tools.make_subplots,
    horizontal_spacing=0.005,
    vertical_spacing=0.005,
    print_grid=False
)
