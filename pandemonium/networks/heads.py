from pandemonium.networks.utils import layer_init
from torch import nn


class LinearNet(nn.Module):
    def __init__(self, output_dim, body):
        super().__init__()
        self.body = body
        self.head = layer_init(nn.Linear(body.feature_dim, output_dim))

    def forward(self, x):
        return self.head(self.body(x))


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
