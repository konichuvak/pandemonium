from torch import nn

from pandemonium.networks.utils import layer_init


class LinearNet(nn.Module):
    def __init__(self, output_dim, body):
        super().__init__()
        self.body = body
        self.head = layer_init(nn.Linear(body.feature_dim, output_dim))

    def forward(self, x):
        return self.head(self.body(x))
