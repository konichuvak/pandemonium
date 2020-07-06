import torch
from torch import nn

from pandemonium.networks.bodies import BaseNetwork
from pandemonium.networks.utils import layer_init
from pandemonium.utilities.utilities import get_all_classes


class LinearNet(nn.Module):
    """ A fully connected head that follows the network base. """

    def __init__(self, output_dim, body: BaseNetwork):
        super().__init__()
        self.body = body
        self.head = layer_init(nn.Linear(body.feature_dim, output_dim))

    def forward(self, x):
        return self.head(self.body(x))


class Reshape(nn.Module):
    """ A convenience module for reshaping the output at the end of the net. """

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ForwardModel(nn.Module):
    """ Predicts next features, given the current features and an action. """

    def __init__(self, action_dim: int, feature_dim: int):
        super().__init__()
        action_features = feature_dim // 2
        self.action_encoder = nn.Embedding(action_dim, action_features)
        self.head = nn.Sequential(
            nn.Linear(action_features + feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x0, action):
        action = self.action_encoder(action)
        return self.head(torch.cat((x0, action), dim=-1))


class InverseModel(nn.Module):
    """ Predicts an action from a pair of consecutive feature vectors. """

    def __init__(self, action_dim: int, feature_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            layer_init(nn.Linear(feature_dim * 2, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, action_dim))
        )

    def forward(self, x0, x1):
        assert len(x0.shape) == len(x1.shape) == 2
        return self.head(torch.cat((x0, x1), dim=1))


__all__ = get_all_classes(__name__)