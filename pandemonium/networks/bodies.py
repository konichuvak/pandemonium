import torch
from pandemonium.networks.utils import layer_init, conv2d_size_out
from torch import nn
from torch.functional import F


class Identity(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


class FCBody(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_units: tuple = (64,),
                 activation=nn.ReLU):
        super().__init__()
        dims = (state_dim,) + hidden_units
        self.feature_dim = dims[-1]

        modules = []
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            modules.append(layer_init(nn.Linear(dim_in, dim_out)))
            modules.append(activation())
        self.ff = nn.Sequential(*modules)

    def forward(self, x):
        return self.ff(x)


class ConvBody(nn.Module):
    """ A convolutional neural network, also known as `nature CNN` in RL """

    def __init__(self, d: int, w: int, h: int,
                 feature_dim: int = 256,
                 channels=(8, 16, 32),
                 kernels=(2, 2, 2),
                 strides=(1, 1, 1)):
        assert len(channels) == len(kernels) == len(strides)

        ch = (d,) + channels
        conv = list()
        for i in range(len(channels)):
            l = layer_init(nn.Conv2d(ch[i], ch[i + 1], kernels[i], strides[i]))
            conv += [l, nn.ReLU()]
            w = conv2d_size_out(w, kernels[i], strides[i])
            h = conv2d_size_out(h, kernels[i], strides[i])

        super().__init__()
        self.conv = nn.Sequential(*conv)
        self.fc = layer_init(nn.Linear(w * h * ch[len(channels)], feature_dim))
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x, lstm_state = self.lstm(x, lstm_state)
        return x.squeeze(0)

    @property
    def lstm_state(self):
        return self.hidden_state, self.memory_state

    @lstm_state.setter
    def lstm_state(self, other: Tuple[torch.Tensor, torch.Tensor]):
        self.hidden_state, self.memory_state = other

    @property
    def hidden_state(self):
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, other: torch.Tensor):
        self._hidden_state = other

    @property
    def memory_state(self):
        return self._memory_state

    @memory_state.setter
    def memory_state(self, other: torch.Tensor):
        self._memory_state = other
