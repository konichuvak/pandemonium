from typing import Tuple, Optional

import torch
from torch import nn
from torch.functional import F

from pandemonium.networks.utils import layer_init, conv2d_size_out
from pandemonium.utilities.registrable import Registrable


class BaseNetwork(nn.Module, Registrable):
    """ ABC for all networks that allows for registration

    Attributes
    ----------
    feature_dim: tuple
        dimensions of the input to the network

    """

    def __init__(self, obs_shape: tuple, **kwargs):
        super().__init__()


@BaseNetwork.register('identity')
class Identity(BaseNetwork):
    def __init__(self, obs_shape: tuple):
        super().__init__(obs_shape)
        self.feature_dim = obs_shape

    def forward(self, x):
        return x


@BaseNetwork.register('fc_body')
class FCBody(BaseNetwork):
    def __init__(self,
                 obs_shape: tuple,
                 hidden_units: tuple = (64,),
                 activation=nn.ReLU):
        super().__init__(obs_shape)
        dims = obs_shape + hidden_units
        modules = list()
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            modules.append(layer_init(nn.Linear(dim_in, dim_out)))
            modules.append(activation())
        self.ff = nn.Sequential(*modules)
        self.feature_dim = hidden_units[-1]

    def forward(self, x):
        return self.ff(x)


@BaseNetwork.register('conv_body')
class ConvBody(BaseNetwork):
    """ A convolutional neural network, also known as `nature CNN` in RL """

    def __init__(self,
                 obs_shape: tuple,
                 feature_dim: int = 256,
                 channels=(8, 16, 32),
                 kernels=(2, 2, 2),
                 strides=(1, 1, 1)):
        assert len(channels) == len(kernels) == len(strides)
        assert len(obs_shape) == 3, obs_shape

        d, w, h = obs_shape
        ch = (d,) + channels
        conv = list()
        for i in range(len(channels)):
            l = layer_init(nn.Conv2d(ch[i], ch[i + 1], kernels[i], strides[i]))
            conv += [l, nn.ReLU()]
            w = conv2d_size_out(w, kernels[i], strides[i])
            h = conv2d_size_out(h, kernels[i], strides[i])

        super().__init__(obs_shape)
        self.conv = nn.Sequential(*conv)
        self.fc = layer_init(nn.Linear(w * h * ch[len(channels)], feature_dim))
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


@BaseNetwork.register('conv_lstm')
class ConvLSTM(ConvBody):
    """ A convolutional neural net with a recurrent LSTM layer on top

    Used for tackling partial observability in the environment.
    A comprehensive example is given in https://arxiv.org/pdf/1507.06527.pdf
    `Deep Recurrent Q-Learning for Partially Observable MDPs`.
    """

    def __init__(self,
                 hidden_units: int = 256,
                 lstm_layers: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=hidden_units,
                            num_layers=lstm_layers)
        self._hidden_state = torch.zeros(1, 1, self.feature_dim)
        self._memory_state = torch.zeros(1, 1, self.feature_dim)

    def forward(self,
                x: torch.Tensor,
                lstm_state: Optional[tuple] = None):
        x = ConvBody.forward(self, x).unsqueeze(0)
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
