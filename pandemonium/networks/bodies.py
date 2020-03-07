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
    def __init__(self, d: int, w: int, h: int,
                 feature_dim: int = 256,
                 channels=(8, 16, 32),
                 kernels=(2, 2, 2),
                 strides=(1, 1, 1)):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv1 = layer_init(nn.Conv2d(
            d, channels[0], kernels[1], strides[0]
        ))
        self.conv2 = layer_init(nn.Conv2d(
            channels[0], channels[1], kernels[1], strides[1]
        ))
        self.conv3 = layer_init(nn.Conv2d(
            channels[1], channels[2], kernels[2], strides[2]
        ))

        for i in range(3):
            w = conv2d_size_out(w, kernels[i], strides[i])
            h = conv2d_size_out(h, kernels[i], strides[i])

        self.fc = layer_init(nn.Linear(w * h * channels[2], self.feature_dim))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class ConvLSTM(nn.Module):
    def __init__(self, d: int, w: int, h: int, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv1 = layer_init(nn.Conv2d(d, 16, 8, 4))
        self.conv2 = layer_init(nn.Conv2d(16, 32, 4, 2))

        dim = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)

        self.fc = layer_init(nn.Linear(dim ** 2 * 32, self.feature_dim))

        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)

    def forward(self, x: torch.Tensor, last_action_reward, lstm_state):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x, lstm_state = self.lstm(x, lstm_state)
        return x.squeeze(0), lstm_state
