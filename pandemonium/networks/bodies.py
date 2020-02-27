from torch import nn
from torch.functional import F

from pandemonium.networks.utils import layer_init, conv2d_size_out


class Identity(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


class ConvBody(nn.Module):
    def __init__(self, d: int, w: int, h: int, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv1 = layer_init(nn.Conv2d(d, 8, 2, 1))
        self.conv2 = layer_init(nn.Conv2d(8, 16, 2, 1))
        self.conv3 = layer_init(nn.Conv2d(16, 32, 2, 1))

        w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        self.fc = layer_init(nn.Linear(w * h * 32, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc(y))
        return y


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
