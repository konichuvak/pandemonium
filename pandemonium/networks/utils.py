import torch


def conv2d_size_out(size, kernel_size=2, stride=1):
    """ Determines the dimensionality (w or h) of the 2d convolutional layer """
    return (size - (kernel_size - 1) - 1) // stride + 1


def layer_init(layer, scaling_factor: float = 1.0):
    torch.nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(scaling_factor)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias.data, 0)
    return layer
