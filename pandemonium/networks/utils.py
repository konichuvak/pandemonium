import torch


def conv2d_size_out(size, kernel_size=2, stride=1, padding=0):
    """ Determines the dimensionality (w or h) of the 2d convolutional layer """
    return (size - kernel_size + 2 * padding) // stride + 1


def deconv2d_size_out(size, kernel_size, stride, padding: int = 0):
    return stride * (size - 1) + kernel_size - 2 * padding


def layer_init(layer, scaling_factor: float = 1.0):
    torch.nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(scaling_factor)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias.data, 0)
    return layer
