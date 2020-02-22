import numpy as np
import torch


def randargmax(b: np.ndarray, rng: np.random.RandomState = None):
    """ A random tie-breaking argmax """
    if rng is None:
        rng = np.random
    return np.argmax(rng.random(b.shape) * (b == b.max()))


def torch_argmax_mask(q: torch.Tensor, dim: int):
    """ Returns a random tie-breaking argmax mask

    Example:
        >>> import torch
        >>> torch.manual_seed(1337)
        >>> q = torch.ones(3, 2)
        >>> torch_argmax_mask(q, 1)
        # tensor([[False,  True],
        #         [ True, False],
        #         [ True, False]])
        >>> torch_argmax_mask(q, 1)
        # tensor([[False,  True],
        #         [False,  True],
        #         [ True, False]])

    """
    rand = torch.rand_like(q)
    if dim == 0:
        mask = rand * (q == q.max(dim)[0])
        mask = mask == mask.max(dim)[0]
        assert int(mask.sum()) == len(q.shape)
    elif dim == 1:
        mask = rand * (q == q.max(dim)[0].unsqueeze(1).expand(q.shape))
        mask = mask == mask.max(dim)[0].unsqueeze(1).expand(q.shape)
        assert int(mask.sum()) == int(q.shape[0])
    else:
        raise NotImplemented("Only vectors and matrices are supported")

    return mask
