from typing import Union

import gym
import numpy as np
from torch.distributions import Distribution


class Policy:
    r"""

    .. todo::
        consider making an ``option space``

    """

    def __init__(self,
                 action_space: gym.spaces.Discrete,
                 rng: np.random.RandomState = np.random.RandomState(1337)):
        self.action_space = action_space
        self.rng = rng

    def dist(self, *args, **kwargs) -> Distribution:
        """ Returns a distribution over actions """
        raise NotImplementedError

    def act(self, state, *args, **kwargs) -> Union['Option', 'Action']:
        return self.dist(state, *args, **kwargs).sample()

    def action_filter(self, state):
        """ Filters the actions available at a given state

        Closely relates to option initiation sets:
        >>> [o for o in options if o.initiation(state) == 1]

        .. seealso:: Interest Functions by K. Khetarpal et al. 2020
            https://arxiv.org/pdf/2001.00271.pdf

        .. seealso:: Ray RLlib has a working implementation
            https://ray.readthedocs.io/en/latest/rllib-models.html#variable-length-parametric-action-spaces
        """
        return self.action_space
