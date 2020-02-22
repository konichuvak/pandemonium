from typing import Union

import gym
import numpy as np


class Policy:
    r"""

    .. todo::
        consider making an ``option space``

    """

    def __init__(self,
                 action_space: gym.spaces.Discrete,
                 rng: np.random.RandomState = np.random.RandomState(1337)):
        self.action_space = action_space
        self.sampling_space = np.arange(self.action_space.n)
        self.rng = rng

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    def dist(self, *args, **kwargs):
        """ Returns a distribution over actions """
        raise NotImplementedError

    def act(self, x, *args, **kwargs) -> Union['Option', 'Action']:
        prob = self.dist(x, *args, **kwargs)
        return self.rng.choice(self.sampling_space, p=prob)

    def action_filter(self, x):
        """ Filters the actions available at a given state

        Closely relates to option initiation sets:
            [o for o in options if o.initiation(x) == 1]

        .. seealso:: Interest Functions by K. Khetarpal et al. 2020
            https://arxiv.org/pdf/2001.00271.pdf

        .. seealso:: Ray RLlib has a working implementation
            https://ray.readthedocs.io/en/latest/rllib-models.html#variable-length-parametric-action-spaces
        """
        return self.action_space
