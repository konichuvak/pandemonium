from typing import Tuple, Dict

import gym
from torch.distributions import Distribution

PolicyInfo = Dict[str, Distribution]


class Policy:
    r""" Base abstract class for decision making rules """

    def __init__(self, action_space: gym.spaces.Space):
        self.action_space = action_space

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    def dist(self, *args, **kwargs) -> Distribution:
        """ Returns a distribution over actions """
        raise NotImplementedError

    def act(self, *args, **kwargs) -> Tuple['Action', PolicyInfo]:
        dist = self.dist(*args, **kwargs)
        return dist.sample(), {'action_dist': dist}

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
