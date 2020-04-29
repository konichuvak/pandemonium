from typing import Tuple, Dict

import gym
from torch.distributions import Distribution

from pandemonium.utilities.registrable import Registrable

PolicyInfo = Dict[str, Distribution]


class Policy(Registrable):
    r""" Base abstract class for decision making rules

    Mathematically, a policy is a function $\pi: \mathcal{X} -> \mathcal{A}$
    that maps the space of features onto the space of actions.
    """

    def __init__(self,
                 feature_dim: int,
                 action_space: gym.spaces.Space,
                 **params):
        """

        Parameters
        ----------
        feature_dim: int
            dimensionality of the feature vector (domain of the function)
        action_space: gym.spaces.Space
            (range of the functions)
        **params:
            any additional parameters required to initialize the Policy
        """
        self.feature_dim = feature_dim
        self.action_space = action_space

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    def dist(self, *args, **kwargs) -> Distribution:
        """ Produces a distribution over actions """
        raise NotImplementedError

    def act(self, *args, **kwargs) -> Tuple['Action', PolicyInfo]:
        """ Samples an action from a distribution over actions """
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
