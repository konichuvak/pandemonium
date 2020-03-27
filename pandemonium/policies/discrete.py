from typing import Union

import gym
import torch
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule
from torch.distributions import Categorical, Distribution

from pandemonium.policies import Policy, torch_argmax_mask
from pandemonium.utilities.spaces import OptionSpace
from pandemonium.utilities.utilities import get_all_classes

Schedule = Union[ConstantSchedule, LinearSchedule]


class Discrete(Policy):
    """ Base class for discrete policies """

    def __init__(self, action_space):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError()
        super().__init__(action_space=action_space)

    def dist(self, *args, **kwargs) -> Distribution:
        raise NotImplementedError


class Random(Discrete):
    """ Picks an option at random """

    def dist(self, *args, **kwargs):
        return self.action_space

    def __str__(self):
        return f'RandomPolicy({self.action_space.n})'


class Egreedy(Discrete):
    r""" :math:`\epsilon`-greedy policy for discrete action spaces.

    Picks the optimal action wrt to Q with probability 1 - :math:`\epsilon`
    """

    def __init__(self, epsilon: Schedule, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ε = epsilon
        self.t = 0

    @property
    def epsilon(self) -> float:
        self.t += 1
        return self.ε.value(self.t)

    @epsilon.setter
    def epsilon(self, schedule: Schedule):
        self.t = 0
        self.ε = schedule

    def dist(self, features, vf) -> Categorical:
        """ Creates a Categorical distribution with :math:`\epsilon`-greedy support

        Assumes that Q-values are of shape (batch, actions, states)
        """
        ε, q = self.epsilon, vf(features)
        assert len(q.shape) > 1
        assert q.shape[1] == self.action_space.n  # (batch, action, ...)
        probs = torch.empty_like(q).fill_(ε / (self.action_space.n - 1))
        probs[torch_argmax_mask(q, 1)] = 1 - ε
        return Categorical(probs=probs)

    def act(self, *args, **kwargs):
        dist = self.dist(*args, **kwargs)
        return dist.sample(), {'entropy': dist.entropy(),
                               'epsilon': self.ε.value(self.t)}

    def __str__(self):
        return f'ε-greedy({self.ε})'


class SoftmaxPolicy(Discrete):

    def __init__(self, temperature: Schedule, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.τ = temperature
        self.t = 0

    @property
    def temperature(self) -> float:
        self.t += 1
        return self.τ.value(self.t)

    @temperature.setter
    def temperature(self, schedule: Schedule):
        self.t = 0
        self.τ = schedule

    def dist(self, features, vf) -> Categorical:
        probs = (vf(features) / self.temperature).softmax(dim=2)
        return Categorical(probs=probs)

    def act(self, *args, **kwargs):
        dist = self.dist(*args, **kwargs)
        return dist.sample(), {'entropy': dist.entropy(),
                               'temperature': self.τ.value(self.t)}

    def __str__(self):
        return f'Softmax({self.τ})'


class HierarchicalPolicy(Discrete):
    """ A decision rule for discrete option spaces.

    In order to produce an action $a$ this policy picks an option $ω$ from
    the space of available options $Ω$ first. To pick an option, it queries
    initiation set $I$ of each of the available options, picking the one that
    has the highest score. It then it uses the internal policy $π$ of the
    chosen option to produce the action that will be made by the agent.


    .. todo::
        Currently starts with a random option from the space.
        Maybe wait for initial state to initialize the option instead, then
        pick the one with the highest `interest`

    .. todo::
        explore the interplay between initiation of one option and termination
        of another

    .. todo::
        might be better to move the `OptionSpace` into this file since it is
        discrete at the moment
    """

    def __init__(self, option_space: OptionSpace):
        super().__init__(action_space=option_space)
        self.Ω = self.option_space = self.action_space
        self.idx = torch.randint(0, len(self.Ω), (1,)).item()
        self.ω = self.Ω[self.idx]

    def dist(self, *args, **kwargs) -> Distribution:
        raise NotImplementedError

    def act(self, state, vf):
        β = self.ω.continuation(state)
        if round(β.item()) == 0:
            # TODO: initiation set check
            option_dist = self.dist(state, vf)
            self.idx = option_dist.sample().item()
            self.ω = self.Ω[self.idx]
        action_dist = self.ω.policy.dist(state)
        info = {'beta': β, 'entropy': action_dist.entropy(), 'option': self.idx}
        return action_dist.sample().item(), info


class EgreedyOverOptions(Egreedy, HierarchicalPolicy):
    r""" :math:`\epsilon`-greedy policy over options.

    Picks the optimal option wrt to Q with probability 1 - :math:`\epsilon`.
    """


__all__ = get_all_classes(__name__)
