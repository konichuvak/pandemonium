from typing import Union

import gym
import torch
from pandemonium.policies import Policy, torch_argmax_mask
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule
from torch.distributions import Categorical

Schedule = Union[ConstantSchedule, LinearSchedule]


class Discrete(Policy):
    """ Base class for discrete policies """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError()


class Random(Discrete):
    """ Picks an option at random """

    def dist(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return f'RandomPolicy({self.action_space.n})'


class Egreedy(Discrete):
    """ Picks the optimal option wrt to Q with probability 1 - ε """

    def __init__(self, epsilon: Schedule, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._epsilon = epsilon
        self.t = 0
        self.n = self.action_space.n

    @property
    def epsilon(self) -> float:
        self.t += 1
        return self._epsilon.value(self.t)

    @epsilon.setter
    def epsilon(self, schedule: Schedule):
        self.t = 0
        self._epsilon = schedule

    def dist(self, state, vf, *args, **kwargs) -> Categorical:
        ε, q = self.epsilon, vf(state)
        probs = torch.empty_like(q).fill_(ε / (self.n - 1))
        probs[torch_argmax_mask(q, len(q.shape) - 1)] = 1 - ε
        return Categorical(probs=probs)

    @torch.no_grad()
    def act(self, state, *args, **kwargs) -> 'Action':
        dist = self.dist(state, *args, **kwargs)
        return dist.sample()

    def __str__(self):
        return f'ε-greedy({self._epsilon})'


class Boltzman(Discrete):
    pass
