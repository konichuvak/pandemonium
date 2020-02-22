from typing import Union

import torch
from pandemonium.experience import Trajectory, Transition
from pandemonium.utilities.utilities import get_all_classes


class ContinuationFunction:
    r""" State-dependent discount factor

    Generalizes the notion of MDP discount factor $\gamma$, making it dependent
    on the state the agent is in. This allows for a finer control over the
    continuation / termination signal for a particular GVF.
    """

    def __init__(self, gamma, *args, **kwargs):
        self.γ = self.gamma = gamma

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.gamma})'


class ConstantContinuation(ContinuationFunction):
    r""" Special case of state independent discounting

    Here we hold $\gamma$ constant at all times, until we receive a termination
    signal from the environment (i.e. a `done` flag in gym envs), in which case
    we set $\gamma=1$.
    """

    def __init__(self, gamma):
        assert isinstance(gamma, float)
        super().__init__(gamma)

    def __call__(self, t: Union[Transition, Trajectory]) -> Union[
        float, torch.Tensor]:
        if isinstance(t.done, torch.Tensor):
            return abs(t.done.int() - 1) * self.gamma
        return self.γ if not t.done else 0


__all__ = get_all_classes(__name__)
