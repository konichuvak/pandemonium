from typing import NamedTuple, Union, Collection, Iterator

import torch
from torch.distributions import Distribution


class Transition(NamedTuple):
    """ An experience tuple

    The main purpose of the class is to hold $(s,a,r,s')$ tuples.
    Additional info required by some algorithms can also be added.

    .. todo::
        consider using dataclases from python3.7

    """
    s0: torch.Tensor  # raw observation before state construction
    a: torch.Tensor  # action taken by the agent
    r: torch.Tensor  # reward received from the environment
    s1: torch.Tensor  # raw observation after transition
    done: bool = False  # episode termination indicator
    x0: torch.Tensor = None  # current feature vector
    x1: torch.Tensor = None  # next feature vector
    a_dist: Distribution = None  # distribution from which `a` was generated
    o: 'Option' = None  # an option that was followed during transition
    info: dict = {}


Transitions = Union[Collection[Transition], Iterator[Collection[Transition]]]


class Trajectory(Transition):
    """ A batch of transitions """

    done: torch.Tensor
    advantages: torch.Tensor = None

    @classmethod
    def from_transitions(cls, t: Transitions):
        batch = cls(*zip(*t))
        device = batch.s0[0].device

        if len(batch.s0[0].shape) > 1:
            s0 = torch.cat(batch.s0)
            s1 = torch.cat(batch.s1)
        else:
            s0 = torch.stack(batch.s0)
            s1 = torch.stack(batch.s1)
        a = torch.tensor(batch.a, device=device)
        z = torch.tensor(batch.r, device=device)
        done = torch.tensor(batch.done, device=device, dtype=torch.bool)

        return cls(s0, a, z, s1, done)

    def __len__(self):
        return self.s0.shape[0]
