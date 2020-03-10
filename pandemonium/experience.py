from typing import NamedTuple, Union, List

import torch
from pandemonium.utilities.utilities import get_all_classes
from torch.distributions import Distribution

Experience = Union['Transition', 'Trajectory']
Transitions = Union[List['Transition']]


class Transition(NamedTuple):
    """ An experience tuple

    The main purpose of the class is to hold $(s,a,r,s')$ tuples.
    Additional info required by some algorithms can also be added.

    .. todo::
        consider using dataclases from python3.7

    """
    s0: torch.Tensor  # raw observation before state construction
    a: torch.Tensor = None  # action taken by the agent
    r: torch.Tensor = None  # reward received from the environment
    s1: torch.Tensor = None  # raw observation after transition
    done: bool = False  # episode termination indicator
    x0: torch.Tensor = None  # current feature vector
    x1: torch.Tensor = None  # next feature vector
    a_dist: Distribution = None  # distribution from which `a` was generated
    o: 'Option' = None  # an option that was followed during transition
    info: dict = dict()


class Trajectory(Transition):
    """ A batch of transitions

    .. todo::
        - make an option to iterate over transitions
    """

    done: torch.Tensor

    @classmethod
    def from_transitions(cls, t: Transitions):
        batch = cls(*zip(*t))
        device = batch.s0[0].device

        if len(batch.s0[0].shape) > 1:
            s0 = torch.cat(batch.s0)
            s1 = torch.cat(batch.s1)
            x0 = torch.cat(batch.x0)
            x1 = torch.cat(batch.x1)
        else:
            s0 = torch.stack(batch.s0)
            s1 = torch.stack(batch.s1)
            x0 = torch.stack(batch.x0)
            x1 = torch.stack(batch.x1)
        a = torch.tensor(batch.a, device=device)
        r = torch.tensor(batch.r, device=device, dtype=torch.float)
        done = torch.tensor(batch.done, device=device, dtype=torch.bool)

        # Try concatenating information in info dictionary
        info = {k: [d[k] for d in batch.info] for k in batch.info[0]}
        for k, v in info.items():
            if isinstance(v[0], torch.Tensor):
                info[k] = torch.cat(v)
            elif isinstance(v[0], int):
                info[k] = torch.tensor(v, device=device)
            else:
                pass

        return cls(s0, a, r, s1, done, x0, x1, info=info)

    def __len__(self):
        return self.s0.shape[0]


__all__ = get_all_classes(__name__)
