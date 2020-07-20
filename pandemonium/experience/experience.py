from typing import NamedTuple, Union, List

import torch
from torch.distributions import Distribution

Experience = Union['Transition', 'Trajectory']
Transitions = Union[List['Transition']]


class Transition(NamedTuple):
    """ An experience tuple.

    The main purpose of the class is to hold $(s,a,r,s')$ tuples.
    Additional info required by some algorithms can also be added.

    TODO: consider using dataclasses from python3.7
        pros:
            mutability
        cons:
            potentially larger memory footprint
            incompatible with python3.6
            need to write a custom __iter__ method

    TODO: also consider using a TypedDict instead!

    """
    s0: torch.Tensor  # raw observation before state construction
    a: torch.Tensor = None  # action taken by the agent
    r: torch.Tensor = None  # reward received from the environment
    s1: torch.Tensor = None  # raw observation after transition
    done: bool = False  # episode termination indicator
    x0: torch.Tensor = None  # current feature vector
    x1: torch.Tensor = None  # next feature vector
    a1: torch.Tensor = None  # next action selected by the agent
    a_dist: Distribution = None  # distribution from which `a` was generated
    o: 'Option' = None  # an option that was followed during transition
    ρ: float = 1.  # importance sampling weight
    buffer_index: int = -1  # corresponding index in the replay buffer
    info: dict = dict()


class Trajectory(Transition):
    r""" A series of consecutive transitions experienced by the agent. """

    done: torch.Tensor
    ρ: torch.Tensor
    buffer_index: torch.Tensor

    @classmethod
    def from_transitions(cls, t: Transitions):
        """ Creates a Trajectory from a collection of Transitions. """
        # if len(t) == 1:
        #     return cls(t[0])  # Online RL

        batch = cls(*zip(*t))
        device = batch.s0[0].device

        s0 = torch.cat(batch.s0)
        s1 = torch.cat(batch.s1)

        # TODO: asses the trade-offs with replacing by a batched version
        #   i.e. demon.feature(batch.x0)
        x0 = torch.cat(batch.x0)
        x1 = torch.cat(batch.x1)

        a0 = torch.tensor(batch.a, device=device)
        a1 = torch.tensor(batch.a1, device=device)
        r = torch.tensor(batch.r, dtype=torch.float, device=device)
        done = torch.tensor(batch.done, dtype=torch.bool, device=device)
        ρ = torch.tensor(batch.ρ, device=device).unsqueeze(1)
        buffer_index = torch.tensor(batch.buffer_index, dtype=torch.long,
                                    device=device)

        # Try concatenating information in info dictionary
        info = {k: [d[k] for d in batch.info] for k in batch.info[0]}
        for k, v in info.items():
            if isinstance(v[0], torch.Tensor):
                info[k] = torch.cat(v)
            elif isinstance(v[0], int):
                info[k] = torch.tensor(v, torch.long, device=device)
            else:
                pass

        return cls(s0=s0, a=a0, r=r, s1=s1, done=done, x0=x0, x1=x1, a1=a1,
                   ρ=ρ, buffer_index=buffer_index, info=info,
                   o=[None for _ in range(len(t))],
                   a_dist=[None for _ in range(len(t))]
                   )

    def __len__(self):
        return self.s0.shape[0]


__all__ = ['Transition', 'Trajectory', 'Transitions', 'Experience']
