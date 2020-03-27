import random
import sys
from typing import List

import numpy as np
from ray.rllib.utils.schedules import Schedule, ConstantSchedule

from pandemonium.experience import Transitions, Transition
from pandemonium.experience.segment_tree import SumSegmentTree, MinSegmentTree


class ER:
    """ Experience Replay buffer

    References
    ----------
    - Self-Improving Reactive Agents Based On Reinforcement Learning,
      Planning and Teaching (Lin, 1992)
    - Ray RLLib v9.0
    """

    def __init__(self, size: int, batch_size: int):
        """
        Parameters
        ----------
        size
            Max number of transitions to store in the buffer.
            When the buffer overflows the old memories are dropped.
        batch_size
            Number of transitions to sample from the buffer
        """
        self.batch_size = batch_size

        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._hit_count = np.zeros(size)
        self._num_added = 0
        self._num_sampled = 0
        self._est_size_bytes = 0

    def __len__(self):
        return len(self._storage)

    @property
    def capacity(self):
        return self._maxsize

    @property
    def is_full(self):
        return len(self._storage) == self._maxsize

    @property
    def is_empty(self):
        return not len(self._storage)

    def add(self, transition: Transition, weight: float = None) -> None:
        self._num_added += 1

        if self._next_idx >= len(self._storage):
            self._storage.append(transition)
            self._est_size_bytes += sum(sys.getsizeof(d) for d in transition)
        else:
            self._storage[self._next_idx] = transition

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def add_batch(self, transitions: Transitions, weights: List[float] = None):
        if weights is None:
            weights = [None] * len(transitions)
        for experience, weight in zip(transitions, weights):
            self.add(experience, weight)

    def sample(self, batch_size: int = None, contiguous: bool = True):
        r""" Randomly draws a batch of transitions

        Parameters
        ----------
        batch_size
            Number of transitions to sample from the buffer.
        contiguous
            Whether transitions should be contiguous or not.
            This is particularly useful when using $n$-step methods.
        """
        if self.is_empty:
            return None

        if batch_size is None:
            batch_size = self.batch_size

        self._num_sampled += batch_size

        if contiguous:
            ix = np.random.randint(batch_size, len(self))
            samples = self._storage[ix:ix + batch_size]
        else:
            ix = np.random.randint(batch_size, len(self), size=batch_size)
            samples = [self._storage[i] for i in ix]

        return samples


class PER(ER):
    """ Prioritized Experience Replay buffer

    References
    ----------
    - Prioritized Experience Replay (Schaul et al., 2015)
    - Ray RLLib.
    """

    def __init__(self,
                 size: int,
                 batch_size: int,
                 alpha: float = 0.6,
                 beta: Schedule = ConstantSchedule(0.4, framework='torch'),
                 epsilon: float = 1e-6,
                 ):
        """

        Parameters
        ----------
        size
            Max number of transitions to store in the buffer.
            When the buffer overflows the old memories are dropped.
        batch_size
            Number of transitions to sample from the buffer
        alpha
            Controls how much prioritization is used on a scale from 0 to 1.
            0 - no prioritization (uniform sampling)
            1 - full prioritization (sampling proportional to td-error)
        beta
            Degree to which IS correction is performed on a scale from 0 to 1.
            0 - no corrections
            1 - full correction
        epsilon
            Ensures that experiences with 0 td-error are sampled too.

        See Also
        --------
        ER.__init__
        """
        super().__init__(size, batch_size)
        assert alpha > 0
        assert epsilon > 0
        self.α = alpha
        self.β = beta
        self.ε = epsilon

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, transition: Transition, weight: float = None):
        idx = self._next_idx
        super().add(transition, weight)
        if weight is None:
            weight = self._max_priority
        self._it_sum[idx] = weight ** self.α
        self._it_min[idx] = weight ** self.α

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size: int = None, contiguous: bool = True):

        if batch_size is None:
            batch_size = self.batch_size

        self._num_sampled += batch_size

        if contiguous:
            idx = self._sample_proportional(1)[0]
            idxes = list(range(max(idx - batch_size, 0), idx))
        else:
            idxes = self._sample_proportional(batch_size)

        # Add importance sampling ratios and buffer index to the transition
        transitions = []
        β = self.β.value(self._num_sampled)
        total_priority = self._it_sum.sum()
        p_min = self._it_min.min() / total_priority
        ρ_max = (self.capacity * p_min) ** (-β)
        for i in idxes:
            transition = self._storage[i]._asdict()
            prob = self._it_sum[i] / total_priority
            transition['ρ'] = (self.capacity * prob) ** (-β) / ρ_max
            transition['buffer_index'] = i
            transitions.append(Transition(**transition))

        return transitions

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
          List of idxes of sampled transitions
        priorities: [float]
          List of updated priorities corresponding to
          transitions at the sampled idxes denoted by
          variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self.α
            self._it_min[idx] = priority ** self.α
            self._max_priority = max(self._max_priority, priority)


__all__ = ['ER', 'PER']
