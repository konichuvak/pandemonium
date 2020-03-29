import random
import sys
from typing import List, Callable

import numpy as np
import torch
from pandemonium.experience import Transitions, Transition
# from ray.rllib.optimizers.segment_tree import SumSegmentTree, MinSegmentTree
from pandemonium.experience.segment_tree import SumSegmentTree, MinSegmentTree
from pandemonium.utilities.schedules import Schedule, ConstantSchedule
from torch.distributions import Categorical

__all__ = ['ER', 'PER']


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

    @property
    def capacity(self):
        return self._maxsize

    @property
    def is_full(self):
        return len(self) == self._maxsize

    @property
    def is_empty(self):
        return not len(self)

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
            ix = np.random.randint(0, len(self), size=batch_size)
            samples = [self._storage[i] for i in ix]

        return samples

    def __len__(self):
        return len(self._storage)

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'batch_size={self.batch_size}, ' \
               f'capacity={self.capacity})'


class SegmentedER(ER):
    """ Segmented Experience Replay

    Allows for partitioning the ER into multiple segments, specifying a
    sampling distribution over segments. Used in the UNREAL architecture for
    reward prediction task.

    References
    ----------
    RL with unsupervised auxiliary tasks (Jaderberd et al., 2016)
    """

    def __init__(self,
                 size: int,
                 batch_size: int,
                 segments: int,
                 criterion: Callable[[Transition], int],
                 dist: Categorical = None):
        """
        Parameters
        ----------
        segments
            Number of segments to split the memory into.
        size
            Max number of transitions stored across all segments.
            The memory size is split uniformly across segments.
        batch_size
            Number of transitions to sample from the buffer
        criterion
            A rule deciding which segment of the memory to put transition into
        dist
            Sampling distribution across segments (defaults to uniform).
        """
        super().__init__(size=size, batch_size=batch_size)

        # Experience is stored in the individual buffers
        del self._storage
        self.segments = segments
        self.buffers = [ER(size // segments, batch_size) for _ in
                        range(segments)]

        self.criterion = criterion

        if dist is None:
            dist = Categorical(torch.ones(segments) * 1 / segments)
        self.dist = dist

    def add(self, transition: Transition, weight: float = None) -> None:
        self.buffers[self.criterion(transition)].add(transition, weight)

    def sample(self, batch_size: int = None, contiguous: bool = True):
        if batch_size is None:
            batch_size = self.batch_size
        buffer = self.buffers[self.dist.sample().item()]
        samples = buffer.sample(batch_size=batch_size, contiguous=contiguous)
        return samples

    def __len__(self):
        # The size of the buffer is the sum of the sizes of individual buffers
        return sum([len(b) for b in self.buffers])


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
            # extra mass to ensure that we avoid sampling idx < batch_size
            mass = random.random() * self._it_sum.sum(batch_size, len(self))
            extra_mass = self._it_sum.sum(0, batch_size)
            idx = self._it_sum.find_prefixsum_idx(mass + extra_mass)
            assert idx >= batch_size
            idxes = list(range(idx - batch_size, idx))
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

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'batch_size={self.batch_size}, ' \
               f'capacity={self.capacity}, ' \
               f'α={self.α}, ' \
               f'β={self.β}, ' \
               f'ε={self.ε})'
