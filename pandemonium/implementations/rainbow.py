from warnings import warn

import torch
from torch import nn

from pandemonium.demons import ParametricDemon
from pandemonium.demons.control import (CategoricalQ, DuellingMixin, QLearning,
                                        OfflineTDControl)
from pandemonium.demons.offline_td import TTD
from pandemonium.experience import (ER, PER, ReplayBufferMixin, Trajectory,
                                    Transitions)
from pandemonium.networks import TargetNetMixin
from pandemonium.policies import Policy, torch_argmax_mask
from pandemonium.utilities.utilities import get_all_classes


class DQN(OfflineTDControl,
          QLearning,
          TTD,
          ParametricDemon,
          ReplayBufferMixin,
          TargetNetMixin,
          CategoricalQ,
          DuellingMixin):
    """ Deep Q-Network with all the bells and whistles mixed in.

    References
    ----------
    "Rainbow: Combining Improvements in Deep RL" by Hessel et. al
        https://arxiv.org/pdf/1710.02298.pdf
    """

    def __init__(self,
                 feature: callable,
                 behavior_policy: Policy,
                 replay_buffer: ER,
                 aqf: callable = None,
                 avf: callable = None,
                 target_update_freq: int = 0,
                 warm_up_period: int = None,
                 num_atoms: int = 1,
                 v_min: float = None,
                 v_max: float = None,
                 duelling: bool = False,
                 double: bool = False,
                 **kwargs):

        # Adds a replay buffer
        priority = None
        if isinstance(replay_buffer, PER):
            # Use cross-entropy loss as a measure of priority
            priority = 'ce_loss' if num_atoms > 1 else 'td_error'
        ReplayBufferMixin.__init__(self, replay_buffer, priority)

        # By default, learning does not start until the replay buffer is full
        if warm_up_period is None:
            warm_up_period = replay_buffer.capacity // replay_buffer.batch_size
        self.warm_up_period = warm_up_period

        # Initialize Q-network demon
        if aqf is None:
            aqf = nn.Linear(feature.feature_dim, behavior_policy.action_space.n)

        super(DQN, self).__init__(aqf=aqf, avf=avf, feature=feature,
                                  behavior_policy=behavior_policy, **kwargs)

        # Replaces `avf` implied by `aqf` with an independent estimator
        self.duelling = duelling
        if duelling:
            DuellingMixin.__init__(self)

        # Adds ability to approximate expected values
        # via learning a distribution
        if num_atoms > 1:
            CategoricalQ.__init__(self, num_atoms=num_atoms,
                                  v_min=v_min, v_max=v_max)

        # Adds a target network to stabilize SGD
        TargetNetMixin.__init__(self, target_update_freq)
        if self.target_aqf == self.aqf:
            # TODO: fix the distributional case
            warn('target aqf == aqf')
        if self.target_avf == self.avf:
            warn('target avf == avf')

        # Ensures that that target network exists
        self.double = double
        if self.double:
            assert target_update_freq > 0

    def q_tm1(self, x, a):
        return self.predict_q(x)[torch.arange(a.size(0)), a]

    @torch.no_grad()
    def q_t(self, trajectory: Trajectory):
        q = self.predict_target_q(trajectory.x1)
        if self.double:
            mask = torch_argmax_mask(self.predict_q(trajectory.x1), 1)
            q = (mask * q).sum(1)
        else:
            q = q.max(1)[0]
        return q

    def learn(self, transitions: Transitions):

        self.store(transitions)
        self.sync_target()

        # Wait until warm up period is over
        self.update_counter += 1  # TODO: move the counter up the hierarchy?
        if self.update_counter < self.warm_up_period:
            return None, dict()

        # Learn from experience
        # TODO: differentiate between n-step and batched one-step
        #   We can EITHER sample n transitions at random from a replay buffer
        #   and do a batched one-step backup on them OR we can sample n
        #   consequent transition and do a multistep update with them.
        #   This should be controlled via `n_step` parameter passed to DQN upon
        #   initialization.
        transitions = self.replay_buffer.sample()
        if not transitions:
            return None, dict()  # not enough experience in the buffer
        trajectory = Trajectory.from_transitions(transitions)
        δ, info = self.delta(trajectory)

        # Update the priorities of the collected transitions
        if isinstance(self.replay_buffer, PER):
            self._update_priorities(trajectory, info)

        return δ, info

    def __repr__(self):
        demon = ParametricDemon().__repr__()
        params = f'(replay_buffer): {repr(self.replay_buffer)}\n' \
                 f'(warmup): {self.warm_up_period}\n' \
                 f'(target_update_freq): {self.target_update_freq}\n' \
                 f'(double): {self.double}\n' \
                 f'(duelling): {self.duelling}\n'
        return f'{demon}\n{params}'

    def __str__(self):
        return super().__str__()


__all__ = get_all_classes(__name__)
