from copy import deepcopy

import torch
import torch.nn.functional as F

from pandemonium.demons import (Demon, Loss, ControlDemon, PredictionDemon,
                                ParametricDemon)
from pandemonium.experience import ER, PER, Trajectory, Transitions


class OfflineTD(Demon):
    r""" Base class for forward-view :math:`\text{TD}` methods.

    This class is used as a base for most of the DRL algorithms due to
    synergy with batching.
    """

    def __init__(self, criterion=F.smooth_l1_loss, **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion  # loss function for regression

    def delta(self, trajectory: Trajectory) -> Loss:
        """ Updates a value of a state using information in the trajectory """
        raise NotImplementedError

    def target(self, *args, **kwargs):
        """ Computes discounted returns for each step in the trajectory """
        raise NotImplementedError

    def learn(self, transitions: Transitions):
        """

        As opposed to the online case, where we learn on individual
        `Transition`s, in the offline case we learn on a sequence of
        transitions referred to as `Trajectory`.
        """
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)


class DeepOfflineTD(OfflineTD, ParametricDemon):
    """ Mixin for offline :math:`\text{TD}` methods with non-linear FA

    Tries to deal with instabilities of training deep neural network by
    using techniques like target network and replay buffer.
    """

    def __init__(self,
                 replay_buffer: ER,
                 target_update_freq: int = 0,
                 warm_up_period: int = None,
                 **kwargs):

        super().__init__(**kwargs)

        self.update_counter = 0  # keeps track of the number of updates so far

        # Use replay buffer for breaking correlation in the experience samples
        self.replay_buffer = replay_buffer

        # By default, learning does not start until the replay buffer is full
        if warm_up_period is None:
            warm_up_period = replay_buffer.capacity // replay_buffer.batch_size
        self.warm_up_period = warm_up_period

        # Create a target network to stabilize training
        self.target_update_freq = target_update_freq
        if self.target_update_freq:
            self.target_feature = deepcopy(self.φ)
            self.target_feature.load_state_dict(self.φ.state_dict())
            self.target_feature.eval()

            self.target_avf = deepcopy(self.avf)
            self.target_avf.load_state_dict(self.avf.state_dict())
            self.target_avf.eval()

            self.target_aqf = deepcopy(self.aqf)
            self.target_aqf.load_state_dict(self.aqf.state_dict())
            self.target_aqf.eval()

        else:
            self.target_feature = self.φ
            self.target_avf = self.avf
            self.target_aqf = self.aqf

    def learn(self, transitions: Transitions):

        # Add transitions to the replay buffer
        if isinstance(self.replay_buffer, PER):
            # Compute TD-error to determine priorities for transitions
            trajectory = Trajectory.from_transitions(transitions)
            _, info = self.delta(trajectory)
            priorities = info['td_error'].abs() + self.replay_buffer.ε
            priorities = [w.item() for w in priorities]
            self.replay_buffer.add_batch(transitions, priorities)
        else:
            self.replay_buffer.add_batch(transitions)

        self.sync_target()

        # Wait until warm up period is over
        self.update_counter += 1
        if self.update_counter < self.warm_up_period:
            return None, dict()

        # Learn from experience
        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(transitions)
        δ, info = self.delta(trajectory)

        # Update the priorities according to the new TD-error
        if isinstance(self.replay_buffer, PER):
            priorities = info['td_error'].abs() + self.replay_buffer.ε
            priorities = [w.item() for w in priorities]
            indexes = trajectory.buffer_index.tolist()
            self.replay_buffer.update_priorities(indexes, priorities)
        return δ, info

    def sync_target(self):
        if self.target_update_freq and self.update_counter % self.target_update_freq == 0:
            self.target_feature.load_state_dict(self.φ.state_dict())
            self.target_avf.load_state_dict(self.avf.state_dict())
            self.target_aqf.load_state_dict(self.aqf.state_dict())

    def __repr__(self):
        demon = ControlDemon.__repr__(self)
        params = f'(replay_buffer): {repr(self.replay_buffer)}\n' \
                 f'(warmup): {self.warm_up_period}\n' \
                 f'(target_update_freq): {self.target_update_freq}'
        return f'{demon}\n{params}'

    def __str__(self):
        return ControlDemon.__str__(self)


class OfflineTDPrediction(OfflineTD, PredictionDemon):
    r""" Offline :math:`\text{TD}(\lambda)` for prediction tasks """

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        """ Computes value targets from states in the trajectory """
        raise NotImplementedError

    def delta(self, trajectory: Trajectory) -> Loss:
        x = self.feature(trajectory.s0)
        v = self.predict(x)
        u = self.target(trajectory).detach()
        loss = self.criterion(input=v, target=u, reduction='none')
        loss = (loss * trajectory.ρ).mean()  # weighted IS
        return loss, {'loss': loss.item(), 'td_error': u - v}

    def target(self, trajectory: Trajectory):
        return super().target(trajectory, v=self.v_target(trajectory))


class OfflineTDControl(OfflineTD, ControlDemon):

    @torch.no_grad()
    def q_target(self, trajectory: Trajectory):
        """ Computes value targets from action-value pairs in the trajectory """
        raise NotImplementedError

    COUNT = 0

    def delta(self, trajectory: Trajectory) -> Loss:
        x = self.feature(trajectory.s0)
        a = torch.arange(x.size(0)), trajectory.a
        v = self.predict_q(x)[a].unsqueeze(1)
        u = self.target(trajectory).detach()
        assert u.shape == v.shape, f'{u.shape} vs {v.shape}'
        loss = self.criterion(input=v, target=u, reduction='none')
        if len(loss.shape) > 2:
            # take average across states
            loss = loss.mean(tuple(range(2, len(loss.shape))))
        loss = (loss * trajectory.ρ).mean()  # weighted IS
        if trajectory.r.bool().any():
            self.COUNT += 1
            print(self.COUNT)
        return loss, {'loss': loss.item(), 'td_error': u - v}

    def target(self, trajectory: Trajectory):
        return super().target(trajectory, v=self.q_target(trajectory))


class TTD(OfflineTD):
    r""" Truncated :math:`\text{TD}(\lambda)`

    Notes
    -----
    Generalizes $n$-step $\text{TD}$ by allowing arbitrary mixing of
    $n$-step returns via $\lambda$ parameter.

    Depending on the algorithm, vector `v` would contain different
    bootstrapped estimates of values:

    .. math::
        - \text{TD}(\lambda) (forward view): state value estimates \text{V_t}(s)
        - \text{Q}(\lambda): action value estimates \max\limits_{a}(Q_t(s_t, a))
        - \text{SARSA}(\lambda): action value estimates Q_t(s_t, a_t)

    The resulting vector `u` contains target returns for each state along
    the trajectory, with $V(S_i)$ for $i \in \{0, 1, \dots, n-1\}$ getting
    updated towards $[n, n-1, \dots, 1]$-step $\lambda$ returns respectively.

    References
    ----------
    - Sutton and Barto (2018) ch. 12.3, 12.8, equation (12.18)
    - van Seijen (2016) Appendix B, https://arxiv.org/pdf/1608.05151v1.pdf
    """

    def target(self, trajectory: Trajectory, v: torch.Tensor):
        γ = self.gvf.continuation(trajectory)
        z = self.gvf.cumulant(trajectory)
        λ = self.λ(trajectory)
        g = v[-1]
        u = torch.empty_like(v, dtype=torch.float)
        for i in range(len(trajectory) - 1, -1, -1):
            g = u[i] = z[i] + γ[i] * ((1 - λ[i]) * v[i] + λ[i] * g)
        return u


class TDn(TTD):
    r""" :math:`\text{n-step TD}` for estimating :math:`V ≈ v_{\pi}`

    Targets are calculated using forward view from $n$-step returns, where
    $n$ is determined by the length of trajectory. $\text{TDn}$ is a special
    case of truncated $\text{TD}$ with $\lambda=1$.
    """

    def __init__(self, **kwargs):
        super().__init__(
            eligibility=lambda trajectory: torch.ones_like(trajectory.r),
            **kwargs
        )
