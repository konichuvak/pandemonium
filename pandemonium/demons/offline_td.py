from copy import deepcopy
from typing import Callable

import torch
import torch.nn.functional as F

from pandemonium.demons import (Demon, Loss, ControlDemon, PredictionDemon,
                                ParametricDemon)
from pandemonium.experience import Trajectory, Transitions
from pandemonium.utilities.replay import Replay


class OfflineTD(Demon):
    r""" Base class for forward-view :math:`\text{TD}` methods.

    This class is used as a base for most of the DRL algorithms due to
    synergy with batching.
    """

    def __init__(self, criterion=F.smooth_l1_loss, **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion  # loss function for regression

    def delta(self, trajectory: Trajectory) -> Loss:
        """ Updates a value of a state using information collected offline """
        raise NotImplementedError

    def target(self, *args, **kwargs):
        """ Computes discounted returns for each step in the trajectory """
        raise NotImplementedError

    def learn(self, transitions: Transitions):
        """

        As opposed to the online case, where we learn on individual transitions,
        in the offline case we learn on a sequence of transitions often
        referred to as `Trajectory`.
        """
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)


class DeepOfflineTD(OfflineTD, ParametricDemon):
    """ Mixin for offline :math:`\text{TD}` methods with non-linear FA

    Tries to deal with instabilities of training deep neural network by
    using techniques like target network and replay buffer.
    """

    def __init__(self,
                 replay_buffer: Replay,
                 target_update_freq: int = 0,
                 warm_up_period: int = 0,
                 **kwargs):

        super().__init__(**kwargs)

        self.update_counter = 0
        self.warm_up_period = warm_up_period
        self.target_update_freq = target_update_freq

        # Create a target network to stabilize training
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

        # Use replay buffer for breaking correlation in the experience samples
        self.replay_buffer = replay_buffer

    def learn(self, transitions: Transitions):
        self.update_counter += 1
        self.replay_buffer.feed_batch(transitions)
        self.sync_target()

        if self.update_counter < self.warm_up_period:
            return None, dict()

        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)

    def sync_target(self):
        if self.target_update_freq and self.update_counter % self.target_update_freq == 0:
            self.target_feature.load_state_dict(self.φ.state_dict())
            self.target_avf.load_state_dict(self.avf.state_dict())
            self.target_aqf.load_state_dict(self.aqf.state_dict())

    def __repr__(self):
        # model = torch.nn.Module.__str__(self)[:-2]
        demon = ControlDemon.__repr__(self)
        buffer = f'  (replay_buffer): {self.replay_buffer}'
        hyperparams = f'  (hyperparams):\n' \
                      f'    (warmup): {self.warm_up_period}\n' \
                      f'    (target_update_freq): {self.target_update_freq}\n'
        return f'{demon}\n' \
               f'{buffer}\n' \
               f'{hyperparams}\n)'


class OfflineTDPrediction(OfflineTD, PredictionDemon):
    r""" Offline :math:`\text{TD}(\lambda)` for prediction tasks """

    def delta(self, trajectory: Trajectory) -> Loss:
        x = self.feature(trajectory.s0)
        v = self.predict(x)
        u = self.target(trajectory).detach()
        δ = self.criterion(v, u)
        return δ, {'td': δ.item()}

    def target(self, trajectory: Trajectory):
        return super().target(trajectory, v=self.avf(trajectory.x1))


class OfflineTDControl(OfflineTD, ControlDemon):

    def q_target(self, trajectory: Trajectory, target_fn: Callable = None):
        """ Computes targets for action-value pairs in the trajectory """
        raise NotImplementedError

    def delta(self, trajectory: Trajectory) -> Loss:
        x = self.feature(trajectory.s0)
        v = self.predict_q(x)[torch.arange(x.size(0)), trajectory.a][:, None]
        u = self.target(trajectory).detach()
        δ = self.criterion(v, u)
        return δ, {'td': δ.item()}

    def target(self, trajectory: Trajectory):
        return super().target(trajectory, v=self.q_target(trajectory))


class TTD(OfflineTD):
    r""" Truncated :math:`\text{TD}(\lambda)` """

    def target(self, trajectory: Trajectory, v: torch.Tensor):
        r"""

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
        $[n, n-1, \dots, 1]$-step $\lambda$ returns respectively.
        """

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
