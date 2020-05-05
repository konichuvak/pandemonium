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
    def v_target(self, trajectory: Trajectory):
        """ Computes value targets from action-value pairs in the trajectory """
        raise NotImplementedError

    # COUNT = 0   # TESTING PER

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
        # if trajectory.r.bool().any():
        #     self.COUNT += 1
        #     print(self.COUNT)
        return loss, {'loss': loss.item(), 'td_error': u - v}

    def target(self, trajectory: Trajectory):
        return super().target(trajectory, v=self.v_target(trajectory))


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


class ReplayBufferMixin:
    r""" Mixin that adds a replay buffer to an agent.

    Was originally designed as a means to make RL more data efficient [1].
    Later on adapted in DQN architecture to make the data distribution more
    stationary [2].

    References
    ----------
    [1] "Self-Improving Reactive Agents Based On RL, Planning and Teaching"
        by Lin. http://www.incompleteideas.net/lin-92.pdf
    [2] "Playing Atari with Deep Reinforcement Learning"
        https://arxiv.org/pdf/1312.5602.pdf
    """
    delta: callable

    def __init__(self,
                 replay_buffer: ER,
                 priority_measure: str = 'td_error'):
        """

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            An instance of a replay buffer
        priority_measure: str
            Priorities to experiences are assigned based on this metric in
            prioritized ER case. Defaults to using `td_error`, but could also
            be `ce_loss` in case of distributional value learning.
        """
        self.replay_buffer = replay_buffer
        self.priority_measure = priority_measure

    def store(self, transitions: Transitions):
        """ Adds transitions to the replay buffer.

        Pre-computes priorities in case Prioritized Experience Replay is used.
        """

        if isinstance(self.replay_buffer, PER):
            trajectory = Trajectory.from_transitions(transitions)
            _, info = self.delta(trajectory)
            priorities = info[self.priority_measure]
            priorities = priorities.abs() + self.replay_buffer.ε
            self.replay_buffer.add_batch(transitions, priorities.tolist())
        else:
            self.replay_buffer.add_batch(transitions)

    def _update_priorities(self, trajectory: Trajectory, info: dict):
        priorities = info[self.priority_measure]
        priorities = priorities.abs() + self.replay_buffer.ε
        indexes = trajectory.buffer_index.tolist()
        self.replay_buffer.update_priorities(indexes, priorities.tolist())


class TargetNetMixin:
    r""" Mixin that adds a target network to the agent.

    Duplicates the estimator networks that are used to estimate targets.
    These clones are updated at a `target_update_freq` frequency which allows
    to stabilize the training process and make targets less non-stationary.

    References
    ----------
    [1] "Playing Atari with Deep Reinforcement Learning"
        https://arxiv.org/pdf/1312.5602.pdf
    """
    aqf: callable
    avf: callable
    φ: callable

    def __init__(self, target_update_freq: int = 0):

        self.update_counter = 0  # keeps track of the number of updates so far

        # Create target networks to stabilize training
        self.target_networks = dict()
        self.target_update_freq = target_update_freq
        if self.target_update_freq:
            self.target_feature = deepcopy(self.φ)
            self.target_feature.load_state_dict(self.φ.state_dict())
            self.target_feature.eval()
            self.target_networks[self.target_feature] = self.φ

            if isinstance(self.avf, nn.Module):
                # `avf` is a network in duelling DQN and is implicit in `aqf`
                # in other cases
                self.target_avf = deepcopy(self.avf)
                self.target_avf.load_state_dict(self.avf.state_dict())
                self.target_avf.eval()
                self.target_networks[self.target_avf] = self.avf

            if isinstance(self.aqf, nn.Module):
                # `aqf` is a network in all DQN family except distributional
                # case, where it is implicit in the `azf`
                self.target_aqf = deepcopy(self.aqf)
                self.target_aqf.load_state_dict(self.aqf.state_dict())
                self.target_aqf.eval()
                self.target_networks[self.target_aqf] = self.aqf

            if hasattr(self, 'azf') and isinstance(self.azf, nn.Module):
                # only for distributional agents
                self.target_azf = deepcopy(self.azf)
                self.target_azf.load_state_dict(self.azf.state_dict())
                self.target_azf.eval()
                self.target_networks[self.target_azf] = self.azf

    def sync_target(self):
        if self.target_update_freq:
            if self.update_counter % self.target_update_freq == 0:
                for target_net, net in self.target_networks.items():
                    target_net.load_state_dict(net.state_dict())
