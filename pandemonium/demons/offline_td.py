import torch
import torch.nn.functional as F
from pandemonium.demons import Demon, Loss, ControlDemon, PredictionDemon
from pandemonium.experience import (Trajectory, Transitions)



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
