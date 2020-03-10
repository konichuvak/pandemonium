from typing import Type, Callable

import torch
import torch.nn.functional as F
from pandemonium.demons import Demon, Loss, ControlDemon, PredictionDemon
from pandemonium.experience import Trajectory, Transition, Transitions
from pandemonium.traces import EligibilityTrace, AccumulatingTrace


class OfflineTD(Demon):
    r""" Base class for forward-view :math:`\text{TD}` methods.

    This class is used as a base for most of the DRL algorithms due to
    synergy with batching. Uses $\lambda$ returns to determine the target for
    regression.
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

    def target(self, trajectory: Trajectory, v: torch.Tensor = None):
        if v is None:
            v = self.q_target(trajectory)
        return super().target(trajectory=trajectory, v=v)


class SARSA(OfflineTDControl):
    r""" :math:`n`-step semi-gradient :math:`\text{SARSA}` """

    def q_target(self, trajectory: Trajectory, target_fn: Callable = None):
        if target_fn is None:
            target_fn = self.aqf
        q = target_fn(trajectory.x1)
        a = self.gvf.π(trajectory.x1, vf=target_fn)
        v = q[torch.arange(q.size(0)), a]
        return v


class SARSE(OfflineTDControl):
    r""" :math:`n`-step semi-gradient expected :math:`\text{SARSA}` """

    def q_target(self, trajectory: Trajectory, target_fn: Callable = None):
        if target_fn is None:
            target_fn = self.aqf
        q = target_fn(trajectory.x1)
        dist = self.gvf.π.dist(trajectory.x1, vf=target_fn)
        v = q * dist.probs
        return v


class QLearning(OfflineTDControl):

    def q_target(self, trajectory: Trajectory, target_fn: Callable = None):
        if target_fn is None:
            target_fn = self.aqf
        # TODO: To re-compute or not to re-compute x?
        x = self.feature(trajectory.s1)
        # x = trajectory.x1
        q = target_fn(x)
        v = q.max(1, keepdim=True)[0]
        return v


# ------------------------------------------------------------------------------
class OnlineTD(Demon):
    r""" Base class for backward-view :math:`\text{TD}` methods. """

    def __init__(self,
                 feature,
                 trace_decay: float,
                 eligibility: Type[EligibilityTrace] = AccumulatingTrace,
                 **kwargs):
        e = eligibility(trace_decay, feature.feature_dim)
        super().__init__(eligibility=e, feature=feature, **kwargs)


class TDlambda(OnlineTD):
    r""" Semi-gradient :math:`\text{TD}\lambda` rule for estimating :math:`\tilde{v} ≈ v_{\pi}`

    .. math::
        \begin{align*}
            e_t &= γ_t λ e_{t-1} + \nabla \tilde{v}(x_t) \\
            w_{t+1} &= w_t + \alpha (z_t + γ_{t+1} \tilde{v}(x_{t+1}) - \tilde{v}(x_t))e_t
        \end{align*}
    """

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        e = self.λ(γ, t.x0)
        δ = z + γ * self.predict(t.x1) - self.predict(t.x0)
        return δ * e, {'td': δ.item(), 'eligibility': e.item()}


class TD0(OnlineTD):
    r""" A special case of :math:`\text{TD}\lambda` with :math:`\lambda = 0`

    $\text{TD}(0)$ is known as one-step $\text{TD}$ algorithm with
    $e_t = \nabla \tilde{v}(x_t)$.
    """

    def __init__(self, **kwargs):
        super().__init__(trace_decay=0, **kwargs)


class TrueOnlineTD(OnlineTD):
    pass
