import torch

from pandemonium.demons.demon import PredictionDemon
from pandemonium.demons.td import TemporalDifference
from pandemonium.experience import Transition, Trajectory


class TD(TemporalDifference, PredictionDemon):
    r""" Semi-gradient $TD(\lambda)$ rule for estimating $\tilde{v}$

    .. math::
        e_t = γ_t λ e_{t-1} + \Nabla \Tilde{v}(x_t)
        w_{t+1} = w_t + \alpha (z_t + γ_{t+1}\Tilde{v}(x_{t+1}) - \Tilde{v}(x_t))e_t

    In case when $\lambda=0$ we recover one-step TD(0) algorithm:

    .. math::
        e_t = \Nabla \Tilde{v}(x_t)
        w_{t+1} = w_t + \alpha (z_t + γ_{t+1}V(x_{t+1}) - V(x_t}))e_t

    """

    def delta(self, t: Transition):
        γ = self.gvf.continuation(t)
        e = self.λ(γ, t.x0)
        u = self.gvf.z(t) + γ * self.predict(t.s1)
        return (u - self.predict(t.s0)) * e


class TDn(TemporalDifference, PredictionDemon):
    """ Bootstrapped Temporal Difference

    Targets are calculated using forward view from n-step returns, where
    n is determined by the length of trajectory.
    """

    def delta(self, traj: Trajectory):
        targets = self.n_step_target(traj)
        values = self.predict(traj.s0)
        loss = torch.functional.F.smooth_l1_loss(values, targets)
        return loss

    def n_step_target(self, traj: Trajectory):
        γ = self.gvf.continuation(traj)
        targets = torch.empty_like(traj.r, dtype=torch.float)
        target = self.predict(traj.s1[-1, None])  # preserving batch dim
        for i in range(len(traj)-1, -1, -1):
            target = targets[i] = traj.r[i] + γ[i] * target
        return targets
