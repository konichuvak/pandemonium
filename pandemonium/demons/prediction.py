import torch
import torch.nn.functional as F
from torch import nn

from pandemonium.demons.demon import Loss, LinearDemon
from pandemonium.demons.td import TemporalDifference
from pandemonium.experience import Transition, Trajectory
from pandemonium.utilities.replay import Replay
from pandemonium.utilities.utilities import get_all_classes


class TD(TemporalDifference):
    r""" Semi-gradient TD:math:`\lambda` rule for estimating :math:`\tilde{v} ≈ v_{\pi}`

    .. math::
        \begin{align*}
            e_t &= γ_t λ e_{t-1} + \nabla \tilde{v}(x_t) \\
            w_{t+1} &= w_t + \alpha (z_t + γ_{t+1} \tilde{v}(x_{t+1}) - \tilde{v}(x_t))e_t
        \end{align*}

    In case when $\lambda=0$ we recover one-step $TD(0)$ algorithm with
    $e_t = \nabla \tilde{v}(x_t)$.

    """

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        e = self.λ(γ, t.x0)
        u = self.gvf.z(t) + γ * self.gvf.π(t.x1)
        δ = (u - self.predict(t.x0)) * e
        return δ, {'value_loss': δ.item()}


class TDn(TemporalDifference):
    """ :math:`n`-step :math:`TD` for estimating :math:`V ≈ v_{\pi}`

    Targets are calculated using forward view from $n$-step returns, where
    $n$ is determined by the length of trajectory.
    """

    def delta(self, traj: Trajectory) -> Loss:
        u = self.n_step_target(traj).detach()
        v = self.predict(self.feature(traj.s0)).squeeze(1)
        δ = F.smooth_l1_loss(v, u)
        return δ, {'value_loss': δ.item()}

    def n_step_target(self, traj: Trajectory):
        """
        .. todo::
            ActionVF:
                a = self.gvf.π(traj.x1[-1])
                v = self.target_avf(traj.x1[-1])[a]
            VF:
                v = self.target_avf(traj.x1[-1])
        """
        γ = self.gvf.continuation(traj)
        z = self.gvf.cumulant(traj)
        v = self.avf(traj.x1[-1])
        u = torch.empty_like(z, dtype=torch.float)
        for i in range(len(traj) - 1, -1, -1):
            v = u[i] = z[i] + γ[i] * v
        return u.flip(0)


class RewardPrediction(LinearDemon):
    """ Classifies reward at the end of n-step sequence of states

     Used as an auxiliary task in UNREAL architecture as a 3 class classifier
     for negative, zero and positive rewards.
     """

    def __init__(self,
                 replay_buffer: Replay,
                 output_dim: int = 3,
                 sequence_size: int = 3,
                 **kwargs):
        super().__init__(eligibility=None, output_dim=output_dim, **kwargs)
        self.replay_buffer = replay_buffer
        self.sequence_size = sequence_size
        in_features = self.φ.feature_dim * sequence_size
        self.avf = nn.Linear(in_features, output_dim)

    def delta(self, traj: Trajectory) -> Loss:
        # TODO: special skewed sampling
        x = self.feature(traj.s0).view(1, -1)  # stack features together
        v = self.predict(x)
        u = traj.r[-1].long() + 1  # (-1, 0, 1) -> (0, 1, 2) in dm-lab
        δ = F.cross_entropy(v, u.unsqueeze(0))
        return δ, {'value_loss': δ.item(), 'rp': v.softmax(0)}

    def learn(self, *args, **kwargs):
        if not self.replay_buffer.is_full:
            return None, dict()
        transitions = self.replay_buffer.sample(self.sequence_size)
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        return loss, {**info, **{'rp_traj': trajectory}}


class ValueReplay(LinearDemon, TDn):
    """ N-step TD performed on the past experiences from the replay buffer

    Used in UNREAL architecture as an auxiliary task that helps representation
    learning. This demon re-samples recent historical sequences from the
    behavior policy distribution and performs extra value function regression.
    """

    def __init__(self, replay_buffer: Replay, **kwargs):
        super().__init__(output_dim=1, **kwargs)
        self.replay_buffer = replay_buffer

    def learn(self, *args, **kwargs) -> Loss:
        if not self.replay_buffer.is_full:
            return None, dict()
        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        return loss, {**info, **{'vr_traj': trajectory}}


__all__ = get_all_classes(__name__)
