import torch
import torch.nn.functional as F
from torch import nn

from pandemonium.demons.demon import PredictionDemon, Demon, Loss
from pandemonium.demons.td import TemporalDifference
from pandemonium.experience import Transition, Trajectory, Transitions
from pandemonium.utilities.replay import Replay
from pandemonium.utilities.utilities import get_all_classes


class TD(TemporalDifference, PredictionDemon):
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
        u = self.gvf.z(t) + γ * self.predict(t.s1)
        return (u - self.predict(t.s0)) * e, dict()


class TDn(TemporalDifference, PredictionDemon):
    """ :math:`n`-step :math:`TD` for estimating :math:`V ≈ v_{\pi}`

    Targets are calculated using forward view from $n$-step returns, where
    $n$ is determined by the length of trajectory.
    """

    def delta(self, traj: Trajectory) -> Loss:
        targets = self.n_step_target(traj)
        values = self.predict(traj.s0).squeeze()
        loss = torch.functional.F.smooth_l1_loss(values, targets)
        return loss, dict()

    def n_step_target(self, traj: Trajectory):
        γ = self.gvf.continuation(traj)
        z = self.gvf.cumulant(traj)
        targets = torch.empty_like(z, dtype=torch.float)
        target = self.predict(traj.s1[-1, None])  # preserving batch dim
        for i in range(len(traj) - 1, -1, -1):
            target = targets[i] = z[i] + γ[i] * target
        return targets

    def learn(self, transitions: Transitions):
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)


class RewardPrediction(Demon):
    """ Classifies reward at the end of n-step sequence of states

     Used as an auxiliary task in UNREAL architecture as a 3 class classifier
     for negative, zero and positive rewards.
     """

    def __init__(self,
                 replay_buffer: Replay,
                 output_dim: int = 2,
                 sequence_size: int = 3,
                 **kwargs):
        super().__init__(eligibility=None, output_dim=output_dim, **kwargs)
        self.replay_buffer = replay_buffer
        self.sequence_size = sequence_size
        in_features = self.φ.feature_dim * sequence_size

        # hack to avoid name collisions in the named parameters
        del self.value_head
        self.reward_predictor = nn.Linear(in_features, output_dim)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        return self.reward_predictor(self.feature(state).view(1, -1))

    def delta(self, *args, **kwargs) -> Loss:
        info = dict()
        if not self.replay_buffer.is_full:
            return None, info
        # TODO: special skewed sampling
        transitions = self.replay_buffer.sample(self.sequence_size)
        trajectory = Trajectory.from_transitions(zip(*transitions))
        values = self.predict(trajectory.s0)
        target = trajectory.r[-1]
        return F.cross_entropy(values, (target > 0).unsqueeze(0).long()), info

    def learn(self, transitions: Transitions):
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)


class ValueReplay(TDn):
    """ N-step TD performed on the past experiences from the replay buffer

    Used in UNREAL architecture as an auxiliary task that helps representation
    learning. This demon re-samples recent historical sequences from the
    behaviour policy distribution and performs extra value function regression.
    """

    def __init__(self, replay_buffer: Replay, **kwargs):
        super().__init__(**kwargs)
        self.replay_buffer = replay_buffer

        # hack to avoid name collisions in the named parameters
        self.value_replay = self.value_head
        del self.value_head

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_replay(self.feature(state))

    def delta(self, traj: Trajectory) -> Loss:
        if not self.replay_buffer.is_full:
            return None, dict()
        batch = Trajectory.from_transitions(zip(*self.replay_buffer.sample()))
        return super().delta(batch)


__all__ = get_all_classes(__name__)
