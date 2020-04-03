import torch
import torch.nn.functional as F
from torch import nn

from pandemonium.demons.demon import LinearDemon, ParametricDemon, Loss
from pandemonium.demons.offline_td import TDn, OfflineTDPrediction
from pandemonium.experience import ER, SkewedER, Trajectory, Transitions
from pandemonium.networks import Reshape
from pandemonium.utilities.utilities import get_all_classes


class RewardPrediction(ParametricDemon):
    """ Classifies reward at the end of a state sequence

    Used as an auxiliary task in UNREAL architecture.

    References
    ----------
    RL with unsupervised auxiliary tasks (Jaderberd et al., 2016)
    """

    def __init__(self,
                 replay_buffer: SkewedER,
                 feature,
                 output_dim: int = 3,
                 sequence_size: int = 3,
                 **kwargs):
        self.sequence_size = sequence_size
        avf = nn.Sequential(
            Reshape(1, -1),  # stacks frames together
            nn.Linear(feature.feature_dim * sequence_size, output_dim),
        )
        super().__init__(
            avf=avf,
            feature=feature,
            eligibility=None,
            **kwargs
        )
        self.replay_buffer = replay_buffer
        self.criterion = F.cross_entropy

    def predict(self, x):
        return self.avf(x)

    def target(self, trajectory: Trajectory):
        """ Ternary classification target for -, 0, + rewards """
        g = trajectory.r[-1].item()
        if g > 0:
            g = torch.tensor([0])
        elif g < 0:
            g = torch.tensor([1])
        else:
            g = torch.tensor([2])
        return g.long().to(trajectory.r.device)

    def delta(self, trajectory: Trajectory) -> Loss:
        x = self.feature(trajectory.s0)
        v = self.predict(x)
        u = self.target(trajectory).detach()
        loss = self.criterion(input=v, target=u)
        return loss, {'loss': loss.item()}

    def learn(self, transitions: Transitions):
        self.replay_buffer.add_batch(transitions)

        if not self.replay_buffer.is_full:
            return None, dict()

        transitions = self.replay_buffer.sample(self.sequence_size)
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        info.update({'rp_traj': trajectory})
        return loss, info


class ValueReplay(LinearDemon, OfflineTDPrediction, TDn):
    r""" :math:`n`-step :math:`\text{TD}` performed on the past experiences

    This demon re-samples recent historical sequences from the behavior policy
    distribution and performs extra value function regression. It is used
    in the UNREAL architecture as an auxiliary task that helps representation
    learning.

    References
    ----------
    RL with unsupervised auxiliary tasks (Jaderberd et al., 2016)
    """

    def __init__(self, replay_buffer: ER, **kwargs):
        super().__init__(output_dim=1, **kwargs)
        self.replay_buffer = replay_buffer

    def v_target(self, trajectory: Trajectory):
        return self.avf(trajectory.x1)

    def learn(self, transitions: Transitions):
        self.replay_buffer.add_batch(transitions)

        if not self.replay_buffer.is_full:
            return None, dict()

        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        info.update({'rp_traj': trajectory})
        return loss, info


__all__ = get_all_classes(__name__)
