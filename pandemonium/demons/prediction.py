import torch.nn.functional as F
from torch import nn

from pandemonium.demons.demon import Loss, LinearDemon, ParametricDemon
from pandemonium.demons.offline_td import TDn, OfflineTDPrediction
from pandemonium.experience import ER, Trajectory
from pandemonium.networks import Reshape
from pandemonium.utilities.utilities import get_all_classes


class RewardPrediction(ParametricDemon, OfflineTDPrediction):
    """ Classifies reward at the end of a state sequence

    Used as an auxiliary task in UNREAL architecture.

    References
    ----------
    RL with unsupervised auxiliary tasks (Jaderberd et al., 2016)
    """

    def __init__(self,
                 replay_buffer: ER,
                 feature,
                 output_dim: int = 3,
                 sequence_size: int = 3,
                 **kwargs):
        self.sequence_size = sequence_size
        avf = nn.Sequential(
            Reshape(1, -1),
            nn.Linear(feature.feature_dim * sequence_size, output_dim),
        )
        super().__init__(
            avf=avf,
            feature=feature,
            eligibility=None,
            criterion=F.cross_entropy,
            **kwargs
        )
        self.replay_buffer = replay_buffer

    def learn(self, *args, **kwargs):
        # TODO: special skewed sampling
        if not self.replay_buffer.is_full:
            return None, dict()
        transitions = self.replay_buffer.sample(self.sequence_size)
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        return loss, {**info, **{'rp_traj': trajectory}}

    def target(self, trajectory: Trajectory):
        """ Ternary classification target for -, 0, + rewards """
        g = trajectory.r[-1].long() + 1  # (-1, 0, 1) -> (0, 1, 2)
        return g.unsqueeze(0)


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

    def learn(self, *args, **kwargs) -> Loss:
        if not self.replay_buffer.is_full:
            return None, dict()
        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        return loss, {**info, **{'vr_traj': trajectory}}


__all__ = get_all_classes(__name__)
