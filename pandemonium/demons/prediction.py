import torch.nn.functional as F
from pandemonium.demons.demon import Loss, LinearDemon, ParametricDemon
from pandemonium.demons.td import OfflineTD, TDn, OfflineTDPrediction
from pandemonium.experience import Trajectory
from pandemonium.networks import Reshape
from pandemonium.policies import Policy
from pandemonium.utilities.replay import Replay
from pandemonium.utilities.utilities import get_all_classes
from torch import nn


class RewardPrediction(ParametricDemon, OfflineTDPrediction):
    """ Classifies reward at the end of n-step sequence of states

     Used as an auxiliary task in UNREAL architecture.
     """

    def __init__(self,
                 replay_buffer: Replay,
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
    r""" :math:`n`-step :math:`|text{TD}` performed on the past experiences

    This demon re-samples recent historical sequences from the behavior policy
    distribution and performs extra value function regression. It is used
    in the UNREAL architecture as an auxiliary task that helps representation
    learning.
    """

    def __init__(self,
                 replay_buffer: Replay,
                 behavior_policy: Policy,
                 **kwargs):
        super().__init__(output_dim=behavior_policy.action_space.n,
                         behavior_policy=behavior_policy,
                         **kwargs)
        self.replay_buffer = replay_buffer

    def learn(self, *args, **kwargs) -> Loss:
        if not self.replay_buffer.is_full:
            return None, dict()
        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        return loss, {**info, **{'vr_traj': trajectory}}


__all__ = get_all_classes(__name__)
