from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness, PixelChange
from pandemonium.demons.demon import (LinearDemon, PredictionDemon, Loss,
                                      ParametricDemon)
from pandemonium.demons.offline_td import TDn
from pandemonium.demons.prediction import OfflineTDPrediction
from pandemonium.experience import Trajectory
from pandemonium.experience import Transitions
from pandemonium.experience.buffers import ER, SkewedER
from pandemonium.implementations.a2c import AC
from pandemonium.implementations.rainbow import DQN
from pandemonium.networks import Reshape
from pandemonium.policies import Policy
from pandemonium.policies.discrete import Greedy
from pandemonium.utilities.utilities import get_all_members


class RewardPrediction(PredictionDemon, ParametricDemon):
    """ A demon that maximizes un-discounted :math:`n`-step return.

    Learns the sign (+, 0, -) of the reward at the end of a state sequence.
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
            Reshape(1, -1),  # stacks feature vectors together
            nn.Linear(feature.feature_dim * sequence_size, output_dim),
        )
        super().__init__(avf=avf, feature=feature, eligibility=None, **kwargs)
        self.replay_buffer = replay_buffer
        self.criterion = F.cross_entropy

    @staticmethod
    def target(trajectory: Trajectory):
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
        if not transitions:
            return None, dict()  # not enough samples in the replay
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        info.update({'rp_traj': trajectory})
        return loss, info


class ValueReplay(LinearDemon, OfflineTDPrediction, TDn):
    r""" :math:`n \text{-step} \TD` performed on the past experiences

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

    def learn(self, transitions: Transitions):
        self.replay_buffer.add_batch(transitions)

        if not self.replay_buffer.is_full:
            return None, dict()

        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(transitions)
        loss, info = self.delta(trajectory)
        info.update({'vr_traj': trajectory})
        return loss, info


class PixelControl(DQN):
    """ Duelling de-convolutional network for auxiliary pixel control task

    References
    ----------
    RL with unsupervised auxiliary tasks (Jaderberg et al. 2016)
    """

    def __init__(self,
                 feature,
                 behavior_policy: Policy,
                 channels: int = 32,
                 kernel: int = 4,
                 stride: int = 2,
                 **kwargs):
        # deconv2d_size_out(6, 2, 1) == 7 (7x7 observation in minigrids)
        # deconv2d_size_out(9, 4, 2) == 20 (20x20 avg pooled pixel change vals)
        # TODO: remove redundant second pass through FC through `feature` method
        fc_reshape = nn.Sequential(
            nn.Linear(feature.feature_dim, 9 * 9 * channels),
            Reshape(-1, channels, 9, 9),
            nn.ReLU()
        )
        action_dim = behavior_policy.action_space.n
        avf = nn.Sequential(OrderedDict({
            'fc_reshape': fc_reshape,
            'deconv_v': nn.ConvTranspose2d(channels, 1, kernel, stride)
        }))
        aqf = nn.Sequential(OrderedDict({
            'fc_reshape': fc_reshape,
            'deconv_q': nn.ConvTranspose2d(channels, action_dim, kernel, stride)
        }))
        super().__init__(duelling=True, avf=avf, aqf=aqf, feature=feature,
                         behavior_policy=behavior_policy, **kwargs)


def create_demons(config, env, φ, μ) -> Horde:
    demons = list()

    # Target policy is greedy
    π = Greedy(feature_dim=φ.feature_dim, action_space=env.action_space)

    # ==========================================================================
    # Main task performed by control demon
    # ==========================================================================

    # The main task is to optimize for the extrinsic reward
    optimal_control = GVF(
        target_policy=π,
        cumulant=Fitness(env),
        continuation=ConstantContinuation(config['gamma'])
    )

    demons.append(AC(gvf=optimal_control, behavior_policy=μ, feature=φ))

    # ==========================================================================
    # Auxiliary tasks performed by a mix of prediction and control demons
    # ==========================================================================

    # Create a shared Experience Replay between PC and VR demons
    replay = None
    if config['pc_weight'] or config['vr_weight']:
        replay = ER(size=config['buffer_size'],
                    batch_size=config['rollout_fragment_length'])

    # --------------------------------------------------------------------------
    # Tracks the color intensity over patches of pixels in the image
    # --------------------------------------------------------------------------
    if config['pc_weight']:
        demons.append(PixelControl(
            gvf=GVF(
                target_policy=π,
                cumulant=PixelChange(),
                continuation=ConstantContinuation(config['gamma']),
            ),
            feature=φ,
            behavior_policy=μ,
            replay_buffer=replay,  # shared with value replay demon
            target_update_freq=config['target_update_freq'],
            double=True,
        ))

    # --------------------------------------------------------------------------
    # The objective of the value replay is to speed up & stabilize A3C agent
    # --------------------------------------------------------------------------
    if config['vr_weight']:
        demons.append(ValueReplay(gvf=optimal_control,
                                  feature=φ,
                                  behavior_policy=μ,
                                  replay_buffer=replay))

    # --------------------------------------------------------------------------
    # Auxiliary task of maximizing un-discounted n-step return
    # --------------------------------------------------------------------------
    if config['rp_weight']:
        demons.append(RewardPrediction(
            gvf=GVF(
                target_policy=π,
                cumulant=Fitness(env),
                continuation=ConstantContinuation(0.),
            ),
            feature=φ,
            behavior_policy=μ,
            replay_buffer=SkewedER(config['buffer_size'],
                                   config['rollout_fragment_length']),
            sequence_size=config.get('rp_sequence_size', 3)
        ))

    # ==========================================================================
    # Combine demons into a horde
    # ==========================================================================

    # TODO: provide a `device` for each demon
    device = torch.device('cpu')

    weights = [
        config.get('ac_weight'),
        config.get('pc_weight'),
        config.get('vr_weight'),
        config.get('rp_weight')
    ]
    demon_weights = torch.tensor(
        data=[w for w in weights if w],
        dtype=torch.float
    ).to(device)

    horde = Horde(
        demons=demons,
        aggregation_fn=lambda losses: demon_weights.dot(losses),
        device=device
    )

    return horde


__all__ = get_all_members(__name__)
