from collections import Callable, OrderedDict
from copy import deepcopy

import torch
from pandemonium.demons import (ControlDemon, Loss, ParametricDemon)
from pandemonium.demons.td import (OfflineTD, SARSA, QLearning, SARSE,
                                   OfflineTDControl, TTD, TDn,
                                   OfflineTDPrediction)
from pandemonium.experience import Trajectory, Transitions
from pandemonium.networks import Reshape
from pandemonium.policies import Policy, HierarchicalPolicy
from pandemonium.policies.gradient import DiffPolicy
from pandemonium.utilities.replay import Replay
from pandemonium.utilities.utilities import get_all_classes
from torch import nn


class DeepOfflineTD(OfflineTD, ParametricDemon):
    """ Mixin for offline :math:`\text{TD}` methods with non-linear FA """

    def __init__(self,
                 replay_buffer: Replay,
                 target_update_freq: int = 0,
                 warm_up_period: int = 0,
                 **kwargs):

        super().__init__(**kwargs)

        self.update_counter = 0
        self.warm_up_period = warm_up_period
        self.target_update_freq = target_update_freq

        # Create a target network to stabilize training
        if self.target_update_freq:
            self.target_feature = deepcopy(self.φ)
            self.target_feature.load_state_dict(self.φ.state_dict())
            self.target_feature.eval()

            self.target_avf = deepcopy(self.avf)
            self.target_avf.load_state_dict(self.avf.state_dict())
            self.target_avf.eval()

            self.target_aqf = deepcopy(self.aqf)
            self.target_aqf.load_state_dict(self.aqf.state_dict())
            self.target_aqf.eval()

        else:
            self.target_feature = self.φ
            self.target_avf = self.avf
            self.target_aqf = self.aqf

        # Use replay buffer for breaking correlation in the experience samples
        self.replay_buffer = replay_buffer

    def learn(self, transitions: Transitions):
        self.update_counter += 1
        self.replay_buffer.feed_batch(transitions)
        self.sync_target()

        if self.update_counter < self.warm_up_period:
            return None, dict()

        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)

    def sync_target(self):
        if self.target_update_freq and self.update_counter % self.target_update_freq == 0:
            self.target_feature.load_state_dict(self.φ.state_dict())
            self.target_avf.load_state_dict(self.avf.state_dict())

    def __repr__(self):
        # model = torch.nn.Module.__str__(self)[:-2]
        demon = ControlDemon.__repr__(self)
        buffer = f'  (replay_buffer): {self.replay_buffer}'
        hyperparams = f'  (hyperparams):\n' \
                      f'    (warmup): {self.warm_up_period}\n' \
                      f'    (target_update_freq): {self.target_update_freq}\n'
        return f'{demon}\n' \
               f'{buffer}\n' \
               f'{hyperparams}\n)'


class DeepOfflineTDControl(DeepOfflineTD, OfflineTDControl):

    def __init__(self, duelling: bool = False, *args, **kwargs):
        self.duelling = duelling
        super().__init__(*args, **kwargs)

    def predict_q(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        In the duelling case we treat 'aqf' as an advantage estimator?
        """
        if self.duelling:
            v, q = self.avf(x), self.aqf(x)
            return q - (q - v).mean(1, keepdim=True)
        return self.aqf(x)


class DeepSARSA(DeepOfflineTDControl, SARSA):
    pass


class DeepSARSE(DeepOfflineTDControl, SARSE):
    pass


class DQN(DeepOfflineTDControl, QLearning, TDn):
    """ Deep Q-Network """

    def q_target(self, trajectory: Trajectory, target_fn: Callable = None):
        q = self.predict_q(trajectory.x1)
        v = q.max(1, keepdim=True)[0]
        return v


class AC(ParametricDemon):
    """ Base class for Actor-Critic architectures that operate on batches

    https://hadovanhasselt.files.wordpress.com/2016/01/pg1.pdf
    """

    def __init__(self, actor: DiffPolicy, **kwargs):
        super().__init__(behavior_policy=actor, **kwargs)

    def critic_loss(self, trajectory: Trajectory):
        raise NotImplementedError

    def actor_loss(self, trajectory: Trajectory, weights):
        x = self.feature(trajectory.s0)
        return self.μ.delta(x, trajectory.a, weights)

    def delta(self, trajectory: Trajectory) -> Loss:
        stats = dict()
        critic_loss, weights, info = self.critic_loss(trajectory)
        stats.update(**info)
        # TODO: what happens if we do not detach?
        actor_loss, info = self.actor_loss(trajectory, weights.detach())
        stats.update(**info)
        loss = actor_loss + 0.5 * critic_loss
        return loss, stats


class A2C(AC, ControlDemon):
    """ Advantage Actor Critic

    Using two demons to approximate $v_{\pi_{\theta}}(s)$ and $q_{\pi_{\theta}}(s, a)$
    at the same time.
    """

    def __init__(self, feature, actor, **kwargs):
        super().__init__(
            avf=nn.Linear(feature.feature_dim, 1),
            aqf=nn.Linear(feature.feature_dim, actor.action_space.n),
            actor=actor,
            feature=feature, **kwargs
        )

    def critic_loss(self, trajectory: Trajectory):
        raise NotImplementedError
        # v = self.predict(trajectory.x0)
        # u = self.target(trajectory).detach()
        # δ = F.smooth_l1_loss(v, u)
        # return δ, u - v


class QAC(AC):
    pass


class TDAC(AC, OfflineTDPrediction, TTD):
    r""" A variation of AC that approximates advantage with :math:`\delta`

    This version is using truncated $\lamda$ returns to compute $\delta$, i.e.

    .. math::
        \Delta \theta = (v^{\lambda}_t - v_t(s)) \nabla_{\theta}log\pi_{\theta}(s_t, a_t)

    """

    def critic_loss(self, trajectory: Trajectory):
        δ, info = OfflineTDPrediction.delta(self, trajectory)
        return δ, δ.clone(), info


class OC(AC, DQN):
    """ DQN style Option-Critic architecture """

    def __init__(self, actor: HierarchicalPolicy, **kwargs):
        super().__init__(
            output_dim=len(actor.option_space),
            actor=actor,
            # TODO: make a sequential replay that samples last
            #  BATCH_SIZE transitions in order
            replay_buffer=Replay(memory_size=0, batch_size=0),
            **kwargs
        )

        # Collect parameters of all the options into one computational graph
        # TODO: instead of manually collecting we can make HierarchicalPolicy
        #  and OptionSpace subclass nn.Module
        for idx, o in self.μ.option_space.options.items():
            for k, param in o.policy.named_parameters(f'option_{idx}'):
                self.register_parameter(k.replace('.', '_'), param)
            for k, param in o.continuation.named_parameters(f'option_{idx}'):
                self.register_parameter(k.replace('.', '_'), param)

    def learn(self, transitions: Transitions):
        self.update_counter += 1
        self.sync_target()
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)

    def delta(self, traj: Trajectory) -> Loss:
        η = 0.001
        ω = traj.info['option'].unsqueeze(1)
        β = traj.info['beta']
        π = traj.info['action_dist']

        # ------------------------------------
        # Value gradient
        # TODO: should we pass through feature generator again?
        #   if yes, just use: value_loss, info = super(DQN, self).delta(traj)
        #   with actions replaced by options
        x = traj.x0
        values = self.predict(x)[torch.arange(x.size(0)), ω]
        targets = self.n_step_target(traj).detach()
        value_loss = self.criterion(values, targets)
        values = values.detach()

        # ------------------------------------
        # Policy gradient
        # TODO: re-compute targets with current net instead of target net?
        #  see PLB p. 116
        advantage = targets - values
        log_probs = torch.cat([pi.log_prob(a) for pi, a in zip(π, traj.a)])
        policy_grad = (-log_probs * advantage).mean(0)

        entropy = torch.cat([pi.entropy() for pi in π])
        entropy_reg = [o.policy.β for o in self.μ.option_space[ω.squeeze(1)]]
        entropy_reg = torch.tensor(entropy_reg, device=entropy.device)
        entropy_loss = (entropy_reg * entropy).mean(0)

        policy_loss = policy_grad - entropy_loss

        # ------------------------------------
        # Termination gradient
        termination_advantage = values - values.max()
        beta_loss = (β * (termination_advantage + η)).mean()

        loss = policy_loss + value_loss + beta_loss
        return loss, {
            'policy_grad': policy_loss.item(),
            'entropy': entropy_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'beta_loss': beta_loss.item(),
            'loss': loss.item(),
        }

    def n_step_target(self, traj: Trajectory):
        ω = traj.info['option'].unsqueeze(1)
        β = traj.info['beta']
        γ = self.gvf.continuation(traj)
        z = self.gvf.cumulant(traj)
        q = self.target_avf(traj.x1[-1])

        targets = torch.empty_like(z, dtype=torch.float)
        u = β[-1] * q[ω[-1]] + (1 - β[-1]) * q.max()
        for i in range(len(traj) - 1, -1, -1):
            u = targets[i] = z[i] + γ[i] * u
        return targets


class PixelControl(DQN):
    """ Duelling de-convolutional network for auxiliary pixel control task """

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


class UNREAL(TDAC, TDn):
    """ A version of AC that stores experience in the replay buffer """

    def __init__(self, feature, actor, replay_buffer: Replay, **kwargs):
        avf = nn.Linear(feature.feature_dim, 1)
        super().__init__(avf=avf, feature=feature, actor=actor, **kwargs)
        self.replay_buffer = replay_buffer

    def learn(self, transitions: Transitions) -> Loss:
        loss, info = super().learn(transitions)
        self.replay_buffer.feed_batch(transitions)
        return loss, info


__all__ = get_all_classes(__name__)
