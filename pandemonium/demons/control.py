from collections import OrderedDict
from functools import partial

import torch
from torch import nn

from pandemonium.demons import Loss, ParametricDemon, ControlDemon
from pandemonium.demons.offline_td import OfflineTDControl
from pandemonium.demons.offline_td import TTD, TDn, OfflineTDPrediction
from pandemonium.experience import (ER, PER, ReplayBufferMixin, Trajectory,
                                    Transitions)
from pandemonium.networks import Reshape, TargetNetMixin
from pandemonium.policies import Policy, HierarchicalPolicy, DiffPolicy
from pandemonium.policies.utils import torch_argmax_mask
from pandemonium.utilities.distributions import cross_entropy, l2_projection
from pandemonium.utilities.utilities import get_all_classes


class DuellingMixin:
    r""" Mixin for value-based control algorithms that uses two separate
    estimators for action-values and state-values.

    References
    ----------
    "Dueling Network Architectures for DRL" by Wang et al.
        https://arxiv.org/pdf/1511.06581

    """
    avf: callable
    aqf: callable
    target_avf: callable
    target_aqf: callable

    def __init__(self):
        if not isinstance(self, ControlDemon):
            raise TypeError(f'Duelling architecture is only supported'
                            f'by control algorithms')
        self.avf = nn.Linear(self.φ.feature_dim, 1)

    def predict_q(self, x, target: bool = False):
        v = self.avf(x) if not target else self.target_aqf(x)
        q = self.aqf(x) if not target else self.target_aqf(x)
        return q - (q - v).mean(1, keepdim=True)


class DQN(OfflineTDControl,
          TDn,
          ParametricDemon,
          ReplayBufferMixin,
          TargetNetMixin,
          CategoricalQ,
          DuellingMixin):
    """ Deep Q-Network with all the bells and whistles mixed in.

    References
    ----------
    "Rainbow: Combining Improvements in Deep RL" by Hessel et. al
        https://arxiv.org/pdf/1710.02298.pdf
    """

    def __init__(self,
                 feature: callable,
                 behavior_policy: Policy,
                 replay_buffer: ER,
                 target_update_freq: int = 0,
                 warm_up_period: int = None,
                 num_atoms: int = 1,
                 v_min: float = None,
                 v_max: float = None,
                 duelling: bool = False,
                 double: bool = False,
                 **kwargs):

        # Adds a replay buffer
        ReplayBufferMixin.__init__(self, replay_buffer)

        # By default, learning does not start until the replay buffer is full
        if warm_up_period is None:
            warm_up_period = replay_buffer.capacity // replay_buffer.batch_size
        self.warm_up_period = warm_up_period

        # Initialize Q-network demon
        aqf = nn.Linear(feature.feature_dim, behavior_policy.action_space.n)
        super(DQN, self).__init__(aqf=aqf, feature=feature,
                                  behavior_policy=behavior_policy, **kwargs)

        # Replaces `avf` implied by `aqf` with an independent estimator
        self.duelling = duelling
        if duelling:
            DuellingMixin.__init__(self)

        # Adds ability to approximate expected values
        # via learning a distribution
        if num_atoms > 1:
            CategoricalQ.__init__(self, num_atoms=num_atoms,
                                  v_min=v_min, v_max=v_max)

        # Value-based policies require a value function to determine the
        # preference for actions. Since we now know what our action-value
        # function is (after possibly being affected by CategoricalQ mixin),
        # we can pass it to the policy via `partial`
        self.μ.act = partial(self.μ.act, q_fn=self.aqf)

        # Adds a target network to stabilize SGD
        TargetNetMixin.__init__(self, target_update_freq)
        if self.target_aqf == self.aqf:
            # TODO: fix the distributional case
            print('Warning target aqf == aqf')
        if self.target_avf == self.avf:
            print('Warning target avf == avf')

        # Adds double Q-learning for tackling maximization bias
        self.double = double
        if double:
            assert target_update_freq > 0

    def predict_q(self, x, target: bool = False):
        return self.aqf(x) if not target else self.target_aqf(x)

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        q = self.predict_q(trajectory.x1, target=True)
        if self.double:
            if isinstance(self, PixelControl):
                # TODO: q[mask] returns flat obs
                raise NotImplementedError
            v = q[torch_argmax_mask(self.aqf(trajectory.x1), 1)].unsqueeze(-1)
        else:
            v = q.max(1, keepdim=True)[0]
        return v

    def learn(self, transitions: Transitions):

        self.store(transitions)
        self.sync_target()

        # Wait until warm up period is over
        self.update_counter += 1  # TODO: move the counter up the hierarchy?
        if self.update_counter < self.warm_up_period:
            return None, dict()

        # Learn from experience
        # TODO: differentiate between n-step and batched one-step
        #   We can EITHER sample n transitions at random from a replay buffer
        #   and do a batched one-step backup on them OR we can sample n
        #   consequent transition and do a multistep update with them.
        #   This should be controlled via `n_step` parameter passed to DQN upon
        #   initialization.
        transitions = self.replay_buffer.sample()
        if not transitions:
            return None, dict()  # not enough experience in the buffer
        trajectory = Trajectory.from_transitions(transitions)
        δ, info = self.delta(trajectory)

        # Update the priorities of the collected transitions
        if isinstance(self.replay_buffer, PER):
            self._update_priorities(trajectory, info)

        return δ, info

    def __repr__(self):
        demon = ParametricDemon().__repr__()
        params = f'(replay_buffer): {repr(self.replay_buffer)}\n' \
                 f'(warmup): {self.warm_up_period}\n' \
                 f'(target_update_freq): {self.target_update_freq}\n' \
                 f'(double): {self.double}\n' \
                 f'(duelling): {self.duelling}\n'
        return f'{demon}\n{params}'

    def __str__(self):
        return super().__str__()


class DeepSARSA(OfflineTDControl):
    r""" :math:`n`-step semi-gradient :math:`\text{SARSA}` """

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        q = self.predict_q(trajectory.x1)
        a = self.gvf.π(trajectory.x1, vf=self.aqf)
        v = q[torch.arange(q.size(0)), a]
        return v


class DeepSARSE(OfflineTDControl):
    r""" :math:`n`-step semi-gradient expected :math:`\text{SARSA}` """

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        q = self.predict_q(trajectory.x1)
        dist = self.gvf.π.dist(trajectory.x1, vf=self.aqf)
        v = q * dist.probs
        return v


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


class AC(ParametricDemon):
    """ Base class for Actor-Critic architectures that operate on batches

    References
    ----------
    https://hadovanhasselt.files.wordpress.com/2016/01/pg1.pdf
    """

    def __init__(self, behavior_policy: DiffPolicy, **kwargs):
        super().__init__(behavior_policy=behavior_policy, **kwargs)

    def critic_loss(self, trajectory: Trajectory):
        raise NotImplementedError

    def actor_loss(self, trajectory: Trajectory, weights):
        x = self.feature(trajectory.s0)
        return self.μ.delta(x, trajectory.a, weights.squeeze())

    def delta(self, trajectory: Trajectory) -> Loss:
        stats = dict()
        critic_loss, weights, info = self.critic_loss(trajectory)
        stats.update(**info)
        # TODO: what happens if we do not detach?
        actor_loss, info = self.actor_loss(trajectory, weights.detach())
        stats.update(**info)
        loss = actor_loss + 0.5 * critic_loss
        return loss, stats


class A2C(AC, OfflineTDControl):
    r""" Advantage Actor Critic

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
        x = self.feature(trajectory.s0)
        v = self.predict_q(x)[torch.arange(x.size(0)), trajectory.a][:, None]
        u = self.target(trajectory).detach()
        δ = self.criterion(v, u)
        return δ, u - v, {'td': δ.item()}


class TDAC(AC, OfflineTDPrediction, TTD):
    r""" A variation of AC that approximates advantage with :math:`\delta`

    This version is using truncated $\lamda$ returns to compute $\delta$, i.e.

    .. math::
        \Delta \theta = (v^{\lambda}_t - v_t(s)) \nabla_{\theta}log\pi_{\theta}(s_t, a_t)

    """

    def v_target(self, trajectory: Trajectory):
        return self.avf(trajectory.x1)

    def critic_loss(self, trajectory: Trajectory):
        loss, info = OfflineTDPrediction.delta(self, trajectory)
        return loss, info.pop('td_error'), info


class OC(AC, DQN):
    """ DQN style Option-Critic architecture """

    def __init__(self, actor: HierarchicalPolicy, **kwargs):
        super().__init__(
            output_dim=len(actor.option_space),
            actor=actor,
            # TODO: make a sequential replay that samples last
            #  BATCH_SIZE transitions in order
            replay_buffer=ER(size=0, batch_size=0),
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


__all__ = get_all_classes(__name__)
