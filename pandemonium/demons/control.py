from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from pandemonium.demons import (ControlDemon, Demon, Loss, ParametricDemon,
                                LinearDemon)
from pandemonium.demons.prediction import TDn
from pandemonium.demons.td import TemporalDifference
from pandemonium.experience import Transition, Trajectory, Transitions
from pandemonium.policies import HierarchicalPolicy
from pandemonium.policies.gradient import DiffPolicy
from pandemonium.utilities.replay import Replay
from pandemonium.utilities.utilities import get_all_classes
from torch import nn


class Sarsa(TemporalDifference, ControlDemon):
    r""" One-step semi-gradient :math:`SARSA` for estimating :math:`\tilde{q}`

    On-policy control method suitable for episodic tasks.
    """

    def delta(self, exp: List[Transition]):
        batch = Trajectory.from_transitions(exp)

        q = self.predict(batch.s0).gather(1, batch.a.unsqueeze(1)).squeeze(1)
        next_a = self.behavior_policy(batch.s1).sample().unsqueeze(1)
        next_q = self.predict(batch.s1).gather(1, next_a).squeeze(1)

        gamma = self.gvf.continuation(batch)
        target_q = batch.r + gamma * next_q
        loss = F.smooth_l1_loss(q, target_q)
        return loss


class SarsaN(ControlDemon, TDn):
    r""" :math:`n`-step semi-gradient :math:`SARSA` for estimating :math:`\tilde{q}`

    This is a classic on-policy control method suitable for episodic tasks.

    .. note::
        $n$-step returns are calculated in an unusual way!
        Each $(S, A)$ tuple in the trajectory gets updated towards a different
        $n$-step return where each $\{n, n-1, \dots, 1\}$ $(S, A)$ of the
        trajectory is updated using $\{1, 2, \dots, n\}$-step returns
        respectively.

    .. todo::
        - implement true n-step return (each state gets updated towards n-step
        return)
        - alternatively, subdivide the trajectory into n-batches instead of
        assuming that n = batch size


    """

    def delta(self, traj: Trajectory):
        true_n_step = False
        if true_n_step:
            q = self.predict(traj.s0[0])[traj.a[0]]
            u = self.n_step_target(traj)[0]
        else:
            q = self.predict(traj.s0).gather(1, traj.a.unsqueeze(1)).squeeze(1)
            u = self.n_step_target(traj)

        δ = F.smooth_l1_loss(q, u.detach())
        return δ, {'value_loss', δ.item()}


class DQN(TemporalDifference, ParametricDemon, ControlDemon):
    """ Deep Q-Network

    ... in all of its incarnations.
    """

    # TODO: manually resolve inheritance order?

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
            self.target_avf = deepcopy(self.avf)
            self.target_feature.load_state_dict(self.φ.state_dict())
            self.target_avf = deepcopy(self.avf)
            self.target_feature.eval()
            self.target_avf.eval()
        else:
            self.target_feature = self.φ
            self.target = self.avf

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

    def delta(self, traj: Trajectory) -> Loss:
        v = self.predict(traj.s0).gather(1, traj.a.unsqueeze(1)).squeeze()
        u = self.n_step_target(traj).detach()
        δ = nn.functional.smooth_l1_loss(v, u)
        return δ, {'value_loss': δ.item()}

    def n_step_target(self, traj: Trajectory):
        γ = self.gvf.continuation(traj)
        z = self.gvf.cumulant(traj)
        x = self.target_feature(traj.s1[-1, None])
        # a = self.gvf.π(x, vf=self.target_net)
        # TODO: Use actions `a` taken by the target policy instead of max
        v = self.target_avf(x).max(1)[0]
        u = torch.empty_like(z, dtype=torch.float)
        for i in range(len(traj) - 1, -1, -1):
            v = u[i] = z[i] + γ[i] * v
        return u.flip(0)

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


class AC(TDn, LinearDemon):
    """ Actor-Critic architecture """

    def __init__(self, actor: DiffPolicy, output_dim: int = 1, **kwargs):
        super().__init__(
            behavior_policy=actor,
            output_dim=output_dim,
            **kwargs)

    def behavior_policy(self, x):
        return self.μ(x, self.avf)

    def delta(self, traj: Trajectory) -> Loss:
        # Value gradient
        targets = self.n_step_target(traj).detach()
        x = self.feature(traj.s0)
        values = self.avf(x).squeeze(1)
        value_loss = F.smooth_l1_loss(values, targets)

        # Policy gradient
        advantages = targets - values.detach()
        policy_loss, info = self.μ.delta(x, traj.a, advantages)

        # Weighted loss
        loss = policy_loss + 0.5 * value_loss
        info.update({'value_loss': value_loss.item(), 'loss': loss.item()})
        return loss, info

    def learn(self, transitions: Transitions):
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)


class OC(AC, DQN):
    """ DQN style Option-Critic architecture """

    def __init__(self,
                 actor: HierarchicalPolicy,
                 target_update_freq: int, **kwargs):
        super().__init__(
            output_dim=len(actor.option_space),
            actor=actor,
            replay_buffer=Replay(memory_size=0, batch_size=0),
            target_update_freq=target_update_freq,
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

    def delta(self, traj: Trajectory) -> Loss:
        η = 0.001
        ω = traj.info['option'].unsqueeze(1)
        β = traj.info['beta']
        π = traj.info['action_dist']

        # ------------------------------------
        # Value gradient
        targets = self.n_step_target(traj).detach()
        values = self.value_head(traj.x0).gather(1, ω).squeeze(1)
        value_loss = F.smooth_l1_loss(values, targets)
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

        targets = torch.empty_like(z, dtype=torch.float)
        target = self.target_predict(traj.s1[-1, None]).squeeze()
        target = β[-1] * target[ω[-1]] + (1 - β[-1]) * target.max()
        for i in range(len(traj) - 1, -1, -1):
            target = targets[i] = z[i] + γ[i] * target
        return targets

    def learn(self, transitions: Transitions):
        self.update_counter += 1
        self.sync_target()
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)


class PixelControl(DQN):
    """ Duelling de-convolutional network for auxiliary pixel control task

    .. todo::
        use ModuleDict or some other container for target networks

    """

    def __init__(self,
                 feature,
                 replay_buffer: Replay,
                 output_dim: int,
                 **kwargs):

        class DeconvNet(nn.Module):

            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(feature.feature_dim, 9 * 9 * 32)
                self.deconv_v = nn.ConvTranspose2d(32, 1, 4, 2)
                self.deconv_a = nn.ConvTranspose2d(32, output_dim, 4, 2)
                # deconv2d_size_out(6, 2, 1) == 7 (7x7 observation in minigrids)
                # deconv2d_size_out(9, 4, 2) == 20 (20x20 avg pooled pixel change vals)

            def forward(self, x: torch.Tensor):
                x = self.fc(x).view(-1, 32, 9, 9)
                value = F.relu(self.deconv_v(x))
                advantage = F.relu(self.deconv_a(x))
                pc_q = value + advantage - advantage.mean(1, keepdim=True)
                return pc_q

        super().__init__(
            avf=DeconvNet(),
            feature=feature,
            warm_up_period=replay_buffer.capacity // replay_buffer.batch_size,
            replay_buffer=replay_buffer,
            **kwargs
        )

    def delta(self, traj: Trajectory) -> Loss:
        x = self.feature(traj.s0)
        v = self.predict(x)[list(range(len(traj))), traj.a]
        u = self.n_step_target(traj).detach()
        δ = F.mse_loss(v, u, reduction='mean')
        return δ, {'value_loss': δ.item(), 'pc_v': v}


__all__ = get_all_classes(__name__)
