from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from docutils.transforms.misc import Transitions
from torch import nn

from pandemonium.demons import ControlDemon, Demon, Loss
from pandemonium.demons.td import TemporalDifference
from pandemonium.experience import Transition, Trajectory
from pandemonium.policies import HierarchicalPolicy
from pandemonium.policies.gradient import DiffPolicy
from pandemonium.utilities.replay import Replay
from pandemonium.utilities.utilities import get_all_classes


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
        loss = torch.functional.F.smooth_l1_loss(q, target_q)
        return loss


class SarsaN(TemporalDifference, ControlDemon):
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

    def delta(self, transitions: List[Transition]):
        traj = Trajectory.from_transitions(transitions)

        true_n_step = False
        if true_n_step:
            q = self.predict(traj.s0[0])[traj.a[0]]
            targets = self.n_step_target(traj)[0]
        else:
            q = self.predict(traj.s0).gather(1, traj.a.unsqueeze(1)).squeeze(1)
            targets = self.n_step_target(traj)

        loss = torch.functional.F.smooth_l1_loss(q, targets)
        return loss

    @torch.no_grad()
    def n_step_target(self, traj: Trajectory):
        γ = self.gvf.continuation(traj)

        # Obtain estimate of the value function at the end of the trajectory
        last_q = self.behavior_policy(traj.s1[-1]).sample()
        target = self.predict(traj.s1[-1])[last_q]

        # Recursively compute the target for each of the transition
        targets = torch.empty_like(traj.r, dtype=torch.float)
        for i in range(len(traj) - 1, -1, -1):
            target = targets[i] = traj.r[i] + γ[i] * target

        return targets


class DQN(TemporalDifference, ControlDemon):
    """ Deep Q-Network

    ... in all of its incarnations.
    """

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
            # TODO: The name of the parameters `target_net` might clash
            #  with other DQN demons
            self.target_feature_net = deepcopy(self.φ)
            self.target_feature_net.load_state_dict(self.φ.state_dict())

            # TODO: although φ is usually shared, the value head is not always
            #  the same across different networks based on the DQN.
            #  Thus need to make a better interface for the target net to
            #  be identifiable.
            self.target_net = deepcopy(self.value_head)
            self.target_net.load_state_dict(self.value_head.state_dict())

            # Set both feature and prediction nets to evaluation mode
            # self.target_net.eval()
            # self.target_feature_net.eval()
        else:
            self.target_feature_net = self.φ
            self.target_net = self.value_head

        # Use replay buffer for breaking correlation in the experience samples
        self.replay_buffer = replay_buffer

        self.optimizer = torch.optim.Adam(self.parameters(), 0.001)

    def sync_target(self):
        if self.target_update_freq and self.update_counter % self.target_update_freq == 0:
            self.target_feature_net.load_state_dict(self.φ.state_dict())
            self.target_net.load_state_dict(self.value_head.state_dict())

    def learn(self, transitions: Transitions):
        self.update_counter += 1
        self.replay_buffer.feed_batch(transitions)
        self.sync_target()

        if self.update_counter < self.warm_up_period:
            return None, dict()

        transitions = self.replay_buffer.sample()
        trajectory = Trajectory.from_transitions(zip(*transitions))
        return self.delta(trajectory)

    def delta(self, traj: Trajectory) -> Loss:
        values = self.predict(traj.s0).gather(1, traj.a.unsqueeze(1)).squeeze()
        targets = self.n_step_target(traj)
        loss = torch.functional.F.smooth_l1_loss(values, targets)
        return loss, dict()

    def n_step_target(self, traj: Trajectory):
        γ = self.gvf.continuation(traj)
        z = self.gvf.cumulant(traj)
        targets = torch.empty_like(z, dtype=torch.float)
        target = self.target_predict(traj.s1[-1, None]).max(1)[0]
        for i in range(len(traj) - 1, -1, -1):
            target = targets[i] = z[i] + γ[i] * target
        return targets

    @torch.no_grad()
    def target_predict(self, s: torch.Tensor):
        return self.target_net(self.target_feature_net(s))

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


class AC(TemporalDifference, Demon):
    """ Actor-Critic architecture """

    def __init__(self, actor: DiffPolicy, output_dim: int = 1, **kwargs):
        super().__init__(
            behavior_policy=actor,
            output_dim=output_dim,
            **kwargs)

    def behavior_policy(self, x):
        return self.μ(x, self.value_head)

    def delta(self, traj: Trajectory) -> Loss:
        # Value gradient
        targets = self.n_step_target(traj).detach()
        x = self.feature(traj.s0)
        values = self.value_head(x).squeeze(1)
        value_loss = torch.functional.F.smooth_l1_loss(values, targets)

        # Policy gradient
        advantages = targets - values.detach()
        policy_loss, info = self.μ.delta(x, traj.a, advantages)

        # Weighted loss
        loss = policy_loss + 0.5 * value_loss
        info.update({'value_loss': value_loss.item(), 'loss': loss.item()})
        return loss, info

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
        value_loss = torch.functional.F.smooth_l1_loss(values, targets)
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
    """ Duelling de-convolutional network for auxiliary pixel control task """

    def __init__(self,
                 output_dim: int,
                 feature_dim: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_fc = nn.Linear(feature_dim, 6 * 6 * 32)
        self.pc_deconv_value = nn.ConvTranspose2d(32, 1, 2, 1)
        self.pc_deconv_advantage = nn.ConvTranspose2d(32, output_dim, 2, 1)
        # deconv2d_size_out(6, 2, 1) == 7 (7x7 observation in minigrids)

        # HACK: override pc target net
        del self.target_net
        self.target_net = self.predict

        # HACK
        del self.value_head

    def predict(self, x: torch.Tensor):
        x = self.pc_fc(x).view(-1, 32, 6, 6)
        value = F.relu(self.pc_deconv_value(x), inplace=True)
        advantage = F.relu(self.pc_deconv_advantage(x), inplace=True)
        pc_q = value + advantage - advantage.mean(1, keepdim=True)
        return pc_q

    def delta(self, *args, **kwargs) -> Loss:
        if not self.replay_buffer.is_full:
            return None, dict()
        transitions = self.replay_buffer.sample()
        traj = Trajectory.from_transitions(zip(*transitions))
        x = self.feature(traj.s0)
        values = self.predict(x)[list(range(len(traj))), traj.a]
        targets = self.n_step_target(traj)
        loss = torch.functional.F.mse_loss(values, targets, reduction='mean')
        return loss, {'value_loss': loss.item()}


__all__ = get_all_classes(__name__)
