from copy import deepcopy
from typing import List

import torch

from pandemonium.demons import ControlDemon
from pandemonium.demons.prediction import TDn
from pandemonium.demons.td import TemporalDifference
from pandemonium.experience import Transition, Trajectory, Transitions
from pandemonium.policies.gradient import PolicyGradient
from pandemonium.utilities.replay import Replay


class DQN(TemporalDifference, ControlDemon):
    """

    .. todo::
        - parametrize config
        - add various options for replay buffer
        - n-step returns

    """

    def __init__(self, replay_buffer: Replay, device: torch.device, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.i = 0  # batch counter
        self.warmup = 10  # in batches
        self.target_update_freq = 200  # in batches

        # Create a target networks to stabilize training
        self.target_feature_net = deepcopy(self.φ)
        self.target_feature_net.load_state_dict(self.φ.state_dict())

        self.target_net = deepcopy(self.value_head)
        self.target_net.load_state_dict(self.value_head.state_dict())

        # Set both feature and prediction nets to evaluation mode
        # self.target_net.eval()
        # self.target_feature_net.eval()

        # Use replay buffer for breaking correlation in the experience samples
        self.replay_buffer = replay_buffer

        self.optimizer = torch.optim.Adam(self.parameters(), 0.001)
        self.to(device)

    def delta(self, *args, **kwargs):
        batch = Trajectory.from_transitions(zip(*self.replay_buffer.sample()))

        q = self.predict(batch.s0).gather(1, batch.a.unsqueeze(1)).squeeze()
        next_q = self.target_predict(batch.s1).max(1)[0]

        gamma = self.gvf.continuation(batch)
        target_q = batch.r + gamma * next_q
        loss = torch.functional.F.smooth_l1_loss(q, target_q)
        return loss

    @torch.no_grad()
    def target_predict(self, s: torch.Tensor):
        return self.target_net(self.target_feature_net(s))

    def learn(self, exp: List[Transition]):

        self.i += 1
        self.replay_buffer.feed_batch(exp)

        if self.i == self.warmup:
            print('learning_starts')
        if self.i > self.warmup:
            super().learn(exp)
        if self.i % self.target_update_freq == 0:
            self.target_feature_net.load_state_dict(self.φ.state_dict())
            self.target_net.load_state_dict(self.value_head.state_dict())

    def __str__(self):
        # model = torch.nn.Module.__str__(self)[:-2]
        demon = ControlDemon.__str__(self)
        buffer = f'  (replay_buffer): {self.replay_buffer}'
        hyperparams = f'  (hyperparams):\n' \
                      f'    (warmup): {self.warmup}\n' \
                      f'    (target_update_freq): {self.target_update_freq}\n'

        return f'{demon}\n' \
               f'{buffer}\n' \
               f'{hyperparams}\n)'


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


class AC(TDn):
    """ Actor-Critic architecture """

    def __init__(self, actor: PolicyGradient, device, *args, **kwargs):
        super().__init__(behavior_policy=actor, *args, **kwargs)

        # This includes parameters for feature generator, actor and critic
        self.optimizer = torch.optim.Adam(self.parameters(), 0.001)
        self.to(device)

    def delta(self, traj: Trajectory, *args, **kwargs):
        # Value gradient
        targets = self.n_step_target(traj).detach()
        x = self.feature(traj.s0)
        values = self.value_head(x).squeeze(1)
        value_loss = torch.functional.F.smooth_l1_loss(values, targets)

        # Policy gradient
        advantages = targets - values.detach()
        policy_loss = self.μ.delta(x, traj.a, advantages)

        # Weighted loss
        loss = policy_loss + 0.5 * value_loss
        return loss

    def learn(self, transitions: Transitions):
        traj = Trajectory.from_transitions(transitions)
        super().learn(traj)
