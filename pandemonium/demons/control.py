from copy import deepcopy
from typing import List

import torch
from pandemonium.demons import ControlDemon
from pandemonium.demons.td import TemporalDifference
from pandemonium.experience import Transition, Trajectory
from pandemonium.utilities.replay import Replay


class Q1(TemporalDifference, ControlDemon):
    """

    .. todolist::
        parametrize config
        add various options for replay buffer

    """

    def __init__(self, replay_buffer: Replay, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Batch counter
        self.i = 0

        # Create a target networks to stabilize training
        self.target_feature_net = deepcopy(self.φ)
        if hasattr(self.φ, 'state_dict'):
            self.target_feature_net.load_state_dict(self.φ.state_dict())

        self.target_net = deepcopy(self.head)
        self.target_net.load_state_dict(self.head.state_dict())

        # Set both feature and prediction nets to evaluation mode
        # self.target_net.eval()
        # self.target_feature_net.eval()

        # Use replay buffer for breaking correlation in the experience samples
        self.replay_buffer = replay_buffer

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
        WARMUP = 10
        TARGET_UPDATE_FREQ = 200

        self.i += 1
        self.replay_buffer.feed_batch(exp)

        if self.i == WARMUP:
            print('learning_starts')
        if self.i > WARMUP:
            super().learn(exp)
        if self.i % TARGET_UPDATE_FREQ == 0:
            if hasattr(self.φ, 'state_dict'):
                self.target_feature_net.load_state_dict(self.φ.state_dict())
            self.target_net.load_state_dict(self.head.state_dict())


class Sarsa1(TemporalDifference, ControlDemon):
    r""" One-step semi-gradient sarsa for estimating $\tilde{q}$

    On-policy control method suitable for episodic tasks.

    .. TODO::
        generalize to n-step target

    """

    def delta(self, exp: List[Transition]):
        batch = Trajectory.from_transitions(exp)

        q = self.predict(batch.s0).gather(1, batch.a.unsqueeze(1)).squeeze(1)
        next_a = self.behavior_policy(batch.s1).unsqueeze(1)
        next_q = self.predict(batch.s1).gather(1, next_a).squeeze(1)

        gamma = self.gvf.continuation(batch)
        target_q = batch.r + gamma * next_q
        loss = torch.functional.F.smooth_l1_loss(q, target_q)
        return loss
