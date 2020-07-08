from copy import deepcopy
from functools import partial
from random import random

import torch

from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons import Loss
from pandemonium.demons.control import (QLearning, OfflineTDControl,
                                        OnlineTDControl)
from pandemonium.demons.offline_td import TTD
from pandemonium.experience import Transition
from pandemonium.policies import Greedy
from pandemonium.policies.utils import torch_argmax_mask
from pandemonium.utilities.utilities import get_all_classes


class MultistepQLearning(QLearning, OfflineTDControl, TTD):
    ...


class OnlineQLearning(QLearning, OnlineTDControl):
    """ Simple online Q-learning. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DoubleQLearning(OnlineQLearning):
    """ Implements online version of Double Q-learning. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q1 = self.aqf
        self.q2 = deepcopy(self.aqf)
        self.q2.load_state_dict(self.q1.state_dict())
        self.μ.act = partial(self.μ.act, q_fn=lambda φ: self.q1(φ) + self.q2(φ))

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        x = self.feature(t.x0)
        q1, q2 = (self.q1, self.q2) if random() > 0.5 else (self.q2, self.q1)
        q_tm1 = q1(x)[torch.arange(t.a.size(0)), t.a]
        q_t = (torch_argmax_mask(q2(t.x1), 1) * q1(t.x1)).sum(1).detach()
        δ = z + γ * q_t - q_tm1
        loss = self.criterion(input=q_tm1, target=z + γ * q_t)
        return loss, {'td_error': δ.item()}


def create_horde(config, env, φ, μ) -> Horde:
    aqf = torch.nn.Linear(φ.feature_dim, env.action_space.n, bias=False)
    torch.nn.init.zeros_(aqf.weight)

    control_demon = OnlineQLearning(
        gvf=GVF(
            target_policy=Greedy(
                feature_dim=φ.feature_dim,
                action_space=env.action_space
            ),
            cumulant=Fitness(env),
            continuation=ConstantContinuation(config['gamma'])
        ),
        aqf=aqf,
        feature=φ,
        behavior_policy=μ,
        trace_decay=config['trace_decay'],
    )

    return Horde(
        demons=[control_demon],
        device=torch.device('cpu')
    )


__all__ = get_all_classes(__name__)
