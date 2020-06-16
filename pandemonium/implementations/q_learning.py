from copy import deepcopy
from random import random

import torch

from pandemonium.demons import Loss
from pandemonium.demons.control import (QLearning, OfflineTDControl,
                                        OnlineTDControl)
from pandemonium.demons.offline_td import TDn
from pandemonium.experience import Transition
from pandemonium.policies.utils import torch_argmax_mask


class MultistepQLearning(QLearning, OfflineTDControl, TDn):
    # TODO: eligibility
    ...


class OnlineQLearning(QLearning, OnlineTDControl):
    ...


class DoubleQLearning(QLearning, OnlineTDControl):
    """ Implements online version of Double Q-learning. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q1 = self.aqf
        self.q2 = deepcopy(self.aqf)
        self.q2.load_state_dict(self.q1.state_dict())

        # Remove dysfunctional methods
        del self.q_tm1
        del self.q_t
        del self.target

    def behavior_policy(self, x: torch.Tensor):
        return self.μ(x, q_fn=lambda φ: self.q1(φ) + self.q2(φ))

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
