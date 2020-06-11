from typing import Type

import torch
import torch.nn.functional as F

from pandemonium.demons import (PredictionDemon, Demon, Loss, ControlDemon,
                                ParametricDemon)
from pandemonium.experience import Transition
from pandemonium.traces import EligibilityTrace, AccumulatingTrace
from pandemonium.utilities.utilities import get_all_classes


class OnlineTD(Demon):
    r""" Base class for backward-view (online) :math:`\TD` methods. """

    def __init__(self,
                 trace_decay: float,
                 eligibility: Type[EligibilityTrace] = AccumulatingTrace,
                 **kwargs):
        super().__init__(eligibility=None, **kwargs)

        # TODO: fails for non-parametric estimators
        if isinstance(self, PredictionDemon):
            trace_dim = next(self.avf.parameters()).shape
        elif isinstance(self, ControlDemon):
            trace_dim = next(self.aqf.parameters()).shape
        else:
            raise TypeError(self)
        self.λ = eligibility(trace_decay, trace_dim)

    def delta(self, t: Transition) -> Loss:
        """ Specifies the update rule for approximate value function (avf)

        Since the algorithms in this family are online, the update rule is
        applied on every `Transition`.
        """
        raise NotImplementedError

    def learn(self, t: Transition):
        assert len(t) == 1
        return self.delta(t[0])


class TD(OnlineTD, PredictionDemon, ParametricDemon):
    r""" Semi-gradient :math:`\TD{(\lambda)}` rule for estimating :math:`\tilde{v} \approx v_{\pi}`

    .. math::
        \begin{align*}
            e_t &= γ_t λ e_{t-1} + \nabla \tilde{v}(x_t) \\
            w_{t+1} &= w_t + \alpha (z_t + γ_{t+1} \tilde{v}(x_{t+1}) - \tilde{v}(x_t))e_t
        \end{align*}
    """

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        Q_tm1 = self.predict(t.x0)
        Q_t = self.predict(t.x1).detach()
        δ = z + γ * Q_t - Q_tm1

        if self.λ.trace_decay == 0:
            loss = F.mse_loss(input=Q_tm1, target=z + γ * Q_t)
        else:
            Q_tm1.backward()  # semi-gradient
            assert self.avf.bias is None
            grad = next(self.avf.parameters()).grad
            e = self.λ(γ, grad)
            with torch.no_grad():
                for param in self.avf.parameters():
                    param.grad = -δ * e
            loss = None

        return loss, {'td_error': δ.item()}


class TrueOnlineTD(OnlineTD):
    pass


class SARSA(OnlineTD, ControlDemon, ParametricDemon):
    r""" Semi-gradient :math:`\SARSA{(\lambda)}`

    .. math::
        \begin{align*}
            e_t &= γ_t λ e_{t-1} + \nabla \tilde{q}(x_t, a_t) \\
            w_{t+1} &= w_t + \alpha (z_t + γ_{t+1} \tilde{Q}(x_{t+1}, a_{t+1}) - \tilde{v}(x_t, a_t))e_t
        \end{align*}
    """

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        Q_tm1 = self.predict_q(t.x0[0])[t.a]
        Q_t = self.predict_q(t.x1[0])[t.a1].detach()
        δ = z + γ * Q_t - Q_tm1

        if self.λ.trace_decay == 0:
            # A shortcut for SARSA(0)
            loss = F.mse_loss(input=Q_tm1, target=z + γ * Q_t)
        else:
            Q_tm1.backward()  # semi-gradient
            assert self.aqf.bias is None
            grad = next(self.aqf.parameters()).grad
            e = self.λ(γ, grad)
            with torch.no_grad():
                for param in self.aqf.parameters():
                    param.grad = -δ * e
            loss = None

        return loss, {'td_error': δ.item()}


class QLearning(OnlineTD, ControlDemon, ParametricDemon):

    def __init__(self, **kwargs):
        # TODO: eligibility traces
        super(QLearning, self).__init__(trace_decay=0, **kwargs)

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        q = self.predict_q(t.x0[0])[t.a][0]
        _q = self.predict_q(t.x1[0]).max().detach()
        loss = F.mse_loss(input=q, target=z + γ * _q)
        return loss, {}


__all__ = get_all_classes(__name__)
