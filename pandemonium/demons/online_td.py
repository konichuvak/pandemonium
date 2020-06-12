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

        # TODO: fails for distributional learning and non-FA
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
        z = self.gvf.cumulant(t)
        v = self.predict(t.x0)
        u = z + γ * self.predict(t.x1).detach()
        δ = u - v

        # Off policy importance sampling correction
        π = self.gvf.π.dist(t.x0, self.aqf).probs[0][t.a]
        b = self.μ.dist(t.x0, self.aqf).probs[t.a]
        ρ = π / b
        δ *= ρ

        info = {'td_error': δ.item()}
        if self.λ.trace_decay == 0:
            loss = F.mse_loss(input=v, target=z + γ * u)
        else:
            v.backward()  # semi-gradient
            assert self.avf.bias is None
            grad = next(self.avf.parameters()).grad
            e = self.λ(γ, grad)
            info['eligibility_norm'] = e.pow(2).sum().sqrt().item()
            with torch.no_grad():
                for param in self.avf.parameters():
                    param.grad = -δ * e
            loss = None

        return loss, info


class TrueOnlineTD(OnlineTD):
    pass


class SARSA(OnlineTD, ControlDemon, ParametricDemon):
    r""" Semi-gradient :math:`\SARSA{(\lambda)}`

    Adapts $\TD{(\lambda)}$ to the control case.
    """

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        v = self.predict_q(t.x0[0])[t.a]
        u = z + γ * self.predict_q(t.x1[0])[t.a1].detach()
        δ = u - v

        info = {'td_error': δ.item()}
        if self.λ.trace_decay == 0:
            # A shortcut for SARSA(0)
            loss = F.mse_loss(input=v, target=u)
        else:
            v.backward()  # semi-gradient
            assert self.aqf.bias is None
            grad = next(self.aqf.parameters()).grad
            e = self.λ(γ, grad)
            info['eligibility_norm'] = e.pow(2).sum().sqrt().item()
            with torch.no_grad():
                for param in self.aqf.parameters():
                    param.grad = -δ * e
            loss = None

        return loss, info


class QLearning(OnlineTD, ControlDemon, ParametricDemon):

    def __init__(self, **kwargs):
        # TODO: eligibility traces
        super(QLearning, self).__init__(trace_decay=0, **kwargs)

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        v = self.predict_q(t.x0[0])[t.a][0]
        u = z + γ * self.predict_q(t.x1[0]).max().detach()
        loss = F.mse_loss(input=v, target=u)
        return loss, {}


__all__ = get_all_classes(__name__)
