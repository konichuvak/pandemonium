from typing import Type

import torch
import torch.nn.functional as F

from pandemonium.demons import (PredictionDemon, Demon, Loss, ControlDemon,
                                ParametricDemon)
from pandemonium.experience import Transition
from pandemonium.policies import Egreedy
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

    References
    ----------
    "Reinforcement Learning: An Introduction"
        Sutton and Barto (2018) ch. 12.2
        http://incompleteideas.net/book/the-book.html
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


class OnlineTDControl(OnlineTD, ControlDemon, ParametricDemon):
    r""" Base class for online :math:`\TD` methods for control tasks. """

    @torch.no_grad()
    def q_t(self, t: Transition):
        """ Estimates the action-value of the next state. """
        raise NotImplementedError

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        v = self.predict_q(t.x0[0])[t.a]
        u = z + γ * self.q_t(t)
        δ = u - v

        info = {'td_error': δ.item()}
        if self.λ.trace_decay == 0:
            loss = F.mse_loss(input=v, target=u)  # a shortcut
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


class SARSA(OnlineTDControl):
    r""" Semi-gradient :math:`\SARSA{(\lambda)}`.

    References
    ----------
    "Reinforcement Learning: An Introduction"
        Sutton and Barto (2018) ch. 12.7
        http://incompleteideas.net/book/the-book.html

    """

    @torch.no_grad()
    def q_t(self, t: Transition):
        q = self.predict_q(t.x1)[0]
        return q[t.a1]


class SARSE(OnlineTDControl):
    r""" Semi-gradient Expected :math:`\SARSA{(\lambda)}`.

    References
    ----------
    "Reinforcement Learning: An Introduction"
        Sutton and Barto (2018) ch. 6.6
        http://incompleteideas.net/book/the-book.html

    "A Theoretical and Empirical Analysis of Expected Sarsa"
        Harm van Seijen et al (2009)
        http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf
    """

    @torch.no_grad()
    def q_t(self, t: Transition):
        q = self.predict_q(t.x1)[0]
        dist = self.μ.dist(t.x1, q_fn=self.aqf)  # use behaviour policy
        return q.dot(dist.probs[0])


class QLearning(OnlineTDControl):
    r""" Off-policy version of :math:`\SARSE`.

    Here we use a generalized Q-learning update rule, inspired by $\SARSE$.
    Since the target policy $\pi$ in canonical Q-learning is greedy wrt to GVF,
    we have the following equality:

    .. math::
        \max_\limits{a \in \mathcal{A}}Q(S_{t+1}, a) = \sum_{a \in \mathcal{A}} \pi(a|S_{t+1})Q(S_{t+1}, a)

    .. todo::
        Add support for eligibility traces
    """

    def __init__(self, **kwargs):
        super(QLearning, self).__init__(trace_decay=0, **kwargs)
        if not isinstance(self.gvf.π, Egreedy):
            raise TypeError(self.gvf.π)
        elif self.gvf.π.ε == 0:
            raise ValueError(self.gvf.π.ε)

    @torch.no_grad()
    def q_t(self, t: Transition):
        q = self.predict_q(t.x1)[0]
        dist = self.gvf.π.dist(t.x1, q_fn=self.aqf)  # use target policy
        return q.dot(dist.probs[0])


__all__ = get_all_classes(__name__)
