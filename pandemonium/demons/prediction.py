import torch
import torch.nn.functional as F

from pandemonium.demons import PredictionDemon, ParametricDemon, Loss
from pandemonium.demons.offline_td import OfflineTD
from pandemonium.demons.online_td import OnlineTD
from pandemonium.experience import Experience, Trajectory, Transition
from pandemonium.utilities.utilities import get_all_classes


class TDPrediction(ParametricDemon, PredictionDemon):

    @torch.no_grad()
    def v_t(self, exp: Experience):
        r""" Computes value targets :math:`V(s_{t+1})`. """
        return self.predict(exp.x1)

    def target(self, *args, **kwargs):
        raise NotImplementedError


class OnlineTDPrediction(OnlineTD, TDPrediction):
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
        v = self.predict(t.x0)
        u = self.target(t, self.v_t(t)).detach()
        δ = u - v

        # TODO: eligibility
        # TODO: abstract manual grad update

        info = {'td_error': δ.item()}
        if self.λ.trace_decay == 0:
            loss = F.smooth_l1_loss(input=v, target=u)
            info['loss'] = loss.item()
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

    def target(self, t: Transition, v: torch.Tensor):
        """ One-step TD error target. """
        γ = self.gvf.continuation(t)
        z = self.gvf.cumulant(t)
        return z + γ * v


class OfflineTDPrediction(OfflineTD, TDPrediction):
    r""" Offline :math:`\TD` for prediction tasks. """

    def delta(self, t: Trajectory) -> Loss:
        x = self.feature(t.s0)
        v = self.predict(x)
        u = self.target(t, self.v_t(t)).detach()
        loss = self.criterion(input=v, target=u, reduction='none')
        loss = (loss * t.ρ).mean()  # weighted IS
        return loss, {'loss': loss.item(), 'td_error': u - v}


__all__ = get_all_classes(__name__)
