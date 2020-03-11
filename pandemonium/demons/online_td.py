from typing import Type

from pandemonium.demons import Demon, Loss
from pandemonium.experience import Transition
from pandemonium.traces import EligibilityTrace, AccumulatingTrace


class OnlineTD(Demon):
    r""" Base class for backward-view (online) :math:`\text{TD}` methods. """

    def __init__(self,
                 feature,
                 trace_decay: float,
                 eligibility: Type[EligibilityTrace] = AccumulatingTrace,
                 **kwargs):
        e = eligibility(trace_decay, feature.feature_dim)
        super().__init__(eligibility=e, feature=feature, **kwargs)


class TDlambda(OnlineTD):
    r""" Semi-gradient :math:`\text{TD}\lambda` rule for estimating :math:`\tilde{v} ≈ v_{\pi}`

    .. math::
        \begin{align*}
            e_t &= γ_t λ e_{t-1} + \nabla \tilde{v}(x_t) \\
            w_{t+1} &= w_t + \alpha (z_t + γ_{t+1} \tilde{v}(x_{t+1}) - \tilde{v}(x_t))e_t
        \end{align*}
    """

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        e = self.λ(γ, t.x0)
        δ = z + γ * self.predict(t.x1) - self.predict(t.x0)
        return δ * e, {'td': δ.item(), 'eligibility': e.item()}


class TD0(TDlambda):
    r""" A special case of :math:`\text{TD}\lambda` with :math:`\lambda = 0`

    $\text{TD}(0)$ is known as one-step $\text{TD}$ algorithm with
    $e_t = \nabla \tilde{v}(x_t)$.
    """

    def __init__(self, **kwargs):
        super().__init__(trace_decay=0, **kwargs)


class TrueOnlineTD(OnlineTD):
    pass
