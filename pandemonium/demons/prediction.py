from pandemonium.demons.demon import PredictionDemon
from pandemonium.demons.td import TemporalDifference
from pandemonium.experience import Transition


class TD(TemporalDifference, PredictionDemon):
    r""" Semi-gradient $TD(\lambda)$ rule for estimating $\tilde{v}$

    .. math::
        e_t = γ_t λ e_{t-1} + \Nabla \Tilde{v}(x_t)
        w_{t+1} = w_t} + α(z_t + γ_{t+1}\Tilde{v}(x_{t+1}) - \Tilde{v}(x_t))e_t

    In case when $\lambda=0$ we recover one-step TD algorithm:

    .. math::
        e_t = \Nabla \Tilde{v}(x_t)
        w_{t+1} = w_t + α(z_t + γ_{t+1}V(x_{t+1}) - V(x_t})) e_t

    .. todo::
        batch

    """

    def delta(self, t: Transition):
        e = self.eligibility(self.gvf.γ(features=t.x0, done=t.done),
                             self.grad(t.x0))
        γ = self.gvf.γ(features=t.x1, done=t.done)
        u = self.gvf.z(t) + γ * self.predict(x=t.x1)
        return (u - self.predict(t.x0)) * e
