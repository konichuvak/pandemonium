""" Actor-Critic methods are a little different from demons in the sense
that only the critic part is estimating some sort of GVF (i.e. doing policy
evaluation), while the actor part is responsible for policy improvement.
TODO: think about framing actors as control demons, whose input is the
    prediction from another demon (critic)
"""
from torch import nn

from pandemonium.demons import Loss, ParametricDemon, TDPrediction, TDControl
from pandemonium.experience import Experience
from pandemonium.policies import DiffPolicy


class AC(ParametricDemon):
    r""" Base class for Actor-Critic architectures.

    .. math::
        \Delta \theta = R_t \nabla_{\theta}\log\pi_{\theta}(S, A)

    where $R_t$ is estimated using some policy evaluation method.
    Different values of $R_t$ correspond to different algorithms:
        $G_t$           REINFORCE
        $q_w(S, A)$     Q Actor-Critic
        $A_w(S, A)$     Advantage Actor-Critic
        $\delta_t$      TD Actor-Critic

    # TODO: use GAE

    References
    ----------
    High-Dimensional Continuous Control Using Generalized Advantage Estimation
        Schulman et al. (2015)
        https://arxiv.org/pdf/1506.02438.pdf

    https://hadovanhasselt.files.wordpress.com/2016/01/pg1.pdf
    """

    def __init__(self, behavior_policy: DiffPolicy, **kwargs):
        super().__init__(behavior_policy=behavior_policy, **kwargs)

    def critic_loss(self, loss: Loss):
        """ Extracts advantage estimate and  """
        raise NotImplementedError

    def actor_loss(self, exp: Experience, weights):
        # weights is some kind of advantage estimate
        x = self.feature(exp.s0)  # TODO: why not use exp.x0?
        return self.μ.delta(x, exp.a, weights.squeeze())

    def delta(self, exp: Experience) -> Loss:
        stats = dict()
        loss = super().delta(exp)
        critic_loss, weights, info = self.critic_loss(loss)
        stats.update(**info)
        # TODO: what happens if do not detach? (hint: residual bellman update)
        actor_loss, info = self.actor_loss(exp, weights.detach())
        stats.update(**info)
        loss = actor_loss + 0.5 * critic_loss
        return loss, stats


class A2C(AC, TDControl):
    r""" Advantage Actor Critic.

    Uses two demons to approximate $v_{\pi_{\theta}}(s)$ and
    $q_{\pi_{\theta}}(s, a)$ at the same time.
    """

    def __init__(self, feature, behavior_policy, **kwargs):
        super().__init__(
            avf=nn.Linear(feature.feature_dim, 1),
            aqf=nn.Linear(feature.feature_dim, behavior_policy.action_space.n),
            behavior_policy=behavior_policy,
            feature=feature, **kwargs
        )

    def critic_loss(self, loss: Loss):
        loss, info = loss
        return loss, info.pop('td_error'), info


class TDAC(AC, TDPrediction):
    r""" Actor-Critic that approximates advantage with :math:`\TD` error.

    .. math::
        \Delta \theta = \delta_t \nabla_{\theta}\log\pi_{\theta}(s_t, a_t)

    """

    def critic_loss(self, loss: Loss):
        loss, info = loss
        return loss, info.pop('td_error'), info
