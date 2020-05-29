from torch.distributions.distribution import Distribution

from pandemonium.continuations import ContinuationFunction
from pandemonium.cumulants import Cumulant
from pandemonium.policies import Policy


class GVF:
    r""" **General Value Function**

    Consider a stream of data $\{ (x_t, A_t) \}^{\infty}_{t=0}$, produced
    by agent-environment interaction. Here, $x$ is a tensor of experience
    (see :class:`pandemonium.experience.Transition`) and $A$ is an action
    from a finite action space $\mathcal{A}$.

    The target $G$ is a summary of the future value of the cumulant
    $Z$, discounted according to the termination function $\gamma$:

    .. math::
        G_t = Z_{t+1} + \sum_{\tau=t+1}^{\infty} \gamma_{\tau} Z_{\tau+1}

    GVF estimates the expected value of the target cumulant,
    given actions are generated according to the target policy:

    .. math::
        \mathbb{E}_π [G_t|S_t = s]

    To make things more concrete, keep in mind an example of predicting a
    robot’s light sensor as it drives around a room. We will stick to this
    example throughout definitions in this abstract class.

    .. note::
        The value produced is not necessarily scalar, i.e. in case of estimating
        an action-function(Q) we get a row vector with values corresponding to
        each possible action.
    """

    def __init__(self,
                 target_policy: Policy,
                 continuation: ContinuationFunction,
                 cumulant: Cumulant):
        # Question about the agent's interactions with the environment
        self.z = cumulant
        self.π = target_policy
        self.γ = continuation

    def target_policy(self, s) -> Distribution:
        r""" The policy, whose value we would like to learn

        .. math::
            \pi: \mathcal{S} \times \mathcal{A} \mapsto [0, 1]
        """
        return self.π(s)

    def continuation(self, s):
        r""" Outputs continuation signal based on the agent’s observation

        .. math::
            \gamma: \mathcal{S} \mapsto[0, 1] \\

        Notice that this is different from an MDP discounting factor $\gamma$
        in classic RL. Here we allow the termination to be state-dependent.

        """
        return self.γ(s)

    def cumulant(self, s):
        r""" Accumulates future values of the signal.

        .. math::
            z: \mathcal{S} \mapsto \mathbb{R}

        For example, this could be current light sensor reading of a robot.
        """
        return self.z(s)

    def __repr__(self):
        return f'GVF(\n' \
               f'\t(z): {self.z}\n' \
               f'\t(π): {self.π}\n' \
               f'\t(γ): {self.γ}\n' \
               f')'


__all__ = ['GVF']
