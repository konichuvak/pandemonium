import textwrap
from typing import Tuple, Optional, Callable

import torch

from pandemonium import GVF
from pandemonium.experience import Experience
from pandemonium.policies import Policy
from pandemonium.traces import EligibilityTrace

Loss = Tuple[Optional[torch.Tensor], dict]


class Demon:
    r""" **General Value Function Approximator**

    Each demon is an independent reinforcement learning agent responsible
    for learning one piece of predictive knowledge about the main agent’s
    interaction with its environment.

    Demon learns an approximate value function $\tilde{V}$ (avf), to the
    general value function (gvf) that corresponds to the setting of the
    three "question" functions: $\pi$, $\gamma$, and z. The tools that the demon
    uses to learn the approximation are called "answer" functions and are
    comprised of $\mu$, $\phi$ and $\lambda$.

    Attributes
    ----------
    gvf:
        General Value Function to be estimated by the demon
    avf:
        Approximate Value Function learned by the demon to approximate `gvf`
    φ:
        Feature generator learning useful state representations
    μ:
        Behavior policy that collects experience
    λ:
        Eligibility trace assigning credit to experiences

    """

    def __init__(self,
                 gvf: GVF,
                 avf: Callable,
                 feature,
                 behavior_policy: Policy,
                 eligibility: Optional[EligibilityTrace],
                 ):
        self.gvf = gvf
        self.avf = self.target_avf = avf

        self.φ = self.target_feature = feature
        self.μ = behavior_policy
        self.λ = eligibility

    def learn(self, experience: Experience):
        return self.delta(experience)

    def delta(self, experience: Experience) -> Loss:
        """ Specifies the update rule for approximate value function (avf)

        Depending on whether the algorithm is online or offline, the demon will
        be learning from a single `Transition` vs a `Trajectory` of experiences.
        """
        raise NotImplementedError

    def predict(self, x):
        r""" Predict the value (or value distribution) of the state """
        return self.avf(x)

    def feature(self, *args, **kwargs):
        r""" A mapping from MDP states to features

        .. math::
            \phi: \mathcal{S} \mapsto \mathbb{R}^n

        Feature tensor could be constructed from the robot’s external
        sensor readings (not just the ones corresponding to light).

        We can use any representation learning module here.
        """
        return self.φ(*args, **kwargs)

    def behavior_policy(self, s):
        r""" Specifies behavior of the agent

        .. math::
            \mu: \mathcal{S} \times \mathcal{A} \mapsto [0, 1]

        The distribution across all possible motor commands of the agent
        could be specified in this way.
        """
        return self.μ(s)

    def eligibility(self, s):
        r""" Specifies eligibility trace-decay rate

        .. math::
            \lambda: \mathcal{S} \mapsto \mathbb{R}
        """
        return self.λ(s)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        Γ = textwrap.indent(repr(self.gvf), "\t")
        φ = textwrap.indent(repr(self.φ), "\t\t")
        μ = textwrap.indent(repr(self.μ), "\t\t")
        λ = textwrap.indent(repr(self.λ), "\t\t")
        return f'{self.__class__.__name__}(\n' \
               f'{Γ}\n' \
               f'\t(φ):\n {φ}\n' \
               f'\t(μ):\n {μ}\n' \
               f'\t(λ):\n {λ}\n' \
               f')'


class PredictionDemon(Demon):
    r""" Collects factual knowledge about environment by learning to predict

    Can be thought of as an accumulator of declarative knowledge.
    """


class ControlDemon(Demon):
    r""" Learns the optimal policy while learning to predict

    Can be thought of as an accumulator of procedural knowledge.

    In addition to the approximate value function (avf), has a an approximate
    Q-value function (aqf) that produces value estimates for state-action pairs.
    """

    def __init__(self, aqf: Callable, avf: Callable = None, **kwargs):
        self.aqf = self.target_aqf = aqf
        super().__init__(avf=self.implied_avf if avf is None else avf, **kwargs)

    def implied_avf(self, x):
        r""" State-value function in terms of action-value function

        .. math::
            V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi (a|s) * Q^{\pi}(a,s)

        Is overridden in duelling architecture by an independent estimator.

        TODO: does not apply for continuous action spaces
        TODO: handle predictions made to compute targets via target_aqf
        """
        return (self.μ.dist(x, vf=self.aqf).probs * self.aqf(x)).sum(1)

    def predict_q(self, x):
        """ Computes action-values in a given state. """
        return self.aqf(x)

    def predict_target_q(self, x):
        """ Computes target action-values in a given state. """
        return self.target_aqf(x)

    def predict_adv(self, x):
        """ Computes the advantage in a given state. """
        return self.aqf(x) - self.avf(x)

    def predict_target_adv(self, x):
        """ Computes the target advantage in a given state. """
        return self.target_aqf(x) - self.target_aqf(x)

    def behavior_policy(self, x: torch.Tensor):
        # Control policies usually require access to value functions.
        return self.μ(x, q_fn=self.aqf)


class ParametricDemon(Demon, torch.nn.Module):
    """ Base class fot parametrized Demons implemented in PyTorch """

    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        super().__init__(**kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.predict(*args, **kwargs)


class LinearDemon(ParametricDemon):
    r""" Approximates state or state-action values using linear projection

    .. math::
        \widetilde{V}(s) = \boldsymbol{x}(s)^{T}\boldsymbol{w}
    """

    def __init__(self, feature, output_dim: int, *args, **kwargs):
        super().__init__(
            avf=torch.nn.Linear(feature.feature_dim, output_dim),
            feature=feature, **kwargs
        )


__all__ = ['Demon', 'ParametricDemon', 'LinearDemon', 'PredictionDemon',
           'ControlDemon', 'Loss']
