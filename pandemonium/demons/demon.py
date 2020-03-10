import textwrap
from abc import ABC
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
    three question functions: $π$, $γ$, and $z$. The tools that the demon uses
    to learn the approximation are called answers functions:
        - $\phi$ (feature generator): learning useful state representations
        - $\mu$ (behavior policy) : collecting experience
        - $\lambda$ (eligibility trace): assigning credit to experiences

    """
    __slots__ = 'gvf', 'avf', 'φ', 'μ', 'λ'

    def __init__(self,
                 gvf: GVF,
                 avf: Callable,
                 feature,
                 behavior_policy: Policy,
                 eligibility: Optional[EligibilityTrace],
                 ):
        super().__init__()
        self.gvf = gvf
        self.avf = avf

        self.φ = feature
        self.μ = behavior_policy
        self.λ = eligibility

    def learn(self, *args, **kwargs):
        raise NotImplementedError

    def delta(self, experience: Experience) -> Loss:
        """ Specifies the update rule for approximate value function (avf)

        Depending on whether the algorithm is online or offline, the demon will
        be learning from a single `Transition` vs a `Trajectory` of experiences.
        """
        raise NotImplementedError

    def predict(self, x):
        r""" Predict the value of the state """
        raise NotImplementedError

    def feature(self, *args, **kwargs) -> torch.Tensor:
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


class ParametricDemon(Demon, torch.nn.Module, ABC):
    """ Parametrized Demons implemented in PyTorch subclass this """

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.predict(*args, **kwargs)


class LinearDemon(ParametricDemon, ABC):
    r""" Approximates state-values using linear projection

    .. math::
        \widetilde{V}(s) = \boldsymbol{x}(s)^{T}\boldsymbol{w}
    """

    def __init__(self, feature, output_dim: int, *args, **kwargs):
        super().__init__(
            avf=torch.nn.Linear(feature.feature_dim, output_dim),
            feature=feature, *args, **kwargs
        )


class PredictionDemon(Demon, ABC):
    r""" Collects factual knowledge about environment by learning to predict

    Can be thought of as an accumulator of declarative knowledge.
    """

    def predict(self, x):
        return self.avf(x)


class ControlDemon(Demon, ABC):
    r""" Learns the optimal policy while learning to predict

    Can be thought of as an accumulator of procedural knowledge.

    In addition to the approximate value function (avf), has a an approximate
    q-value function (aqf) that produces estimates for state-action pairs.
    """
    __slots__ = 'aqf'

    def __init__(self, aqf: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aqf = aqf

    def predict(self, x):
        """ Computes value of a given state """
        return self.avf(x)

    def predict_q(self, x):
        """ Computes action-values in a given state """
        return self.aqf(x)

    def predict_adv(self, x):
        """ Computes the advantage in a given state """
        return self.aqf(x) - self.avf(x)

    def behavior_policy(self, x: torch.Tensor):
        # Control policies usually require access to value functions.
        return self.μ(x, vf=self.aqf)


__all__ = ['Demon', 'ParametricDemon', 'LinearDemon', 'PredictionDemon',
           'ControlDemon', 'Loss']
