import textwrap
from abc import ABC
from typing import Tuple, Optional, Callable

import torch
from pandemonium import GVF
from pandemonium.experience import Transitions
from pandemonium.policies import Policy
from pandemonium.traces import EligibilityTrace

Loss = Tuple[Optional[torch.Tensor], dict]


class Demon:
    r""" **General Value Function Approximator**

    Each demon is an independent reinforcement learning agent responsible
    for learning one piece of predictive knowledge about the main agent’s
    interaction with its environment.

    Each demon learns an approximation to the GVF that corresponds to the
    setting of the three question functions: $π$, $γ$, and $z$. The tools that
    the demon uses to learn the approximation are called answers functions:
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
                 eligibility: EligibilityTrace,
                 ):
        super().__init__()
        self.gvf = gvf
        self.avf = avf

        self.φ = feature
        self.μ = behavior_policy
        self.λ = eligibility

    def learn(self, *args, **kwargs):
        raise NotImplementedError

    def delta(self, exp: Transitions) -> Loss:
        """ Specifies the update rule for approximate value function """
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        r""" Approximate value of a state is linear wrt features

        .. math::
            \widetilde{V}(s) = \boldsymbol{\phi}(s)^{T}\boldsymbol{w}

        """
        return self.avf(x)

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
    """ Parametrized Demons implemented in PyTorch subclass this. """

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.predict(*args, **kwargs)


class LinearDemon(ParametricDemon, ABC):

    def __init__(self, feature, output_dim: int, *args, **kwargs):
        super().__init__(
            avf=torch.nn.Linear(feature.feature_dim, output_dim),
            feature=feature, *args, **kwargs
        )


class PredictionDemon(Demon, ABC):
    r""" Collects factual knowledge about environment by learning to predict

    Can be thought of as an accumulator of declarative knowledge.

    .. note::
        the output of a predictive demon is usually a scalar, corresponding
        to the estimate of a value function at a state
    """

    # def __init__(self, output_dim: int = 1, *args, **kwargs):
    #     super().__init__(output_dim=output_dim, *args, **kwargs)


class ControlDemon(Demon, ABC):
    r""" Learns the optimal policy while learning to predict

    Can be thought of as an accumulator of procedural knowledge.
    Also goes by the name 'adaptive critic' in AC architectures.

    .. note::
        the output of a control demon, in contrast to prediction demon,
        is usually a vector corresponding to the estimate of action values
        under the target policy
    """

    # def __init__(self,
    #              behavior_policy: Policy,
    #              output_dim: int = None,
    #              *args, **kwargs):
    #     output_dim = output_dim or behavior_policy.action_space.n
    #     super().__init__(output_dim=output_dim,
    #                      behavior_policy=behavior_policy,
    #                      *args, **kwargs)

    def behavior_policy(self, x):
        # Control policies usually require access to value functions.
        return self.μ(x, vf=self)


__all__ = ['Demon', 'ParametricDemon', 'LinearDemon', 'PredictionDemon',
           'ControlDemon', 'Loss']
