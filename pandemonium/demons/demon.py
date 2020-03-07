import textwrap
from typing import Tuple, Optional

import torch

from pandemonium import GVF
from pandemonium.experience import Transitions
from pandemonium.policies import Policy
from pandemonium.traces import EligibilityTrace

Loss = Tuple[Optional[torch.Tensor], dict]


class Demon(torch.nn.Module):
    r""" **General Value Function Approximator**

    Each demon is an independent reinforcement learning agent responsible
    for learning one piece of predictive knowledge about the main agent’s
    interaction with its environment.

    Each demon learns an approximation to the GVF that corresponds to the
    setting of the three question functions: $π$, $γ$, and $z$. The tools that
    the demon uses to learn the approximation are called answers functions:
        - $\phi$ (feature generator): learning useful state representations
        - $\mu$ (behaviour policy) : collecting experience
        - $\lambda$ (eligibility trace): assigning credit to experiences

    """

    def __init__(self,
                 gvf: GVF,
                 feature,
                 behavior_policy: Policy,
                 eligibility: EligibilityTrace,
                 output_dim: int,
                 ):
        super().__init__()
        self.gvf = gvf

        self.φ = feature
        self.μ = behavior_policy
        self.λ = eligibility

        self.value_head = torch.nn.Linear(feature.feature_dim, output_dim)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.predict(*args, **kwargs)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        r""" Approximate value of a state is linear wrt features

        .. math::
            \widetilde{V}(s) = \boldsymbol{\phi}(s)^{T}\boldsymbol{w}

        """
        return self.value_head(x)

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
        r""" Specifies behaviour of the agent

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

    def delta(self, exp: Transitions) -> Loss:
        """ Specifies the loss function, i.e. TD error """
        raise NotImplementedError

    def learn(self, *args, **kwargs):
        raise NotImplementedError

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

    .. note::
        the output of a predictive demon is always a scalar, corresponding
        to the estimate of a value function at a state
    """

    def __init__(self, *args, **kwargs):
        super().__init__(output_dim=1, *args, **kwargs)


class ControlDemon(Demon):
    r""" Learns the optimal policy while learning to predict

    Can be thought of as an accumulator of procedural knowledge.

    .. note::
        the output of a control demon, on the other hand, is always a vector
        corresponding to the estimate of action values under the target policy
    """

    def __init__(self,
                 behavior_policy: Policy,
                 output_dim: int = None,
                 *args, **kwargs):
        from gym.spaces import Discrete
        if not isinstance(behavior_policy.action_space, Discrete):
            raise NotImplementedError
        output_dim = output_dim or behavior_policy.action_space.n
        super().__init__(output_dim=output_dim,
                         behavior_policy=behavior_policy,
                         *args, **kwargs)

    def behavior_policy(self, state):
        # Control policies usually require access to value functions.
        return self.μ(state, vf=self)


__all__ = ['Demon', 'PredictionDemon', 'ControlDemon', 'Loss']
