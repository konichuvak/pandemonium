import textwrap

import torch

from pandemonium import GVF
from pandemonium.experience import Transitions
from pandemonium.policies import Policy
from pandemonium.traces import EligibilityTrace


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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict(state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        r""" Approximate value of a state is linear wrt features

        .. math::
            \widetilde{V}(s) = \boldsymbol{\phi}(s)^{T}\boldsymbol{w}

        """
        return self.value_head(self.feature(state))

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
        return self.μ.dist(s)

    def eligibility(self, s):
        r""" Specifies eligibility trace-decay rate

        .. math::
            \lambda: \mathcal{S} \mapsto \mathbb{R}
        """
        return self.λ(s)

    def delta(self, exp: Transitions):
        """ Specifies the loss function, i.e. TD error """
        raise NotImplementedError

    def learn(self, transitions: Transitions):
        """ Updates parameters of the network via auto-diff """
        loss = self.delta(transitions)

        # TODO: pass this to the monitoring system on the first pass
        # make_dot(loss, params=dict(self.named_parameters()))

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.optimizer.step()

    def __str__(self):
        Γ = textwrap.indent(str(self.gvf), "\t")
        φ = textwrap.indent(str(self.φ), "\t\t")
        μ = textwrap.indent(str(self.μ), "\t\t")
        λ = textwrap.indent(str(self.λ), "\t\t")
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
                 *args, **kwargs):
        from gym.spaces import Discrete
        if not isinstance(behavior_policy.action_space, Discrete):
            raise NotImplementedError
        super().__init__(output_dim=behavior_policy.action_space.n,
                         behavior_policy=behavior_policy, *args, **kwargs)

    def behavior_policy(self, state):
        # Control policies usually require access to value functions.
        return self.μ.dist(state, vf=self)
