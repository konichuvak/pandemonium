from typing import List

import torch
from pandemonium import GVF
from pandemonium.experience import Transitions
from pandemonium.networks.heads import LinearNet
from pandemonium.policies import Policy
from pandemonium.traces import EligibilityTrace


class Demon(LinearNet):
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
                 device: torch.device = None,
                 *args, **kwargs):
        super().__init__(body=feature, *args, **kwargs)

        self.gvf = gvf

        self.φ = feature
        self.μ = behavior_policy
        self.λ = eligibility

        # Jointly optimizes feature generation net and prediction net
        parameters = set(self.parameters())
        if isinstance(feature, torch.nn.Module):
            parameters |= set(self.body.parameters())
        self.optimizer = torch.optim.Adam(parameters, 0.001)

        self.to(device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        r""" Approximate value of a state is linear wrt features

        .. math::
            \widetilde{V}(s) = \boldsymbol{\phi}(s)^{T}\boldsymbol{w}

        """
        return self.forward(x)

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

    def delta(self, exp: Transitions):
        """ Specifies the loss function, i.e. TD error """
        raise NotImplementedError

    def learn(self, exp: Transitions):
        """ Updates parameters of the network via auto-diff """
        loss = self.delta(exp)
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
        # Control policies require access to value functions.
        return self.μ(state, vf=self)


class Horde:
    r""" A horde of demons

    Container class for all the demons of a particular agents.
    """

    def __init__(self,
                 control_demon: ControlDemon,
                 prediction_demons: List[PredictionDemon]):
        self.control_demon = control_demon
        self.prediction_demons = prediction_demons
        self.demons = prediction_demons + [control_demon]
