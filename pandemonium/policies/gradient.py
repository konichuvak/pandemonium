from torch import nn
from torch.distributions import Distribution, Categorical

from pandemonium.policies import Policy


class PolicyGradient(Policy, nn.Module):
    """ A class of parametrized policies """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)

    def delta(self, features, actions, weights):
        raise NotImplementedError

    def dist(self, *args, **kwargs) -> Distribution:
        raise NotImplementedError


class VPG(PolicyGradient):
    """ Vanilla Policy Gradient """

    def __init__(self, feature_dim: int, entropy_coefficient: float = 0.01,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_head = nn.Linear(feature_dim, self.action_space.n)
        self.β = entropy_coefficient

    def dist(self, features, *args, **kwargs) -> Categorical:
        logits = self.policy_head(features)
        return Categorical(logits=logits)

    def delta(self, features, actions, weights):
        dist = self.dist(features)
        policy_loss = -(dist.log_prob(actions) * weights).mean()
        entropy_loss = dist.entropy().mean()
        loss = policy_loss - self.β * entropy_loss
        return loss

    def __str__(self):
        model = super().__str__()[:-2]
        return f'{model}\n' \
               f'  (β): {self.β}\n' \
               f')'
