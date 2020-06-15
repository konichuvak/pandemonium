from torch import nn
from torch.distributions import Distribution, Categorical

from pandemonium.policies import Policy
from pandemonium.utilities.utilities import get_all_classes


class DiffPolicy(Policy, nn.Module):
    """ Base class for parametrized policies """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)
        self.forward = self.act

    def dist(self, *args, **kwargs) -> Distribution:
        raise NotImplementedError

    def delta(self, *args, **kwargs):
        raise NotImplementedError


@Policy.register('VPG')
class VPG(DiffPolicy):
    """ Vanilla Policy Gradient """

    def __init__(self, entropy_coefficient: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_head = nn.Linear(self.feature_dim, self.action_space.n)
        self.β = entropy_coefficient

    def dist(self, features, *args, **kwargs) -> Categorical:
        logits = self.policy_head(features)
        return Categorical(logits=logits)

    def delta(self, features, actions, weights):
        dist = self.dist(features)
        policy_loss = -(dist.log_prob(actions) * weights).mean(0)
        entropy_loss = dist.entropy().mean(0)
        loss = policy_loss - self.β * entropy_loss
        return loss, {'policy_grad': policy_loss.item(),
                      'entropy': entropy_loss.item(),
                      'policy_loss': loss.item()}

    def __repr__(self):
        model = super().__repr__()[:-2]
        return f'{model}\n' \
               f'  (β): {self.β}\n' \
               f')'


__all__ = get_all_classes(__name__)
