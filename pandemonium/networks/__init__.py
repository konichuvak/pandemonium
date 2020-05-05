from copy import deepcopy

from pandemonium.networks.bodies import *
from pandemonium.networks.heads import *
from pandemonium.networks.utils import *

""" The organization of the networks module was inspired by 
Shangtong Zhang's library `DeeRL`.

https://github.com/ShangtongZhang/DeepRL/tree/master/deep_rl/network
"""


class TargetNetMixin:
    r""" Mixin that adds a target network to the agent.

    Duplicates the estimator networks that are used to estimate targets.
    These clones are updated at a `target_update_freq` frequency which allows
    to stabilize the training process and make targets less non-stationary.

    References
    ----------
    [1] "Playing Atari with Deep Reinforcement Learning"
        https://arxiv.org/pdf/1312.5602.pdf
    """
    aqf: callable
    avf: callable
    φ: callable

    def __init__(self, target_update_freq: int = 0):

        self.update_counter = 0  # keeps track of the number of updates so far

        # Create target networks to stabilize training
        self.target_networks = dict()
        self.target_update_freq = target_update_freq
        if self.target_update_freq:
            self.target_feature = deepcopy(self.φ)
            self.target_feature.load_state_dict(self.φ.state_dict())
            self.target_feature.eval()
            self.target_networks[self.target_feature] = self.φ

            if isinstance(self.avf, nn.Module):
                # `avf` is a network in duelling DQN and is implicit in `aqf`
                # in other cases
                self.target_avf = deepcopy(self.avf)
                self.target_avf.load_state_dict(self.avf.state_dict())
                self.target_avf.eval()
                self.target_networks[self.target_avf] = self.avf

            if isinstance(self.aqf, nn.Module):
                # `aqf` is a network in all DQN family except distributional
                # case, where it is implicit in the `azf`
                self.target_aqf = deepcopy(self.aqf)
                self.target_aqf.load_state_dict(self.aqf.state_dict())
                self.target_aqf.eval()
                self.target_networks[self.target_aqf] = self.aqf

            if hasattr(self, 'azf') and isinstance(self.azf, nn.Module):
                # only for distributional agents
                self.target_azf = deepcopy(self.azf)
                self.target_azf.load_state_dict(self.azf.state_dict())
                self.target_azf.eval()
                self.target_networks[self.target_azf] = self.azf

    def sync_target(self):
        if self.target_update_freq:
            if self.update_counter % self.target_update_freq == 0:
                for target_net, net in self.target_networks.items():
                    target_net.load_state_dict(net.state_dict())
