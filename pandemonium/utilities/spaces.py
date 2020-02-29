from typing import List

import torch
from gym.spaces import Discrete

from pandemonium.continuations import SigmoidContinuation
from pandemonium.option import Option
from pandemonium.policies import VPG


class OptionSpace(Discrete, dict):
    """ Container class for various Options """

    def __init__(self, options: List[Option]):
        super().__init__(len(options))
        self.options = dict(enumerate(options))
        # self.option_by_id = {id(option) for option in options}
        # self._options = options

    def __len__(self):
        return len(self.options)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.options[item]
        elif isinstance(item, list):
            return [self.options[o] for o in item]
        elif torch.is_tensor(item):
            return [self.options[o] for o in list(item.cpu().numpy())]
        else:
            raise TypeError(item)

    def __setitem__(self, key, value):
        self.options[key] = value


def create_option_space(n: int, action_space, feature_dim):
    options = [Option(
        policy=VPG(feature_dim, action_space=action_space),
        continuation=SigmoidContinuation(feature_dim),
        initiation=lambda x: 1,
    ) for _ in range(n)]
    return OptionSpace(options)

# if __name__ == '__main__':
#     from gym.envs.classic_control.cartpole import CartPoleEnv
#
#     from pandemonium.policies.gradient import VPG
#     from pandemonium.networks.bodies import Identity
#
#     env = CartPoleEnv()
#     obs = env.reset()
#     feature_extractor = Identity(state_dim=obs.shape[0])
#     options = [Option(
#         policy=VPG(
#             feature_extractor.feature_dim,
#             action_space=env.action_space
#         ),
#         initiation=lambda x: 1,
#         continuation=lambda x: 1,
#     )]
#     option_space = OptionSpace(options=options)
#     print(option_space)
