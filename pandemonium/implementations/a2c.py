from torch import device
from torch.nn.functional import mse_loss

from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness
from pandemonium.demons.ac import TDAC
from pandemonium.demons.demon import LinearDemon
from pandemonium.demons.offline_td import TTD
from pandemonium.demons.prediction import OfflineTDPrediction
from pandemonium.policies.discrete import Greedy
from pandemonium.utilities.utilities import get_all_members


class AC(TDAC, OfflineTDPrediction, LinearDemon, TTD):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, output_dim=1)


def create_demons(config, env, feature_extractor, policy) -> Horde:
    control_demon = AC(
        gvf=GVF(
            target_policy=Greedy(
                feature_dim=feature_extractor.feature_dim,
                action_space=env.action_space
            ),
            cumulant=Fitness(env),
            continuation=ConstantContinuation(config['gamma'])),
        behavior_policy=policy,
        feature=feature_extractor,
        criterion=mse_loss,
        trace_decay=config['trace_decay']
    )
    # TODO: pass the device with the demon
    return Horde([control_demon], device('cpu'))


__all__ = get_all_members(__name__)
