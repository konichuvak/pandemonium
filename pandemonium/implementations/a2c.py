from pandemonium.demons.ac import TDAC
from pandemonium.demons.demon import LinearDemon
from pandemonium.demons.offline_td import TDn
from pandemonium.demons.prediction import OfflineTDPrediction
from pandemonium.utilities.utilities import get_all_classes


class AC(TDAC, OfflineTDPrediction, LinearDemon, TDn):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, output_dim=1)


__all__ = get_all_classes(__name__)
