from pandemonium.demons.ac import TDAC
from pandemonium.demons.demon import (LinearDemon)
from pandemonium.demons.offline_td import TDn
from pandemonium.demons.prediction import OfflineTDPrediction


class AC(TDAC, OfflineTDPrediction, LinearDemon, TDn):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, output_dim=1)
