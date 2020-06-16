from pandemonium.demons.control import SARSA, OfflineTDControl, OnlineTDControl
from pandemonium.demons.offline_td import TTD
from pandemonium.utilities.utilities import get_all_classes


class MultistepSARSA(SARSA, OfflineTDControl, TTD):
    ...


class OnlineSARSA(SARSA, OnlineTDControl):
    ...


__all__ = get_all_classes(__name__)
