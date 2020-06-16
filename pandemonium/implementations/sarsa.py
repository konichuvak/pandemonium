from pandemonium.demons.control import SARSA, OfflineTDControl, OnlineTDControl
from pandemonium.demons.offline_td import TTD


class MultistepSARSA(SARSA, OfflineTDControl, TTD):
    ...


class OnlineSARSA(SARSA, OnlineTDControl):
    ...
