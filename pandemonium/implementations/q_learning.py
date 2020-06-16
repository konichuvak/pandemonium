from pandemonium.demons.control import (QLearning, OfflineTDControl,
                                        OnlineTDControl)
from pandemonium.demons.offline_td import TDn


class MultistepQLearning(QLearning, OfflineTDControl, TDn):
    ...


class OnlineQLearning(QLearning, OnlineTDControl):
    ...
