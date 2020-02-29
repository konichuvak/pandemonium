import textwrap
from typing import List

from pandemonium.demons import ControlDemon, PredictionDemon


class Horde:
    r""" A horde of Demons

    Container class for all the demons of a particular agent.
    """

    def __init__(self,
                 control_demon: ControlDemon,
                 prediction_demons: List[PredictionDemon]):
        self.control_demon = control_demon
        self.prediction_demons = prediction_demons
        self.demons = prediction_demons + [control_demon]

    def __str__(self):
        pred = [textwrap.indent(str(), "\t") for _ in self.prediction_demons]
        pred = '\n'.join(pred)
        control = textwrap.indent(str(self.control_demon), "\t")
        return f'Horde(\n' \
               f'\tControl:\n{control}\n' \
               f'\tPrediction:\n{pred}\n' \
               f')'


__all__ = ['Horde']
