import textwrap
from typing import List

import torch
from torchviz import make_dot

from pandemonium.demons import ControlDemon, PredictionDemon


class Horde:
    r""" A horde of Demons

    Container class for all the demons of a particular agent.

    # TODO: weights of individual demon losses (add a loss equation from UNREAL)
    # TODO: lift the assumption of additive loss across demons and use a loss function instead
    """

    def __init__(self,
                 control_demon: ControlDemon,
                 prediction_demons: List[PredictionDemon]):
        self.control_demon = control_demon
        self.prediction_demons = prediction_demons
        self.demons = [control_demon] + prediction_demons

        # Set up the optimizer that will be shared across all demons
        self.params = dict(self.control_demon.named_parameters())
        for demon in self.prediction_demons:
            self.params.update(dict(demon.named_parameters(
                # prefix=demon.__class__.__name__
            )))
        self.optimizer = torch.optim.Adam(set(self.params.values()), 0.001)

        self.first_pass = True

    def learn(self, transitions) -> dict:

        # TODO: thread / mp

        logs = dict()
        total_loss = torch.tensor([0.])

        for demon in self.demons:
            loss, info = demon.learn(transitions)
            if loss is not None:
                total_loss += loss.cpu()
            logs.update({f'{id(demon)}_{k}': v for k, v in info.items()})

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.first_pass:
            graph = make_dot(total_loss, params=self.params)
            self.first_pass = False
            logs.update({'graph': graph})

        logs.update({'total_loss': total_loss.item()})
        return logs

    def __str__(self):
        pred = [textwrap.indent(str(), "\t") for _ in self.prediction_demons]
        pred = '\n'.join(pred)
        control = textwrap.indent(str(self.control_demon), "\t")
        return f'Horde(\n' \
               f'\tControl:\n{control}\n' \
               f'\tPrediction:\n{pred}\n' \
               f')'


__all__ = ['Horde']
