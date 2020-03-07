import textwrap
from typing import List, Callable

import torch
from torchviz import make_dot

from pandemonium.demons import ControlDemon, PredictionDemon


class Horde:
    r""" A horde of Demons

    Container class for all the demons of a particular agent.
    """

    def __init__(self,
                 control_demon: ControlDemon,
                 prediction_demons: List[PredictionDemon],
                 aggregation_fn: Callable[[torch.Tensor], torch.Tensor],
                 ):
        self.control_demon = control_demon
        self.prediction_demons = prediction_demons
        self.demons = [control_demon] + prediction_demons
        self.aggregation_fn = aggregation_fn

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

        losses = torch.empty(len(self.demons))
        logs = dict()

        for i, demon in enumerate(self.demons):
            loss, info = demon.learn(transitions)
            losses[i] = loss if loss is not None else 0
            logs.update(
                {f'{demon}{id(demon)}': {f'{k}': v for k, v in info.items()}})

        total_loss = self.aggregation_fn(losses)
        if total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(set(self.params.values()), 1)
            self.optimizer.step()
        else:
            print('???')

        # TODO: deal with changes in computational graph over time
        #   i.e. the graph will look different before and after experience
        #   collection stage in DQN
        if self.first_pass:
            graph = make_dot(total_loss, params=self.params)
            self.first_pass = False
            logs.update({'graph': graph})

        logs.update({'total_loss': total_loss.item()})
        return logs

    def __repr__(self):
        pred = [textwrap.indent(repr(_), "\t") for _ in self.prediction_demons]
        pred = '\n'.join(pred)
        control = textwrap.indent(repr(self.control_demon), "\t")
        return f'Horde(\n' \
               f'\tControl:\n{control}\n' \
               f'\tPrediction:\n{pred}\n' \
               f')'


__all__ = ['Horde']
