import textwrap
from typing import List, Callable

import torch
from pandemonium.demons import ControlDemon, PredictionDemon
from torch import nn
from torchviz import make_dot


class Horde(torch.nn.Module):
    r""" A horde of Demons

    Container class for all the demons of a particular agent.

    Auto-registers parameters of all the demons through ModuleDict container.

    .. note::
        calling state_dict() will return duplicate parameters, i.e. a convnet
        shared across demons can appear multiple times, which increases memory
        overhead.
        however, named_parameters() handles the duplicates somehow!

    """

    def __init__(self,
                 control_demon: ControlDemon,
                 prediction_demons: List[PredictionDemon],
                 aggregation_fn: Callable[[torch.Tensor], torch.Tensor],
                 device: torch.device,
                 ):
        super().__init__()
        demons = {str(demon): demon for demon in prediction_demons}
        demons.update({str(control_demon): control_demon})
        self.demons = nn.ModuleDict(demons)

        # Determines how the total loss is weighted across demons
        self.aggregation_fn = aggregation_fn

        # Set up the optimizer that will be shared across all demons
        self.optimizer = torch.optim.Adam(self.parameters(), 0.001)
        self.first_pass = True

        self.device = device
        self.to(device)

    def learn(self, transitions) -> dict:

        # TODO: thread / mp

        losses = torch.empty(len(self.demons), device=self.device)
        logs = dict()

        # Aggregate losses from each demon
        for i, (d, demon) in enumerate(self.demons.items()):
            loss, info = demon.learn(transitions)
            losses[i] = loss if loss is not None else 0
            logs.update(
                {f'{d}{id(demon)}': {f'{k}': v for k, v in info.items()}})

        # Optimize joint objective
        total_loss = self.aggregation_fn(losses)
        if not total_loss.requires_grad:
            # Sometimes (e.g. in experience collection stage) there is no
            # gradient step to be performed
            pass
        else:
            self.optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimizer.step()

        # Create a schematic of computational graph
        # graph = make_dot(total_loss, params=dict(self.named_parameters()))

        # logs.update({'total_loss': total_loss.item(), 'graph': graph})
        logs.update({'total_loss': total_loss.item()})
        return logs

    def __str__(self):
        s = [textwrap.indent(repr(d), "\t") for d in self.demons.keys()]
        s = '\n'.join(s)
        return f"Horde(\n{s}\n)"

    def __repr__(self):
        s = [textwrap.indent(repr(d), "\t") for d in self.demons.values()]
        s = '\n'.join(s)
        return f"Horde(\n{s}\n)"


__all__ = ['Horde']
