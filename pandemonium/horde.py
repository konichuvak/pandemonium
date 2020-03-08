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
        self.demons = nn.ModuleList([control_demon] + prediction_demons)

        # Determines how the total loss is weighted across demons
        self.aggregation_fn = aggregation_fn

        # Set up the optimizer that will be shared across all demons
        self.optimizer = torch.optim.Adam(self.parameters(), 0.001)
        self.first_pass = True

        self.to(device)

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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimizer.step()
        else:
            print('???')

        # TODO: deal with changes in computational graph over time
        #   i.e. the graph will look different before and after experience
        #   collection stage in DQN
        # if self.first_pass:
        graph = make_dot(total_loss, params=dict(self.named_parameters()))
        # self.first_pass = False
        logs.update({'graph': graph})

        logs.update({'total_loss': total_loss.item()})
        return logs

    def __repr__(self):
        s = [textwrap.indent(repr(d), "\t") for d in self.demons]
        s = '\n'.join(s)
        return f"Horde(\n{s}\n)"


__all__ = ['Horde']
