from typing import Type

import torch
import torch.nn.functional as F

from pandemonium.demons import PredictionDemon, Demon, Loss, ControlDemon
from pandemonium.experience import Transition
from pandemonium.traces import EligibilityTrace, AccumulatingTrace
from pandemonium.utilities.utilities import get_all_classes


class OnlineTD(Demon):
    r""" Base class for backward-view (online) :math:`\TD` methods. """

    def __init__(self,
                 trace_decay: float,
                 eligibility: Type[EligibilityTrace] = AccumulatingTrace,
                 criterion: callable = F.smooth_l1_loss,
                 **kwargs):
        super().__init__(eligibility=None, **kwargs)
        self.criterion = criterion

        # TODO: fails for distributional learning and non-FA
        if isinstance(self, PredictionDemon):
            trace_dim = next(self.avf.parameters()).shape
        elif isinstance(self, ControlDemon):
            trace_dim = next(self.aqf.parameters()).shape
        else:
            raise TypeError(self)
        self.λ = eligibility(trace_decay, trace_dim)

    def delta(self, t: Transition) -> Loss:
        """ Specifies the update rule for approximate value function (avf)

        Since the algorithms in this family are online, the update rule is
        applied on every `Transition`.
        """
        raise NotImplementedError

    def target(self, t: Transition, v: torch.Tensor):
        """ Computes one-step update target. """
        raise NotImplementedError

    def learn(self, t: Transition):
        assert len(t) == 1

        # # Off policy importance sampling correction
        # π = self.gvf.π.dist(t.x0, self.aqf).probs[0][t.a]
        # b = self.μ.dist(t.x0, self.aqf).probs[0][t.a]
        # ρ = π / b

        return self.delta(t[0])


__all__ = get_all_classes(__name__)
