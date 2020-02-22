from typing import Type

from pandemonium.demons import Demon
from pandemonium.experience import Transition
from pandemonium.traces import EligibilityTrace, AccumulatingTrace


class TemporalDifference(Demon):
    """ Base class for Temporal Difference methods """

    def __init__(self,
                 feature,
                 trace_decay: float = 0,
                 eligibility: Type[EligibilityTrace] = AccumulatingTrace,
                 *args, **kwargs):
        e = eligibility(trace_decay, feature.feature_dim)
        super().__init__(eligibility=e, feature=feature, *args, **kwargs)

    def delta(self, t: Transition):
        raise NotImplementedError
