from ray.rllib.utils.schedules import ConstantSchedule as ConstSchedule
from ray.rllib.utils.schedules import PolynomialSchedule as PolySchedule
from ray.rllib.utils.schedules import Schedule

__all__ = ['Schedule', 'ConstantSchedule', 'LinearSchedule',
           'PolynomialSchedule']


class ConstantSchedule(ConstSchedule):

    def __repr__(self):
        return f'{self.__class__.__name__}({self._v})'


class PolynomialSchedule(PolySchedule):

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'horizon={self.schedule_timesteps}, ' \
               f'{self.initial_p} -> {self.final_p}, ' \
               f'exponent={self.power})'


class LinearSchedule(PolynomialSchedule):

    def __init__(self, **kwargs):
        super().__init__(power=1.0, **kwargs)
