from ray.rllib.utils.schedules import ConstantSchedule as RayConstantSchedule
from ray.rllib.utils.schedules import LinearSchedule as RayLinearSchedule
from ray.rllib.utils.schedules import \
    PolynomialSchedule as RayPolynomialSchedule
from ray.rllib.utils.schedules import Schedule

__all__ = ['Schedule', 'ConstantSchedule', 'LinearSchedule',
           'PolynomialSchedule']


class PolynomialSchedule(RayPolynomialSchedule):

    def __init__(self, *args, framework='torch', **kwargs):
        super().__init__(*args, **kwargs, framework=framework)

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'horizon={self.schedule_timesteps}, ' \
               f'{self.initial_p} -> {self.final_p}, ' \
               f'exponent={self.power})'


class LinearSchedule(RayLinearSchedule):

    def __init__(self, *args, framework='torch', **kwargs):
        super().__init__(*args, **kwargs, framework=framework)

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'horizon={self.schedule_timesteps}, ' \
               f'{self.initial_p} -> {self.final_p})'


class ConstantSchedule(RayConstantSchedule):

    def __init__(self, *args, framework='torch', **kwargs):
        super().__init__(*args, **kwargs, framework=framework)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._v})'
