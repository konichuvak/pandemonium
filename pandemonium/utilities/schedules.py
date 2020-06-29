from ray.rllib.utils.schedules import (Schedule, ConstantSchedule,
                                       LinearSchedule, PolynomialSchedule)

__all__ = ['Schedule', 'ConstantSchedule', 'LinearSchedule',
           'PolynomialSchedule']


def polynomial_repr(self):
    return f'{self.__class__.__name__}(' \
           f'horizon={self.schedule_timesteps}, ' \
           f'{self.initial_p} -> {self.final_p}, ' \
           f'exponent={self.power})'


def linear_repr(self):
    return f'{self.__class__.__name__}(' \
           f'horizon={self.schedule_timesteps}, ' \
           f'{self.initial_p} -> {self.final_p}'


def const_repr(self):
    return f'{self.__class__.__name__}({self._v})'


PolynomialSchedule.__repr__ = polynomial_repr
LinearSchedule.__repr__ = linear_repr
ConstantSchedule.__repr__ = const_repr
