from gym import Env
from torch import nn

from pandemonium.experience import Transition, Trajectory
from pandemonium.utilities.utilities import get_all_classes


class Cumulant:
    r""" A signal of interest :math:`z` accumulated over time

    The classical example of a cumulant is a reward from MDP environment.
    Cumulant is not always maximized; most of the time we are interested
    in predicting the signal. Cumulants can also be vector valued,
    i.e. when we want to learn to predict features of the environment.

    Some interesting cumulants are the once that are tracking a metric in
    the agent itself. In this way we can express intrinsic motivation as a
    cumulant. For example, we might want to track confidence of the agent
    in its own prediction by using rolling average TD error.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Fitness(Cumulant):
    """ Tracks scalar extrinsic reward received from the MDP environment """

    def __init__(self, env: Env):
        self.env = env

    def __call__(self, experience: Transition):
        return experience.r

    def __str__(self):
        return f'Fitness({self.env.unwrapped.__class__.__name__})'


class PixelChange(Cumulant):
    """ A 2-dimensional cumulant tracking pixel change from frame to frame

    Maximization of this cumulant is used as an auxiliary task in the
    UNREAL architecture to help representational learning.
    """

    def __init__(self):
        # self.pooler = nn.AvgPool2d(kernel_size=4, stride=0)
        self.pooler = nn.AvgPool2d(kernel_size=1, stride=0)

    def __call__(self, traj: Trajectory):
        # crop = slice(2, -2)
        crop = slice(None)
        Δ = (traj.s0[..., crop, crop] - traj.s1[..., crop, crop]).abs()
        Δ = Δ.mean(1)  # average over channels
        z = self.pooler(Δ)  # average over pixels within patches
        return z


class FeatureCumulant(Cumulant):
    r""" Tracks a single value from the feature vector :math:`\boldsymbol{x}`

    In case when the features are linearly separable (linea FA) this cumulant
    can track the value of a particular feature in the feature vector over time.

    For example, we can assign a demon to learn to predict the amount of light
    that it will accumulate, if we know that our feature vector index `idx`
    corresponds to the reading of the light sensor.
    """

    def __init__(self, idx: int):
        self.idx = idx

    def __call__(self, experience: Transition):
        return experience.x0[self.idx]


class Surprise(Cumulant):
    r""" Tracks the 'surprise' associated with a new state using a density model

    See Berseth et al. 2019 for details @ https://arxiv.org/pdf/1912.05510.pdf
    """
    pass


class Curiosity(Cumulant):
    r""" Tracks the novelty of the states wrt to the forward model

    See Pathak et al. 2017 for details.
    """
    pass


class Empowerment(Cumulant):
    r""" Measures the amount of causal influence an agent has on the world

    Computed as a $log(S)$, i.e. logarithm of number of reachable (in fixed
    number of steps) states after performing action A.

    See Salge et al. 2014 for details.
    """
    pass


__all__ = get_all_classes(__name__)
