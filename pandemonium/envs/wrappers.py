from enum import IntEnum

import gym
import numpy as np
import torch
from gym import spaces
from gym_minigrid.wrappers import FlatObsWrapper


class RandomRewards(gym.core.Wrapper):
    """ In this experiment, the rewards were selected according to
    normal probability distribution with a standard deviation of 0.1
    and a mean that was different for each state-action pair.
    The means were selected randomly at the beginning of each run uniformly
    from the [−1; 0] interval.
    """

    def __init__(self, env):
        super().__init__(env)
        self.rewards = None
        self.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        ix = (action, env.agent_dir, *reversed(env.agent_pos))
        reward = self.rewards[ix]

        return obs, reward, done, info

    def reset(self, **kwargs):
        env = self.unwrapped
        dim = (len(env.actions), 4, env.height, env.width)
        self.rewards = np.random.normal(
            loc=np.random.uniform(-1, 0, dim),
            scale=0.1 + np.zeros(dim)
        )
        return self.env.reset(**kwargs)


class AgentPosition(gym.core.ObservationWrapper):

    def observation(self, observation):
        env = self.unwrapped
        return (env.agent_dir, *reversed(env.agent_pos))


class OneHotObsWrapper(gym.core.ObservationWrapper):
    """ A binary vector with a single 1 corresponding to agent's position """

    def __init__(self, env):
        super().__init__(env)
        env = self.env.unwrapped
        self.dim = 4 * env.width * env.height
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.dim,),
            dtype='uint8'
        )

    def observation(self, obs) -> np.array:
        env = self.unwrapped
        y, x = env.agent_pos
        pos_index = env.agent_dir * env.width * env.height + x * env.width + y
        one_hot_obs = np.zeros(self.dim, dtype='uint8')
        one_hot_obs[pos_index] = 1
        return one_hot_obs


class Tabular(OneHotObsWrapper):
    """ Returns an index corresponding to the agent's position """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Discrete(self.dim)

    def observation(self, obs) -> np.float:
        obs = super().observation(obs)
        return np.where(obs == 1)[0][0]


class SimplifyObsSpace(gym.core.ObservationWrapper):
    """ The third dimension of the observation is a boolean for whether
    the Door object is open or locked, which can be completely ignored in some
    environments """

    def __init__(self, env, dim_to_keep: int = 0):
        """
        :param dim_to_keep: 0 for object type, 1 for color, 2 for state
        """

        super().__init__(env)
        self.dim_to_keep = dim_to_keep
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(self.env.width, self.env.height),
            dtype='uint8'
        )

    def observation(self, obs):
        if isinstance(obs, dict):
            obs['image'] = obs['image'][:, :, self.dim_to_keep]
        elif isinstance(obs, np.ndarray):
            obs = obs[:, :, self.dim_to_keep]
        else:
            raise ValueError(f'Unknown observation type {type(obs)}')
        return obs


class SimplifyActionSpace(gym.core.Wrapper):
    """ Simple mini-grids only require navigation actions

    We can thus ignore actions of opening the door or picking up objects.
    """

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, env):
        super().__init__(env)

        # Action enumeration for this environment
        self.actions = SimplifyActionSpace.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))


class FlatObsWrapperNoMission(FlatObsWrapper):
    """ One hot-encoding for image """

    def __init__(self, env):
        super().__init__(env)

        imgSpace = env.observation_space.spaces['image']
        imgSize = np.prod(imgSpace)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize),
            dtype='uint8'
        )

    def observation(self, obs):
        image = obs['image']
        obs = image.flatten()
        return obs


class Torch(gym.core.ObservationWrapper):

    def __init__(self, *args, device: torch.device = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def observation(self, obs):
        t = torch.tensor(obs, device=self.device, dtype=torch.float32)
        if len(t.shape) == 3:
            t = t.transpose(0, 2)  # swap (w, h, dir) -> (dir, w, h)
            t = t.unsqueeze(0)  # add batch dim: (dir, w, h) -> (1, dir, w, h)
        return t
