from typing import NamedTuple

import deepmind_lab
import gym
import numpy as np
from gym.spaces import Box

from pandemonium.envs.dm_lab import LEVELS
from pandemonium.envs.dm_lab.display import DeepmindLabDisplay


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class DeepmindLabEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    class Actions(NamedTuple):
        LOOK_LEFT: np.ndarray = _action(-20, 0, 0, 0, 0, 0, 0)
        LOOK_RIGHT: np.ndarray = _action(20, 0, 0, 0, 0, 0, 0)
        # LOOK_UP: np.ndarray = _action(0, 10, 0, 0, 0, 0, 0)
        # LOOK_DOWN: np.ndarray = _action(0, -10, 0, 0, 0, 0, 0)
        STRAFE_LEFT: np.ndarray = _action(0, 0, -1, 0, 0, 0, 0)
        STRAFE_RIGHT: np.ndarray = _action(0, 0, 1, 0, 0, 0, 0)
        FORWARD: np.ndarray = _action(0, 0, 0, 1, 0, 0, 0)
        BACKWARD: np.ndarray = _action(0, 0, 0, -1, 0, 0, 0)
        # FIRE: np.ndarray = _action(0, 0, 0, 0, 1, 0, 0)
        # JUMP: np.ndarray = _action(0, 0, 0, 0, 0, 1, 0)
        # CROUCH: np.ndarray = _action(0, 0, 0, 0, 0, 0, 1)

    def __init__(self, level: str, colors: str = 'RGB_INTERLEAVED',
                 width: int = 84, height: int = 84, fps: int = 60,
                 display_size=(600, 400),
                 **kwargs):
        super().__init__(**kwargs)

        if level not in LEVELS:
            raise Exception(f'level {level} not supported')

        self._colors = colors
        self.lab = deepmind_lab.Lab(
            level,
            [self._colors],
            config={
                'fps': str(fps),
                'width': str(width),
                'height': str(height)
            })

        self.actions = DeepmindLabEnv.Actions()
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = Box(0, 255, (height, width, 3), dtype=np.uint8)

        self.last_obs = None

        self.display = DeepmindLabDisplay(display_size, env=self)

    def step(self, action: int, frame_skip: int = 4):
        reward = self.lab.step(self.actions[action], num_steps=frame_skip)
        done = not self.lab.is_running()
        obs = self.last_obs if done else self.lab.observations()[self._colors]
        self.last_obs = self._normalize_pixels(obs)
        return self.last_obs, reward, done, dict()

    def reset(self):
        self.lab.reset()
        obs = self.lab.observations()[self._colors]
        self.last_obs = self._normalize_pixels(obs)
        return self.last_obs

    def seed(self, seed=None):
        self.lab.reset(seed=seed)

    def close(self):
        self.lab.close()

    def render(self, mode='human', close=False, **kwargs):
        if mode is 'human':
            self.display.show_image(self.last_obs)
        elif mode is 'rgb_array':
            return self.last_obs
        else:
            raise TypeError(mode)

    @staticmethod
    def _normalize_pixels(image):
        image = image.astype(np.float32)
        image = image / 255.0
        return image
