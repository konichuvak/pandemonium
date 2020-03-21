import torch
from gym.core import ObservationWrapper


class Torch(ObservationWrapper):

    def __init__(self, *args, device: torch.device = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def observation(self, obs):
        t = torch.tensor(obs, device=self.device, dtype=torch.float32)
        if len(t.shape) == 3:
            t = t.permute(2, 0, 1)  # swap (w, h, dir) -> (dir, w, h)
        t = t.unsqueeze(0)  # add batch dim: (dir, w, h) -> (1, dir, w, h)
        return t


class Scaler(Torch):
    def __init__(self, coef: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coef = coef

    def observation(self, obs):
        return self.coef * obs


class ImageNormalizer(Scaler):
    def __init__(self, env):
        super().__init__(coef=1 / 255, env=env)
