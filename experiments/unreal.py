from collections import deque
from functools import reduce

import pygame
import torch
from ray.rllib.utils.schedules import ConstantSchedule

from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness, PixelChange
from pandemonium.demons.control import AC, PixelControl
from pandemonium.demons.prediction import ValueReplay, RewardPrediction
from pandemonium.envs import DeepmindLabEnv
from pandemonium.envs.wrappers import Torch
from pandemonium.experience import Transitions, Transition, Trajectory
from pandemonium.networks.bodies import ConvBody
from pandemonium.policies.discrete import Egreedy
from pandemonium.policies.gradient import VPG
from pandemonium.utilities.replay import Replay

__all__ = ['AGENT', 'ENV', 'WRAPPERS', 'BATCH_SIZE', 'device', 'viz']

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# ------------------------------------------------------------------------------
# Specify learning environment
# ------------------------------------------------------------------------------

envs = [
    DeepmindLabEnv('seekavoid_arena_01'),
    # EmptyEnv(size=10),
    # FourRooms(),
    # DoorKeyEnv(size=7),
    # MultiRoomEnv(4, 4),
    # CrossingEnv(),
]
WRAPPERS = [
    # Non-observation wrappers
    # SimplifyActionSpace,

    # Observation wrappers
    # FullyObsWrapper,
    # RGBImgPartialObsWrapper,
    # ImgObsWrapper,
    # OneHotObsWrapper,
    # FlatObsWrapper,
    # lambda e: ImageNormalizer(e),
    lambda e: Torch(e, device=device),
]
ENV = reduce(lambda e, wrapper: wrapper(e), WRAPPERS, envs[0])
ENV.unwrapped.max_steps = float('inf')

# ------------------------------------------------------------------------------
# Specify questions of interest (main and auxiliary tasks)
# ------------------------------------------------------------------------------
π = Egreedy(epsilon=ConstantSchedule(0.), action_space=ENV.action_space)
optimal_control = GVF(
    target_policy=π,
    cumulant=Fitness(ENV),
    continuation=ConstantContinuation(0.9)
)

# Tracks the color intensity over patches of pixels in the image
pixel_control = GVF(
    target_policy=π,
    cumulant=PixelChange(),
    continuation=ConstantContinuation(0.9),
)

# Auxiliary task of maximizing un-discounted n-step return
reward_prediction = GVF(
    target_policy=π,
    cumulant=Fitness(ENV),
    continuation=ConstantContinuation(0.),
)

# The objective of the value replay is to speed up & stabilize A3C agent
value_replay = optimal_control

# ------------------------------------------------------------------------------
# Specify learners that will be answering the questions
# ------------------------------------------------------------------------------

# ==================================
# Representation learning
# ==================================
obs = ENV.reset()
print(obs.shape)
feature_extractor = ConvBody(
    *obs.shape[1:], feature_dim=2 ** 8,
    channels=[32, 64, 64], kernels=[8, 4, 3], strides=[4, 2, 1]
)

# ==================================
# Behavioral Policy
# ==================================

policy = VPG(feature_dim=feature_extractor.feature_dim,
             action_space=ENV.action_space)

# ==================================
# Learning Algorithm
# ==================================
BATCH_SIZE = 20

# TODO: Skew the replay for reward prediction task
replay = Replay(memory_size=2000, batch_size=BATCH_SIZE)

rp = RewardPrediction(gvf=reward_prediction,
                      feature=feature_extractor,
                      behavior_policy=policy,
                      replay_buffer=replay)
vr = ValueReplay(gvf=value_replay,
                 feature=feature_extractor,
                 behavior_policy=policy,
                 replay_buffer=replay)
pc = PixelControl(gvf=pixel_control,
                  feature=feature_extractor,
                  behavior_policy=policy,
                  replay_buffer=replay,
                  output_dim=ENV.action_space.n,
                  target_update_freq=100)


class UNREAL(AC):
    """ A version of AC that stores experience in the replay buffer """

    def __init__(self, replay_buffer: Replay, **kwargs):
        super().__init__(**kwargs)
        self.replay_buffer = replay_buffer

    def learn(self, transitions: Transitions) -> dict:
        info = super().learn(transitions)
        self.replay_buffer.feed_batch(transitions)
        return info


control_demon = UNREAL(gvf=optimal_control,
                       replay_buffer=replay,
                       actor=policy,
                       feature=feature_extractor)

demon_weights = torch.tensor([1, 1, 1, 0.5], dtype=torch.float, device=device)

# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------

horde = Horde(
    control_demon=control_demon,
    prediction_demons=[rp, vr, pc],
    aggregation_fn=lambda losses: demon_weights.dot(losses),
    device=device,
)
AGENT = Agent(feature_extractor=feature_extractor, horde=horde)
print(horde)

# ------------------------------------------------------------------------------
FPS = 60
display_env = Torch(
    env=DeepmindLabEnv(
        level='seekavoid_arena_01',
        width=600,
        height=600,
        fps=FPS,
        display_size=(900, 600)
    ),
    device=device
)


def viz():
    clock = pygame.time.Clock()

    display = display_env.display
    display_env.reset()
    s0 = ENV.reset()
    x0 = feature_extractor(s0)
    v = control_demon(x0)

    # Simple circular buffers for displaying value trace and reward predictions
    states = deque([s0], maxlen=rp.sequence_size)
    values = deque([v], maxlen=100)

    for _ in range(300):

        # Step in the actual environment with the AC demon
        a, policy_info = policy(x0)
        s1, reward, done, info = ENV.step(a)
        x1 = feature_extractor(s1)
        t = Transition(s0, a, reward, s1, done, x0, x1, info=info)
        traj = Trajectory.from_transitions([t])

        # Step in the hi-res env for display purposes
        display_env.step(a)

        # Reset canvas
        display.surface.fill((0, 0, 0))

        # Display agent's observation
        display.show_image(display_env.last_obs)

        # Display rolling value of states
        values.append(control_demon.predict(x1))
        display.show_value(torch.tensor(list(values)).cpu().detach().numpy())

        # Display reward prediction bar
        states.append(s0)
        if len(states) != states.maxlen:
            continue
        x = rp.feature(torch.cat(list(states)))
        v = rp.predict(x.view(1, -1)).softmax(1)
        v = v.squeeze().cpu().detach().numpy()
        display.show_reward_prediction(reward=reward, rp_c=v)

        # Display pixel change
        z = pc.gvf.cumulant(traj).squeeze()
        x = pc.feature(traj.s0)
        v = pc.predict(x)[[0], traj.a]
        v = v.squeeze().cpu().detach().numpy()
        z = z.squeeze().cpu().detach().numpy()
        display.show_pixel_change(z, display.obs_shape[0], 0, 3.0, "PC")
        display.show_pixel_change(v, display.obs_shape[0] + 100, 0, 0.4, "PC Q")

        # Update surface, states and tick
        pygame.display.update()
        s0, x0 = s1, x1
        clock.tick(FPS)
