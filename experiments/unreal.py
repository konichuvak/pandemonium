from functools import reduce

import torch
from gym_minigrid.envs import EmptyEnv, MultiRoomEnv, DoorKeyEnv
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from pandemonium import Agent, GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import Fitness, PixelChange
from pandemonium.demons.control import AC, PixelControl
from pandemonium.demons.prediction import RewardPrediction, ValueReplay
from pandemonium.envs import FourRooms, DeepmindLabEnv
from pandemonium.envs.wrappers import Torch, ImageNormalizer
from pandemonium.experience import Transitions
from pandemonium.networks.bodies import ConvBody, ConvLSTM
from pandemonium.policies.discrete import Egreedy
from pandemonium.policies.gradient import VPG
from pandemonium.utilities.replay import Replay
from ray.rllib.utils.schedules import ConstantSchedule

__all__ = ['AGENT', 'ENV', 'WRAPPERS', 'BATCH_SIZE']

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
    continuation=ConstantContinuation(1.),
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
# feature_extractor = ConvBody(d=3, w=7, h=7, feature_dim=2 ** 8)
feature_extractor = ConvLSTM(*obs.shape[1:], feature_dim=2 ** 8)

# ==================================
# Behavioral Policy
# ==================================

policy = VPG(feature_dim=feature_extractor.feature_dim,
             action_space=ENV.action_space)

# ==================================
# Learning Algorithm
# ==================================
BATCH_SIZE = 32

# TODO: Skew the replay for reward prediction task
replay = Replay(memory_size=2000, batch_size=BATCH_SIZE)

prediction_demons = [
    # RewardPrediction(gvf=reward_prediction,
    #                  feature=feature_extractor,
    #                  behavior_policy=policy,
    #                  replay_buffer=replay),
    ValueReplay(gvf=value_replay,
                feature=feature_extractor,
                behavior_policy=policy,
                replay_buffer=replay),
    PixelControl(gvf=pixel_control,
                 feature=feature_extractor,
                 behavior_policy=policy,
                 replay_buffer=replay,
                 output_dim=ENV.action_space.n),
]
for d in prediction_demons:
    print(d)


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

demon_weights = torch.tensor([1, 1, 0.01], dtype=torch.float, device=device)

# ------------------------------------------------------------------------------
# Specify agent that will be interacting with the environment
# ------------------------------------------------------------------------------

horde = Horde(
    control_demon=control_demon,
    prediction_demons=prediction_demons,
    aggregation_fn=lambda losses: demon_weights.dot(losses)
)
AGENT = Agent(feature_extractor=feature_extractor, horde=horde)
print(horde)
