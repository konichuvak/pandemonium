import time
from typing import Tuple, Iterator

import torch
from pandemonium.experience import Transition, Transitions
from pandemonium.horde import Horde
from pandemonium.policies import Policy


class Agent:

    def __init__(self,
                 feature_extractor,
                 behavior_policy: Policy,
                 horde: Horde
                 ):

        # Representation shared across al demons
        self.feature_extractor = feature_extractor

        # Behaviour policy is unique for all demons
        self.behavior_policy = behavior_policy

        # How to use prediction of some demons as input for the other ones?
        self.horde = horde

    def interact(self,
                 env,
                 s0: torch.Tensor,
                 steps: int) -> Tuple[Transitions, dict]:
        """ Perform number of `steps` in the environment


        Parameters
        ----------
        env
            environment in which to take actions
        s0
            starting state of the environment
        steps
            number of steps to interact for

        Returns
        -------
        A list of transitions with a dictionary of logs

        # TODO: add parallel experience collection
        """
        wall_time = time.time()
        transitions = list()
        done = False
        x0 = self.feature_extractor(s0)
        while len(transitions) < steps and not done:
            a, policy_info = self.behavior_policy(x0)
            s1, reward, done, info = env.step(a)
            x1 = self.feature_extractor(s1)
            info.update(**policy_info)
            t = Transition(s0, a, reward, s1, done, x0, x1, info=info)
            transitions.append(t)
            s0, x0 = s1, x1

        # Record statistics
        logs = {
            'episode_end': done,
            'time': time.time() - wall_time,
            'steps': steps if done else len(transitions),
        }
        return transitions, logs

    def learn(self, env, episodes: int, update_freq: int) -> Iterator[dict]:

        for episode in range(episodes):
            done = False  # indicator for the end of the episode
            env.seed(1337)  # keep the env consistent from episode to episode
            s0 = env.reset()
            while not done:
                transitions, logs = self.interact(env, s0, update_freq)
                learning_stats = self.horde.learn(transitions)
                logs.update(**learning_stats)
                yield logs
                done = logs['episode_end']
