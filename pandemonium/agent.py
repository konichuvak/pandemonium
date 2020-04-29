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

        Stops early if the episode is over.

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
        total_reward = 0.

        x0 = self.feature_extractor(s0)
        while len(transitions) < steps and not done:
            a, policy_info = self.behavior_policy(x0)
            s1, reward, done, info = env.step(a)
            x1 = self.feature_extractor(s1)
            info.update(**policy_info)
            t = Transition(s0, a, reward, s1, done, x0, x1, info=info)
            transitions.append(t)
            s0, x0 = s1, x1
            total_reward += reward

        # Record statistics
        logs = {
            'done': done,
            'interaction_reward': total_reward,
            'interaction_time': time.time() - wall_time,
            'interaction_steps': steps if done else len(transitions),
        }
        return transitions, logs

    def learn(self, env, episodes: int, update_freq: int) -> Iterator[dict]:

        total_steps = 0
        for episode in range(episodes):

            # Initialize metrics
            done = False        # indicator for the end of the episode
            episode_steps = 0   # tracks total steps taken during the episode
            episode_reward = 0  # tracks reward received during the episode
            logs = dict()

            # Initialize environment
            env.seed(1337)  # keeps the env consistent from episode to episode
            s0 = env.reset()

            while not done:
                # Collect experience
                transitions, logs = self.interact(env, s0, update_freq)
                episode_reward += logs.pop('interaction_reward')
                done = logs.pop('done')  # do not set `done` as this is interpreted as the end of trial by Tune

                # Learn from experience
                learning_stats = self.horde.learn(transitions)
                logs.update(**learning_stats)

                # Intra-episode stats
                episode_steps += logs.pop('interaction_steps')
                logs['timesteps_total'] = total_steps + episode_steps

                if not done:
                    yield logs

            # Inter-episode stats
            total_steps += episode_steps
            logs.update({
                'episode_reward': episode_reward,
                'episodes_total': episode,
            })
            yield logs





