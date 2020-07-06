import time
from typing import Tuple, Iterator

import torch

from pandemonium.experience import Transition, Transitions, Trajectory
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
                 steps: int) -> Tuple[Transitions, torch.Tensor, dict]:
        """ Perform number of `steps` in the environment

        Stops early if the episode is over.

        Parameters
        ----------
        env:
            environment in which to take actions
        s0:
            starting state of the environment
        steps:
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

        # Initial feature vector and action
        x0 = self.feature_extractor(s0)
        a0, policy_info = self.behavior_policy(x0)

        while len(transitions) < steps and not done:
            s1, reward, done, info = env.step(a0)
            info.update(**policy_info)
            x1 = self.feature_extractor(s1)
            a1, policy_info = self.behavior_policy(x1)
            # TODO: consider make the action selection conditional?
            #   or maybe simply re-select action in ExpectedSarsa case
            #   instead of using a1 taken here
            # Because  the  update  rule  of  Expected  Sarsa,  unlike
            # Sarsa,  does  not  make  use  of  the  action  taken  in s_t+1,
            # action selection can occur after the update.
            # Doing so can be advantageous  in  problems  containing  states
            # with  returning actions, i.e.P(st+1=st)>0.
            # When s_{t+1} = s_t, performing an update of Q(s_t, a_t), will also
            # update Q(s_{t+1}, a_t), yielding a better estimate before action
            # selection occurs.
            t = Transition(s0, a0, reward, s1, done, x0, x1, a1=a1, info=info)
            transitions.append(t)
            s0, x0, a0 = s1, x1, a1
            total_reward += reward

        # Record statistics
        trajectory = Trajectory.from_transitions(transitions)

        logs = {
            'done': done,
            'interaction_reward': total_reward,
            'interaction_time': time.time() - wall_time,
            'interaction_steps': steps if done else len(transitions),
            # 'actions': trajectory.a.tolist(),
            # 'entropy': trajectory.info['entropy'].tolist(),
            'is_ratios': trajectory.ρ.mean().item()
        }

        # OHLC and mean?
        if 'epsilon' in trajectory.info:
            logs['epsilon'] = trajectory.info['epsilon'][-1]
        if 'temperature' in trajectory.info:
            logs['temperature'] = trajectory.info['temperature'][-1]

        return transitions, s0, logs

    def learn(self, env, episodes: int, horizon: int) -> Iterator[dict]:

        total_steps = 0
        for episode in range(episodes):

            # Initialize metrics
            done = False  # indicator for the end of the episode
            episode_steps = 0  # tracks total steps taken during the episode
            episode_reward = 0  # tracks reward received during the episode
            logs = dict()

            # Initialize environment
            env.seed(1337)  # keeps the env consistent from episode to episode
            state = env.reset()

            while not done:
                # Collect experience
                transitions, state, logs = self.interact(env, state, horizon)
                episode_reward += logs.pop('interaction_reward')
                done = logs.pop('done')

                # Learn from experience
                learning_stats = self.horde.learn(transitions)
                logs.update(**learning_stats)

                # Intra-episode stats
                episode_steps += logs.pop('interaction_steps')
                logs['timesteps_total'] = total_steps + episode_steps

                # Used for HParams tab in TBX in minigrids to evaluate agents
                # based on how many episodes they could compltet in a fixed
                # amount of steps
                logs['episodes_total'] = episode

                if not done:
                    yield logs

            # Inter-episode stats
            total_steps += episode_steps
            logs.update({
                'steps_this_episode': episode_steps,
                'episode_reward': episode_reward,
            })
            yield logs


__all__ = ['Agent']
