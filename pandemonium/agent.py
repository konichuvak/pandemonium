import time

from pandemonium.experience import Transition, Trajectory
from pandemonium.horde import Horde


class Agent:

    def __init__(self,
                 feature_extractor,
                 horde: Horde
                 ):

        # Feature extractor is shared across all demons
        self.feature_extractor = feature_extractor

        # How to use prediction of some demons as input for the other ones?
        self.horde = horde

    def interact(self, BATCH_SIZE, env):
        done = False
        env.seed(1337)  # keep the env consistent from episode to episode
        s0 = env.reset()
        x0 = self.horde.demons[0].feature(s0)

        steps = updates = 0
        wall_time = time.time()
        while not done:

            # Interact
            transitions = list()
            while len(transitions) < BATCH_SIZE and not done:
                a, policy_info = self.horde.demons[0].behavior_policy(x0)
                s1, reward, done, info = env.step(a)
                x1 = self.horde.demons[0].feature(s1)
                info.update(**policy_info)
                t = Transition(s0, a, reward, s1, done, x0, x1, info=info)
                transitions.append(t)
                s0, x0 = s1, x1

            # Learn from experience
            logs = self.horde.learn(transitions)

            # Record statistics
            steps += len(transitions)
            updates += 1
            logs.update({
                'done': False,
                'episode_time': round(time.time() - wall_time),
                'episode_steps': steps,
                'episode_updates': updates,
                'trajectory': Trajectory.from_transitions(transitions),
            })
            yield logs

        yield {
            'done': True,
            'episode_time': round(time.time() - wall_time),
            'episode_steps': steps,
            'episode_updates': updates,
        }
