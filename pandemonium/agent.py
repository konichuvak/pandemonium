import time

from gym_minigrid.minigrid import MiniGridEnv
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

    def interact(self,
                 BATCH_SIZE: 32,
                 env: MiniGridEnv,
                 learn: bool = True,
                 render: bool = False):
        """
        .. todo::
            Add ability to interrupt and resume interactions for on-demand eval
        """
        render_batch = False

        done = False
        env.seed(1337)  # keep the env consistent from episode to episode
        s0 = env.reset()

        steps = updates = 0
        wall_time = time.time()
        while not done:

            if render and env.unwrapped.step_count % 100 == 0:
                render_batch = True

            transitions = list()
            while len(transitions) < BATCH_SIZE and not done:

                if render_batch:
                    env.render()
                    time.sleep(0.1)

                x = self.horde.control_demon.feature(s0).detach()
                dist = self.horde.control_demon.behavior_policy(x)
                a = dist.sample()
                s1, reward, done, info = env.step(a)

                t = Transition(s0, a, reward, s1, done, info=info)
                transitions.append(t)

                s0 = s1
                steps += 1

            logs = dict()
            if learn:
                for demon in self.horde.demons:
                    log = demon.learn(transitions)
                    logs[id(demon)] = log
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

    def report(self):
        pass

    def eval(self):
        pass
