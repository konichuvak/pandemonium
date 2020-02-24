from gym_minigrid.minigrid import MiniGridEnv
from pandemonium.demons import Horde
from pandemonium.experience import Transition


class Agent:

    def __init__(self,
                 feature_extractor,
                 horde: Horde):

        # Feature extractor is shared across all demons
        self.feature_extractor = feature_extractor

        # How to use prediction of some demons as input for the other ones?
        self.horde = horde

    def interact(self,
                 BATCH_SIZE: 32,
                 env: MiniGridEnv,
                 render: bool = False):

        render_batch = False

        done = False
        env.seed(1337)  # keep the env consistent from episode to episode
        s0 = env.reset()

        while not done:

            if render and env.unwrapped.step_count % 100 == 0:
                render_batch = True

            trajectory = list()
            while len(trajectory) < BATCH_SIZE and not done:

                if render_batch:
                    env.render()

                dist = self.horde.control_demon.behavior_policy(s0)
                a = dist.sample()
                s1, reward, done, info = env.step(a)

                t = Transition(s0, a, reward, s1, done, info=info)
                trajectory.append(t)

                s0 = s1

            for demon in self.horde.demons:
                demon.learn(trajectory)

            # Record batch stats here
            render_batch = False

        # Record episodic stats here
        return env.unwrapped.step_count
