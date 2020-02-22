from gym_minigrid.register import register
from gym_minigrid.envs import EmptyEnv


class EmptyEnv10x10(EmptyEnv):
    """ A single room to learn the hallway option on """
    
    def __init__(self):
        super().__init__(size=10, agent_start_pos=None)


register(
    id='MiniGrid-Empty-10x10-v0',
    entry_point='hrl.envs:EmptyEnv10x10'
)
