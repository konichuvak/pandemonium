class Display:

    def __init__(self, env):
        self.env = env
        self.env_img = self.env.render(close=True)
