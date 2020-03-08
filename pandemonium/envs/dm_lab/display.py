import numpy as np
import pygame

BLUE = (128, 128, 255)
RED = (255, 192, 192)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class DeepmindLabDisplay:
    """ Contains visualization tools for DM lab environments.

    Most of the code below was borrowed from:
    https://github.com/miyosuda/unreal/blob/master/display.py
    """

    def __init__(self, display_size, env):
        self.env = env
        self.obs_shape = env.observation_space.shape

        pygame.init()
        self.display_size = display_size
        self.surface = pygame.display.set_mode(display_size, 0, 24)
        self.surface.fill(BLACK)
        pygame.display.set_caption('UNREAL')
        self.font = pygame.font.SysFont(None, 20)

    @staticmethod
    def scale_image(image, scale):
        return image.repeat(scale, axis=0).repeat(scale, axis=1)

    def draw_text(self, text, left, top, color=WHITE):
        text = self.font.render(text, True, color, BLACK)
        text_rect = text.get_rect()
        text_rect.left = left
        text_rect.top = top
        self.surface.blit(text, text_rect)

    def draw_center_text(self, text, center_x, top):
        text = self.font.render(text, True, WHITE, BLACK)
        text_rect = text.get_rect()
        text_rect.centerx = center_x
        text_rect.top = top
        self.surface.blit(text, text_rect)

    def show_pixel_change(self, pixel_change, left, top, rate, label):
        pixel_change_ = np.clip(pixel_change * 255.0 * rate, 0.0, 255.0)
        data = pixel_change_.astype(np.uint8)
        data = np.stack([data for _ in range(3)], axis=2)
        data = self.scale_image(data, 4)
        image = pygame.image.frombuffer(data, (20 * 4, 20 * 4), 'RGB')
        self.surface.blit(image, (left + 8 + 4, top + 8 + 4))
        self.draw_center_text(label, left + 100 / 2, top + 100)

    def show_policy(self, pi):
        start_x = 10
        y = 150

        for i in range(len(pi)):
            width = pi[i] * 100
            pygame.draw.rect(self.surface, WHITE, (start_x, y, width, 10))
            y += 20
        self.draw_center_text("PI", 50, y)

    def show_image(self, state):
        state_ = state * 255.0
        data = state_.astype(np.uint8)
        image = pygame.image.frombuffer(data, self.obs_shape[:2], 'RGB')
        self.surface.blit(image, (8, 8))
        # self.draw_center_text("Observation Input", 50, 100)

    def show_value(self, values):
        min_v = float("inf")
        max_v = float("-inf")

        for v in values:
            min_v = min(min_v, v)
            max_v = max(max_v, v)

        top = 250
        left = self.obs_shape[0] + 15
        width = 100
        height = 100
        bottom = top + width
        right = left + height

        d = max_v - min_v
        last_r = 0.0
        for i, v in enumerate(values):
            r = (v - min_v) / d
            if i > 0:
                x0 = i - 1 + left
                x1 = i + left
                y0 = bottom - last_r * height
                y1 = bottom - r * height
                pygame.draw.line(self.surface, BLUE, (x0, y0), (x1, y1), 1)
            last_r = r

        pygame.draw.line(self.surface, WHITE, (left, top), (left, bottom), 1)
        pygame.draw.line(self.surface, WHITE, (right, top), (right, bottom), 1)
        pygame.draw.line(self.surface, WHITE, (left, top), (right, top), 1)
        pygame.draw.line(self.surface, WHITE, (left, bottom), (right, bottom),
                         1)

        self.draw_center_text("V", left + width / 2, bottom + 10)

    def show_reward_prediction(self, rp_c, reward):
        start_x = self.obs_shape[0] + 15
        reward_index = reward + 1
        y = 150

        labels = ["-", "0", "+"]

        for i in range(len(rp_c)):
            width = rp_c[i] * 100

            if i == reward_index:
                color = RED
            else:
                color = WHITE
            pygame.draw.rect(self.surface, color, (start_x + 15, y, width, 10))
            self.draw_text(labels[i], start_x, y - 1, color)
            y += 20

        self.draw_center_text("RP", start_x + 50 / 2, y)

    def get_frame(self):
        data = self.surface.get_buffer().raw
        return data
