import numpy as np
from env.height_map import HeightMap
from env.lbcp import is_stable
from dataset.random_generator import generate_episode

class ContainerEnv:
    def __init__(self):
        self.L = 59
        self.W = 23
        self.H = 23

        self.height_map = HeightMap(self.L, self.W, self.H)
        self.items = []
        self.current_index = 0
        self.total_volume = self.L * self.W * self.H

    def reset(self):
        self.height_map.reset()
        self.items = generate_episode(max_items=50)
        self.current_index = 0
        return self._get_state()

    def _get_state(self):
        item = self.items[self.current_index]
        return self.height_map.normalize(), item

    def step(self, action):
        if action == self.L * self.W:
            return self._get_state(), -1.0, True

        x = action % self.L
        y = action // self.L

        l, w, h = self.items[self.current_index]

        if not self.valid_position(x, y, l, w, h):
            return self._get_state(), -1.0, False

        base_height = self.height_map.max_height_in_region(x, y, l, w)
        self.height_map.update_region(x, y, l, w, base_height + h)

        reward = (l * w * h) / self.total_volume

        self.current_index += 1
        done = self.current_index >= len(self.items)

        return self._get_state(), reward, done

    def valid_position(self, x, y, l, w, h):
        if x + l > self.L or y + w > self.W:
            return False

        if not is_stable(self.height_map.map, x, y, l, w, h, self.H):
            return False

        return True