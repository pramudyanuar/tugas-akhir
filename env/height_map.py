import numpy as np

class HeightMap:
    def __init__(self, length=59, width=23, height=23):
        self.L = length
        self.W = width
        self.H = height
        self.map = np.zeros((self.L, self.W), dtype=np.int32)

    def reset(self):
        self.map.fill(0)

    def get_region(self, x, y, l, w):
        return self.map[x:x+l, y:y+w]

    def max_height_in_region(self, x, y, l, w):
        return np.max(self.get_region(x, y, l, w))

    def update_region(self, x, y, l, w, new_height):
        self.map[x:x+l, y:y+w] = new_height

    def normalize(self):
        return self.map / self.H