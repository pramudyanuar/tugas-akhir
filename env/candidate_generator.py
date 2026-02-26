import numpy as np

class CandidateGenerator:
    def __init__(self, grid_L, grid_W):
        self.L = grid_L
        self.W = grid_W

    def generate_all(self):
        candidates = []
        for x in range(self.L):
            for y in range(self.W):
                candidates.append((x, y))
        return candidates