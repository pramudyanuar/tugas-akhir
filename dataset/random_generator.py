import numpy as np

def generate_episode(max_items=50):
    items = []

    for _ in range(max_items):
        l = np.random.randint(2, 15)
        w = np.random.randint(2, 12)
        h = np.random.randint(2, 12)
        items.append((l, w, h))

    return items