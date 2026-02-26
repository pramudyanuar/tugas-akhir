import torch
import torch.nn as nn
import torch.optim as optim

class PPO:
    def __init__(self, model, lr=3e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, states, actions, rewards, old_log_probs):
        pass