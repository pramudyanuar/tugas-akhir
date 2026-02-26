import torch
import torch.nn as nn
import torch.nn.functional as F

class LowLevelAgent(nn.Module):
    def __init__(self, L=59, W=23, action_dim=59*23+1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(32*L*W + 3, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, height_map, item_dim):
        x = self.conv(height_map)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, item_dim], dim=1)
        logits = self.fc(x)
        return logits