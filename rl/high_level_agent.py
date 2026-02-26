import torch.nn as nn

class HighLevelAgent(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(59*23 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # 6 orientation + repack + no-op
        )

    def forward(self, state):
        return self.net(state)