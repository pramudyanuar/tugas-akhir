from env.container_env import ContainerEnv
from rl.low_level_agent import LowLevelAgent

def train():
    env = ContainerEnv()
    agent = LowLevelAgent()

    for episode in range(1000):
        state = env.reset()
        done = False

        while not done:
            pass

if __name__ == "__main__":
    train()