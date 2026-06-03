import time
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from src.core.container_env import ContainerEnv

print("Initializing ContainerEnv...", flush=True)
start = time.time()
env = ContainerEnv(dataset_type='perfect_pack_layered', seed=42)
print(f"Env initialized in {time.time() - start:.4f}s", flush=True)

print("Resetting environment (generating items)...", flush=True)
start = time.time()
state, mask = env.reset()
print(f"Environment reset completed in {time.time() - start:.4f}s", flush=True)
print(f"Number of items: {len(env.items)}", flush=True)
