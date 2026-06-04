import numpy as np
from src.core.container_env import ContainerEnv
from src.learning.agents.oracle_policy import OraclePolicy
from src.data.perfect_pack_generator import PerfectPackGenerator

# Monkeypatch PerfectPackGenerator._generate_single_attempt to skip shuffling
original_generate = PerfectPackGenerator._generate_single_attempt

def unshuffled_generate(self, return_positions=False, fixed_height=None, z_offset=0,
                        enforce_stability=False, cog_tolerance=0.15, max_stability_checks=128):
    # Call original
    if return_positions:
        items, positions = original_generate(self, return_positions=True, fixed_height=fixed_height,
                                             z_offset=z_offset, enforce_stability=enforce_stability,
                                             cog_tolerance=cog_tolerance, max_stability_checks=max_stability_checks)
        # Sort them by their original generation order (which is how they were appended)
        # But wait, original_generate shuffles them before returning.
        # So we can't easily undo it unless we patch the shuffle part inside original_generate.
        return items, positions
    else:
        return original_generate(self, return_positions=False, fixed_height=fixed_height,
                                 z_offset=z_offset, enforce_stability=enforce_stability,
                                 cog_tolerance=cog_tolerance, max_stability_checks=max_stability_checks)

# Instead of patching, let's just patch rng.permutation to be identity
import builtins
class MockRNG:
    def __init__(self, original_rng):
        self.original_rng = original_rng
    def __getattr__(self, name):
        if name == 'permutation':
            return lambda x: np.arange(x) if isinstance(x, int) else np.arange(len(x))
        return getattr(self.original_rng, name)

# Let's run a test with patched permutation
env = ContainerEnv(
    container_length=60,
    container_width=24,
    container_height=26,
    dataset_type='perfect_pack_layered',
    layered_min_height=2,
    layered_max_height=6,
    perfect_pack_sigma=4,
    perfect_pack_size_bias=3.0,
    perfect_pack_mean_ratio=0.25,
    fast_stability_mask=True,
)

# Patch the generator's rng
env.dataset_generator.rng = MockRNG(env.dataset_generator.rng)

policy = OraclePolicy(env, priority='dblf')
num_episodes = 5

utils = []
success_rates = []

for ep in range(num_episodes):
    state, action_mask = env.reset(seed=42 + ep)
    done = False
    
    placed_count = 0
    while not done:
        action = policy.select_action(state, action_mask)
        (next_state, next_mask), reward, done, info = env.step(action)
        state = next_state
        action_mask = next_mask
        if info.get('success', False):
            placed_count += 1
            
    utils.append(env.get_utilization())
    success_rates.append(placed_count / len(env.items) * 100.0)
    
print(f"DBLF (Unshuffled Generation Order):")
print(f"  Mean Utilization: {np.mean(utils):.2f}% ± {np.std(utils):.2f}%")
print(f"  Mean Success Rate: {np.mean(success_rates):.2f}%")
