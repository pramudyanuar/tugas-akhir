import numpy as np
from src.core.container_env import ContainerEnv
from src.utils.item_utils import get_item_dims

# Let's run a test with ground truth positions
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
    use_structural_validation=False, # Disable during GT placement to ensure no numerical precision stability check failures
)

num_episodes = 5

for ep in range(num_episodes):
    state, action_mask = env.reset(seed=42 + ep)
    
    # We place each item at its ground truth position
    # Ground truth positions are stored in env.ground_truth_positions
    # Let's map it
    gt_positions = env.ground_truth_positions
    items = env.items
    
    print(f"Episode {ep} | Number of items: {len(items)} | GT positions: {len(gt_positions)}")
    
    # Let's execute placements manually
    env.height_map.reset()
    env.feasibility_map.fill(True)
    env.placed_items = []
    env.placed_positions = []
    
    for idx, (item, (x, y, z)) in enumerate(zip(items, gt_positions)):
        item_l, item_w, item_h = get_item_dims(item)
        base_height = env.height_map.max_height_in_region(x, y, item_l, item_w)
        new_height = base_height + item_h
        env.height_map.update_region(x, y, item_l, item_w, new_height)
        env.placed_items.append(item)
        env.placed_positions.append((x, y, base_height))
        env.top_item_map[x:x+item_l, y:y+item_w] = idx
        
    print(f"  Final Utilization: {env.get_utilization():.2f}%")
