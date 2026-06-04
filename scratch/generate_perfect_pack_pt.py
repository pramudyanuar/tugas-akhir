import os
import torch
import numpy as np
from src.data.perfect_pack_generator import PerfectPackGenerator

def generate_dataset():
    # We will generate for L=60, W=24, H=26, or L=10, W=10, H=10.
    # To keep it generic and fully compatible with the 10x10x10 standard container size,
    # let's generate instances for L=10, W=10, H=10.
    # Why? Because the user requests rs.pt to be replaced and rs.pt was 10x10x10.
    # Let's generate a dataset of 100 episodes of layered perfect packing.
    L, W, H = 10, 10, 10
    generator = PerfectPackGenerator(
        bin_width=L,
        bin_height=W,
        seed=42
    )
    
    dataset = []
    num_episodes = 100
    
    print(f"Generating {num_episodes} layered perfect pack episodes...")
    for ep in range(num_episodes):
        items, positions = generator.generate_layered_perfect_pack_with_positions(
            container_height=H,
            min_layer_height=2,
            max_layer_height=5,
            shuffle=True,  # Shuffle items to simulate online arrival sequence
            enforce_stability=True
        )
        
        # Add weight and load_bearing properties to each item
        episode_items = []
        for item in items:
            l, w, h = item['l'], item['w'], item['h']
            vol = l * w * h
            weight = float(vol) + np.random.uniform(-0.1, 0.1)  # small pertubation
            stacking = item['stacking']
            
            # Stacking capacity based on class
            if stacking == 'fragile':
                load_bearing = 10.0  # Fragile items can only support very small weight
            elif stacking == 'no_stack':
                load_bearing = 0.0   # No stack items cannot support any weight
            else:
                load_bearing = float(vol * 5.0)  # Stackable items can support up to 5x their volume weight
                
            episode_items.append({
                'l': l,
                'w': w,
                'h': h,
                'stacking': stacking,
                'weight': weight,
                'load_bearing': load_bearing
            })
            
        dataset.append(episode_items)
        if (ep + 1) % 20 == 0:
            print(f"  Generated {ep + 1}/{num_episodes} episodes...")
            
    os.makedirs('dataset', exist_ok=True)
    torch.save(dataset, 'dataset/perfect_pack.pt')
    print("Dataset saved successfully to dataset/perfect_pack.pt")

if __name__ == '__main__':
    generate_dataset()
