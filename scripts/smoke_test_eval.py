import sys
import os
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from scripts.evaluate import evaluate
from src.learning.models.high_level_agent import HighLevelAgent
from src.learning.models.actor_critic import ActorCriticNetwork

def main():
    print("=" * 70)
    print("Running Smoke Test on Interrupted Checkpoints")
    print("=" * 70)
    
    # Create models
    model_high = HighLevelAgent(input_dim=60*24 + 4)
    model_low = ActorCriticNetwork(L=60, W=24, action_size=60*24+1)
    
    # Load weights
    checkpoint_high = "logs/training/async_high_level_interrupted.pt"
    checkpoint_low = "logs/training/async_interrupted.pt"
    
    if os.path.exists(checkpoint_high):
        model_high.load_state_dict(torch.load(checkpoint_high, map_location='cpu'))
        print(f"Loaded high-level model from {checkpoint_high}")
    else:
        print("Warning: No high-level model checkpoint found, using random weights.")
        
    if os.path.exists(checkpoint_low):
        model_low.load_state_dict(torch.load(checkpoint_low, map_location='cpu'))
        print(f"Loaded low-level model from {checkpoint_low}")
    else:
        print("Warning: No low-level model checkpoint found, using random weights.")
        
    # Run evaluation
    print("\nRunning evaluation loop...")
    results = evaluate(
        model_high=model_high,
        model_low=model_low,
        num_episodes=3,
        use_mcts=False,
        save_visualizations=True,
        output_csv='logs/evaluation/smoke_test_metrics.csv',
        dataset_type='random'
    )
    print("\n✓ Smoke test evaluation completed successfully!")

if __name__ == "__main__":
    main()
