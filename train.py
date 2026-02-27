import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(__file__))

from env.container_env import ContainerEnv
from rl.ppo import PPO
from utils.metrics import Metrics


class TrainingLogger:
    """
    Logger untuk training metrics.
    """
    
    def __init__(self, log_dir='logs/training'):
        """
        Initialize logger.
        
        Args:
            log_dir (str): Directory untuk save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_utilizations = []
        self.episode_success_rates = []
        
    def log_episode(self, episode_reward, episode_length, utilization, 
                   items_placed, total_items):
        """Log episode metrics."""
        success_rate = (items_placed / total_items * 100) if total_items > 0 else 0.0
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_utilizations.append(utilization)
        self.episode_success_rates.append(success_rate)
    
    def get_stats(self):
        """Get training statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'reward_mean': np.mean(self.episode_rewards[-100:]),
            'reward_std': np.std(self.episode_rewards[-100:]),
            'length_mean': np.mean(self.episode_lengths[-100:]),
            'utilization_mean': np.mean(self.episode_utilizations[-100:]),
            'success_rate_mean': np.mean(self.episode_success_rates[-100:]),
        }
    
    def print_episode_summary(self, episode, reward, length, utilization, items_placed, total_items):
        """Print episode summary."""
        success_rate = (items_placed / total_items * 100) if total_items > 0 else 0.0
        
        print(f"Episode {episode:4d} | "
              f"Reward: {reward:8.4f} | "
              f"Length: {length:3d} | "
              f"Util: {utilization:6.2f}% | "
              f"Success: {success_rate:5.1f}%")


class TrainingLoop:
    """
    Training loop untuk PPO agent dengan 3D bin packing environment.
    
    Features:
    - Collect N steps dari environment
    - Compute returns menggunakan GAE
    - Update network dengan PPO
    - Log metrics: reward, utilization, episode length
    """
    
    def __init__(self, env, ppo_agent, n_steps=2048, 
                 device='cpu', seed=None):
        """
        Initialize training loop.
        
        Args:
            env (ContainerEnv): Environment
            ppo_agent (PPO): PPO agent
            n_steps (int): Number of steps to collect per update
            device (str): 'cpu' atau 'cuda'
            seed (int): Random seed
        """
        self.env = env
        self.ppo = ppo_agent
        self.n_steps = n_steps
        self.device = device
        self.seed = seed
        
        self.logger = TrainingLogger()
        
        self.total_steps = 0
        self.episode_count = 0
    
    def collect_steps(self, n_steps):
        """
        Collect N steps dari environment.
        
        Args:
            n_steps (int): Number of steps to collect
            
        Returns:
            dict: Collected trajectories
        """
        steps_collected = 0
        episode_info = {
            'rewards': [],
            'lengths': [],
            'utilizations': [],
            'placed_items': [],
            'total_items': []
        }
        
        # Reset environment
        state, action_mask = self.env.reset(seed=self.seed)
        if self.seed is not None:
            self.seed += 1  # Increment seed untuk variety
        
        while steps_collected < n_steps:
            # Select action
            action, log_prob, value = self.ppo.select_action(state, action_mask)
            
            # Take step in environment
            (next_state, next_mask), reward, done, info = self.env.step(action)
            
            # Store transition
            self.ppo.store_transition(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                action_mask=action_mask,
                done=1.0 if done else 0.0
            )
            
            steps_collected += 1
            self.total_steps += 1
            
            state = next_state
            action_mask = next_mask
            
            # Episode done
            if done:
                episode_reward = self.env.episode_reward
                episode_length = self.env.episode_length
                utilization = self.env.get_utilization()
                items_placed = len(self.env.placed_items)
                total_items = len(self.env.items)
                
                episode_info['rewards'].append(episode_reward)
                episode_info['lengths'].append(episode_length)
                episode_info['utilizations'].append(utilization)
                episode_info['placed_items'].append(items_placed)
                episode_info['total_items'].append(total_items)
                
                self.logger.log_episode(episode_reward, episode_length, 
                                       utilization, items_placed, total_items)
                self.logger.print_episode_summary(self.episode_count, episode_reward, 
                                                 episode_length, utilization, 
                                                 items_placed, total_items)
                
                self.episode_count += 1
                
                # Reset environment untuk episode baru
                state, action_mask = self.env.reset(seed=self.seed)
                if self.seed is not None:
                    self.seed += 1
        
        # Get next value untuk GAE bootstrap
        if not done:
            _, _, next_value = self.ppo.select_action(state, action_mask)
        else:
            next_value = 0.0
        
        return episode_info, next_value
    
    def train_epoch(self, num_epochs=4, log_frequency=10):
        """
        Train untuk satu epoch (collect steps + update).
        
        Args:
            num_epochs (int): Jumlah PPO update epochs
            log_frequency (int): Frequency untuk print summary
        """
        print(f"\n{'='*70}")
        print(f"Training Epoch {self.episode_count // log_frequency + 1}")
        print(f"{'='*70}\n")
        
        # Collect trajectories
        episode_info, next_value = self.collect_steps(self.n_steps)
        
        # Update PPO network
        print(f"\nUpdating network... ", end='', flush=True)
        self.ppo.update(next_value=next_value, num_epochs=num_epochs, batch_size=64)
        print("Done!\n")
        
        # Print statistics
        stats = self.logger.get_stats()
        print(f"{'='*70}")
        print("Training Statistics (last 100 episodes):")
        print(f"{'='*70}")
        print(f"Episode Reward:      {stats.get('reward_mean', 0):.4f} ± {stats.get('reward_std', 0):.4f}")
        print(f"Episode Length:      {stats.get('length_mean', 0):.1f} steps")
        print(f"Container Util:      {stats.get('utilization_mean', 0):.2f}%")
        print(f"Success Rate:        {stats.get('success_rate_mean', 0):.1f}%")
        print(f"Total Steps:         {self.total_steps}")
        print(f"Total Episodes:      {self.episode_count}")
        print(f"{'='*70}\n")


def train(num_epochs=10, n_steps=2048, max_items=20, seed=42, device='cpu'):
    """
    Main training function.
    
    Args:
        num_epochs (int): Number of training epochs
        n_steps (int): Steps per epoch
        max_items (int): Max items per episode
        seed (int): Random seed
        device (str): 'cpu' atau 'cuda'
    """
    print("\n" + "="*70)
    print("3D BIN PACKING WITH PPO TRAINING")
    print("="*70 + "\n")
    
    # Initialize environment
    print("Initializing environment...")
    env = ContainerEnv(max_items=max_items, seed=seed)
    state, action_mask = env.reset()
    
    state_size = env.state_size
    action_size = env.action_size
    
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Container: {env.L} x {env.W} x {env.H}")
    print(f"  Max items: {max_items}\n")
    
    # Initialize PPO agent
    print("Initializing PPO agent...")
    ppo = PPO(
        state_size=state_size,
        action_size=action_size,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        device=device
    )
    print("  Done!\n")
    
    # Initialize training loop
    training_loop = TrainingLoop(
        env=env,
        ppo_agent=ppo,
        n_steps=n_steps,
        device=device,
        seed=seed
    )
    
    # Training loop
    try:
        for epoch in range(num_epochs):
            training_loop.train_epoch(num_epochs=3, log_frequency=1)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"logs/training/checkpoint_epoch_{epoch+1}.pt"
                ppo.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}\n")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70 + "\n")
    
    # Final statistics
    stats = training_loop.logger.get_stats()
    print("Final Statistics:")
    print(f"  Average Reward:    {stats.get('reward_mean', 0):.4f}")
    print(f"  Average Length:    {stats.get('length_mean', 0):.1f} steps")
    print(f"  Average Util:      {stats.get('utilization_mean', 0):.2f}%")
    print(f"  Average Success:   {stats.get('success_rate_mean', 0):.1f}%")
    print(f"  Total Steps:       {training_loop.total_steps}")
    print(f"  Total Episodes:    {training_loop.episode_count}\n")
    
    return training_loop, ppo


if __name__ == "__main__":
    """Test training loop"""
    
    print("\n" + "="*70)
    print("TESTING TRAINING LOOP")
    print("="*70)
    
    # Quick test dengan sedikit epochs dan items
    training_loop, ppo = train(
        num_epochs=2,
        n_steps=128,
        max_items=5,
        seed=123,
        device='cpu'
    )
    
    print("\nTraining loop test completed successfully!")