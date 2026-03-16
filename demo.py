#!/usr/bin/env python3
"""
DEMO: 3D Bin Packing dengan HRL + LBCP + MCTS + Repacking

Script untuk presentasi ke dosen - menunjukkan semua fitur sistem.
Usage: python demo.py
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(__file__))

from env.container_env import ContainerEnv
from rl.high_level_agent import HighLevelAgent
from rl.low_level_agent import LowLevelAgent
from planning.mcts import MCTS
from planning.repack import Repacker
from visualization import ContainerVisualizer
from evaluate import EvaluationMetrics
import torch


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text):
    """Print formatted section."""
    print(f"\n  ► {text}")
    print("  " + "-" * 76)


def demo_basic_packing():
    """Demo: Basic packing tanpa agents."""
    print_header("DEMO 1: BASIC 3D BIN PACKING")
    
    print_section("Inisialisasi Environment")
    
    env = ContainerEnv(max_items=10, seed=42)
    state, action_mask = env.reset()
    
    print(f"    Container Size: {env.L} × {env.W} × {env.H} = {env.container_volume} units³")
    print(f"    Items to Pack: {len(env.items)}")
    print(f"    Database: {[f'{l}×{w}×{h}' for l, w, h in env.items[:3]]}...")
    
    print_section("Melakukan Packing Secara Otomatis")
    
    items_placed = 0
    total_reward = 0.0
    
    for step in range(min(15, len(env.items) * 2)):
        # Random valid action
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            break
        
        action = np.random.choice(valid_actions)
        (next_state, next_mask), reward, done, info = env.step(action)
        
        if info.get('success'):
            items_placed += 1
            total_reward += reward
            print(f"    Step {step+1}: Item placed ✓ | Utilization: {env.get_utilization():.1f}%")
        else:
            print(f"    Step {step+1}: Item skipped | Current util: {env.get_utilization():.1f}%")
        
        state = next_state
        action_mask = next_mask
        
        if done:
            break
    
    print_section("Hasil Packing")
    print(f"    ✓ Items Placed: {items_placed}/{len(env.items)}")
    print(f"    ✓ Volume Utilization: {env.get_utilization():.2f}%")
    print(f"    ✓ Maximum Height: {env.get_max_height()}")
    print(f"    ✓ Total Reward: {total_reward:.4f}")
    
    return env


def demo_with_mcts():
    """Demo: Packing dengan MCTS planning."""
    print_header("DEMO 2: PACKING DENGAN MCTS PLANNING")
    
    print_section("Inisialisasi")
    
    env = ContainerEnv(max_items=8, seed=123)
    state, action_mask = env.reset()
    
    print(f"    ✓ Environment initialized")
    print(f"    ✓ Container: {env.L}×{env.W}×{env.H}")
    
    # Initialize MCTS
    mcts = MCTS(env, budget=30, c=1.4, gamma=0.99)
    print(f"    ✓ MCTS initialized (budget=30, c=1.4)")
    
    print_section("MCTS Look-Ahead Simulation")
    
    mcts_result = mcts.search(state, action_mask, depth_limit=5)
    
    print(f"    ✓ Simulations Run: {mcts_result['tree_stats']['total_simulations']}")
    print(f"    ✓ Nodes Expanded: {mcts_result['tree_stats']['num_nodes_expanded']}")
    print(f"    ✓ Best Action Found: {mcts_result['best_action']}")
    print(f"    ✓ Best Action Value: {mcts_result['tree_stats']['best_value']:.4f}")
    
    print_section("Executing MCTS-Selected Action")
    
    action = mcts_result['best_action']
    (next_state, next_mask), reward, done, info = env.step(action)
    
    print(f"    ✓ Action executed: {action}")
    print(f"    ✓ Reward received: {reward:.4f}")
    print(f"    ✓ Success: {info.get('success', False)}")
    print(f"    ✓ Current utilization: {env.get_utilization():.2f}%")
    
    return env


def demo_lbcp_clustering():
    """Demo: LBCP clustering untuk load-balanced distribution."""
    print_header("DEMO 3: LOAD-BALANCED CLUSTERING (LBCP)")
    
    print_section("Inisialisasi High-Level Agent")
    
    env = ContainerEnv(max_items=12, seed=42)
    state, action_mask = env.reset()
    
    model_high = HighLevelAgent()
    print(f"    ✓ High-Level Agent (Manager) created")
    
    print_section("LBCP Clustering pada Item Batch")
    
    # Prepare batch
    batch_items = env.items[:6]
    print(f"    Items in batch: {batch_items}")
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    high_output = model_high(state_tensor, items_batch=batch_items)
    
    clusters = high_output['cluster_assignment']
    load_balance = high_output['load_balance']
    
    print(f"\n    ✓ Clustering Result:")
    if clusters:
        for i, cluster in enumerate(clusters):
            volume = sum(l*w*h for l, w, h in cluster)
            print(f"      Cluster {i+1}: {len(cluster)} items | Volume: {volume}")
    
    print(f"\n    ✓ Load Balance Score: {load_balance:.4f}")
    print(f"      (1.0 = perfectly balanced | 0.0 = completely imbalanced)")
    
    return env, model_high


def demo_repacking():
    """Demo: Repacking strategy."""
    print_header("DEMO 4: REPACKING MECHANISM")
    
    print_section("Setup: Placing Items First")
    
    env = ContainerEnv(max_items=15, seed=42)
    state, action_mask = env.reset()
    
    # Place 8 items
    for i in range(8):
        if env.current_index < len(env.items):
            action = (i * 60) % (env.L * env.W)
            (next_state, next_mask), reward, done, info = env.step(action)
    
    old_util = env.get_utilization()
    old_height = env.get_max_height()
    
    print(f"    ✓ Placed {len(env.placed_items)} items")
    print(f"    ✓ Utilization: {old_util:.2f}%")
    print(f"    ✓ Max Height: {old_height}")
    
    print_section("Testing Repacking Strategies")
    
    repacker = Repacker(container_dims=(59, 23, 23))
    
    strategies = ['blf', 'load_balanced', 'min_height']
    results = {}
    
    for strategy in strategies:
        env_copy = ContainerEnv(max_items=15, seed=42)
        env_copy.placed_items = env.placed_items.copy()
        env_copy.placed_positions = env.placed_positions.copy()
        env_copy.height_map.map = env.height_map.map.copy()
        
        repack_result = env_copy.perform_repack(strategy=strategy)
        results[strategy] = repack_result
        
        if repack_result['success']:
            print(f"\n    {strategy.upper()}:")
            print(f"      ✓ Success: Yes")
            print(f"      ✓ Old util: {repack_result['old_utilization']:.2f}%")
            print(f"      ✓ New util: {repack_result['new_utilization']:.2f}%")
            print(f"      ✓ Improvement: {repack_result['improvement']:.2f}x")
            print(f"      ✓ Reward: {repack_result['reward']:.4f}")
        else:
            print(f"\n    {strategy.upper()}: Failed")
    
    # Find best strategy
    best_strategy = max(results.items(), 
                       key=lambda x: x[1]['reward'] if x[1]['success'] else -1)
    
    print_section("Best Repacking Strategy")
    print(f"    ✓ Strategy: {best_strategy[0].upper()}")
    print(f"    ✓ Reward: {best_strategy[1]['reward']:.4f}")
    
    return env


def demo_evaluation_metrics():
    """Demo: Evaluation metrics."""
    print_header("DEMO 5: COMPREHENSIVE EVALUATION")
    
    print_section("Setup Environment")
    
    env = ContainerEnv(max_items=10, seed=42)
    state, action_mask = env.reset()
    
    # Place items
    for i in range(7):
        if env.current_index < len(env.items):
            action = (i * 70) % (env.L * env.W)
            (next_state, next_mask), reward, done, info = env.step(action)
    
    print(f"    ✓ Placed {len(env.placed_items)} items")
    
    print_section("Computing Evaluation Metrics")
    
    evaluator = EvaluationMetrics(container_dims=(59, 23, 23))
    
    # Compute metrics
    utilization = evaluator.compute_volume_utilization(env.placed_items)
    success_rate = evaluator.compute_success_rate(len(env.items), env.placed_items)
    
    load_dist = evaluator.compute_load_distribution_balance(
        env.placed_positions, env.placed_items, env.height_map.map
    )
    
    stability = evaluator.compute_stability_rate(
        env.height_map.map, env.placed_positions, env.placed_items
    )
    
    print(f"\n    1. VOLUME UTILIZATION")
    print(f"       ► {utilization:.2f}%")
    
    print(f"\n    2. LOAD DISTRIBUTION")
    print(f"       ► Balance Score: {load_dist['load_balance']:.4f}")
    print(f"       ► CoG Deviation: {load_dist['cog_deviation']:.4f}")
    
    print(f"\n    3. STABILITY RATE (LBCP)")
    print(f"       ► {stability['stability_rate']:.2%}")
    print(f"       ► {stability['stable_count']}/{stability['total_count']} items stable")
    
    print(f"\n    4. SUCCESS RATE")
    print(f"       ► {success_rate:.2%} items placed")
    
    print_section("Interpretation")
    print(f"    ✓ Utilization: {'EXCELLENT' if utilization > 70 else 'GOOD' if utilization > 50 else 'FAIR' if utilization > 30 else 'POOR'}")
    print(f"    ✓ Load Balance: {'EXCELLENT' if load_dist['load_balance'] > 0.8 else 'GOOD' if load_dist['load_balance'] > 0.6 else 'FAIR'}")
    print(f"    ✓ Stability: {'EXCELLENT' if stability['stability_rate'] > 0.95 else 'GOOD' if stability['stability_rate'] > 0.8 else 'FAIR'}")
    
    return env


def demo_visualization():
    """Demo: Generate visualizations."""
    print_header("DEMO 6: VISUALIZATION FOR PRESENTATION")
    
    print_section("Creating Sample Packing")
    
    env = ContainerEnv(max_items=10, seed=42)
    state, action_mask = env.reset()
    
    # Place items
    for i in range(8):
        if env.current_index < len(env.items):
            action = (i * 50) % (env.L * env.W)
            (next_state, next_mask), reward, done, info = env.step(action)
    
    print(f"    ✓ Placed {len(env.placed_items)} items")
    
    print_section("Generating Visualizations")
    
    viz = ContainerVisualizer(container_dims=(59, 23, 23))
    metrics = viz.save_all_visualizations(env, output_dir='./visualizations')
    
    print(f"\n    Generated files:")
    print(f"      1. 01_packing_2d.png - Top view dengan height map")
    print(f"      2. 02_packing_3d.png - 3D visualization")
    print(f"      3. 03_cross_sections.png - Cross-section views")
    print(f"      4. 04_statistics.png - Final statistics dashboard")
    
    print_section("Metrics Summary")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    • {key}: {value:.2f}")
        else:
            print(f"    • {key}: {value}")
    
    return env


def main():
    """Run semua demos."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  3D BIN PACKING DENGAN HIERARCHICAL REINFORCEMENT LEARNING".center(78) + "█")
    print("█" + "  + LBCP + MCTS + Repacking".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    try:
        # Run demos
        env1 = demo_basic_packing()
        env2 = demo_with_mcts()
        env3, model_high = demo_lbcp_clustering()
        env4 = demo_repacking()
        env5 = demo_evaluation_metrics()
        env6 = demo_visualization()
        
        # Summary
        print_header("SUMMARY - SISTEM SIAP DIPRESENTASIKAN")
        
        print_section("Komponen yang Sudah Diimplentasi")
        print("""
    ✅ 3D Bin Packing Problem (3DPBP)
       - Placement dengan collision detection
       - Action masking untuk valid positions
    
    ✅ Load-Balanced Clustering Problem (LBCP)
       - Weight-based clustering
       - Stability validation
       - Center of gravity tracking
    
    ✅ Hierarchical Reinforcement Learning (HRL)
       - High-Level Agent (Manager)
       - Low-Level Agent (Worker)
       - PPO algorithm
    
    ✅ Monte Carlo Tree Search (MCTS)
       - UCB1 exploration-exploitation
       - Weighted rollout policy
       - Look-ahead mechanism
    
    ✅ Repacking Mechanism
       - Bottom-Left-Fill strategy
       - Load-Balanced strategy
       - Minimize-Height strategy
       - Auto-selection
    
    ✅ Comprehensive Evaluation
       - Volume Utilization
       - Load Distribution
       - Stability Rate
       - Success Rate
       - Center of Gravity Deviation
    
    ✅ Visualization & Reporting
       - 2D Top View
       - 3D Visualization
       - Cross-sections
       - Statistics Dashboard
        """)
        
        print_section("Files untuk Presentasi")
        print("""
    📊 Visualizations (./visualizations/):
       - 01_packing_2d.png
       - 02_packing_3d.png
       - 03_cross_sections.png
       - 04_statistics.png
    
    📁 Source Code:
       - env/container_env.py - Environment
       - rl/high_level_agent.py - HRL Manager
       - rl/low_level_agent.py - HRL Worker
       - planning/mcts.py - MCTS Planning
       - planning/repack.py - Repacking
       - evaluate.py - Evaluation Metrics
       - visualization.py - Visualizations
       - demo.py - This demo script
        """)
        
        print_header("✓ DEMO COMPLETED SUCCESSFULLY!")
        print("\nUntuk presentasi ke dosen:")
        print("  1. Jalankan: python demo.py")
        print("  2. Lihat visualizations di folder ./visualizations/")
        print("  3. Jelaskan architecture dan hasil dari setiap komponen")
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
