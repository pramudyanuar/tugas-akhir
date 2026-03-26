# Hierarchical RL Architecture for 3D Bin Packing

## Overview

This codebase implements a **Two-Level Hierarchical Reinforcement Learning** system for 3D bin packing:

```
DECISION FLOW:
┌─ Level 1: HIGH-LEVEL (WHAT?) ─┐
│                                │
│   HighLevelAgent               │
│   └─> Strategy Selection       │
│       (orientation, repack,    │
│        skip, etc.)             │
│                                │
└─────────────────────────────────┘
           |
           v (strategy)
┌─ Level 2: LOW-LEVEL (WHERE?) ──┐
│                                │
│   PPO Agent                    │
│   └─> Position Selection       │
│       (x, y, z coordinates)    │
│                                │
│   Uses ActorCriticNetwork      │
│   (CNN for feature extraction) │
│                                │
└─────────────────────────────────┘
           |
           v (position)
   Environment.step()
       |
       v
   Execute & Collect Reward
```

## Level 1: High-Level Agent (Strategy Selection)

**File:** [models/high_level_agent.py](models/high_level_agent.py)

**Purpose:** Decide WHAT strategy to apply (macro-level decision)

**Strategies:**
- 6 item orientations (0°, 90°, 180°, 270° rotations + flips)
- 1 repack action (rearrange existing items)
- 1 skip/no-op action

**Key Concepts:**
- Input: Current container state (height map + item dimensions)
- Output: Strategy index + log probability
- Uses LBCP (Load-Balanced Clustering Problem) heuristic
- Ready for future end-to-end hierarchical training

**Usage in Code:**
```python
# In train.py - _select_hierarchical_action()
high_output = self.high_level_agent(state_tensor)
strategy, strategy_log_prob = self.high_level_agent.select_strategy(
    strategy_logits, sample=sample_strategy
)
```

## Level 2: Low-Level Agent (PPO + ActorCriticNetwork)

**Files:** 
- [agents/ppo.py](agents/ppo.py) - Training algorithm
- [models/actor_critic.py](models/actor_critic.py) - Neural network

**Purpose:** Learn WHERE to place items given a strategy (low-level decision)

**Algorithm: Proximal Policy Optimization (PPO)**
- Policy gradient method with clipped objective
- Prevents large policy updates (more stable)
- Generalizable Advantage Estimation (GAE) for better returns

**Network: ActorCriticNetwork (CNN-based)**
```
Input: [height_map_flattened (L*W), item_width, item_height, item_depth]
  |
  v
Conv Blocks (feature extraction)
  |
  v
Policy Head: Output logits for position (L*W + 1 actions = place or skip)
Value Head: Output single value estimate (state value function)
```

**Key Concepts:**
- **Actor:** Policy network that selects positions
- **Critic:** Value network that estimates state value
- **Action Masking:** Only valid positions allowed (based on strategy & geometry)
- **Entropy Bonus:** Encourages exploration
- **GAE:** Balance bias-variance tradeoff in advantage estimation

## Training Pipeline

### Data Collection Phase
```python
# In train.py - collect_steps()
for step in range(steps_per_episode):
    # Level 1: Get strategy
    strategy, strategy_log_prob = high_level_agent.select_action()
    
    # Level 2: Get position
    action, log_prob, value = ppo_agent.select_action(state, strategy)
    
    # Execute in environment
    next_state, reward, done, info = env.step(action)
    
    # Store trajectory
    ppo_agent.store_transition(
        state, action, reward, value, log_prob, done
    )
```

### Training Update Phase
```python
# In train.py - train_episode()
# Gap-based returns calculation
gae_returns, gae_advantages = compute_gae(
    rewards, values, dones, gae_lambda, gamma
)

# PPO mini-batch updates
for epoch in range(num_epochs):
    for batch in mini_batches:
        # Forward pass
        new_logits, new_values = network(states)
        new_log_probs = compute_log_prob(new_logits, actions, action_masks)
        
        # PPO-Clip loss
        ratio = exp(new_log_probs - old_log_probs)
        loss_clip = -min(ratio * advantages, 
                         clipped(ratio, 1-ε, 1+ε) * advantages)
        
        # Value loss
        loss_value = (new_values - returns)^2
        
        # Entropy bonus
        loss_entropy = -entropy(new_logits)
        
        # Total loss
        total_loss = loss_clip + value_coef * loss_value + entropy_coef * loss_entropy
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

## Key Components

### 1. Container Environment
**File:** [env/container_env.py](env/container_env.py)

- Manages 3D bin packing simulation
- Tracks item placement, stability, collisions
- Computes rewards (volume utilization, efficiency)
- Handles deadlock detection

### 2. Action Masking
**Files:**
- [env/action_mask.py](env/action_mask.py) - Generate valid position masks
- [env/candidate_generator.py](env/candidate_generator.py) - Filter positions by strategy

### 3. Memory/Trajectory Storage
**File:** [common/memory.py](common/memory.py)

Stores trajectories for PPO training:
```python
Memory buffer: {
    'states': [...],
    'actions': [...],
    'rewards': [...],
    'values': [...],
    'log_probs': [...],
    'dones': [...]
}
```

### 4. Deadlock Resolution
**File:** [planning/repack.py](planning/repack.py)

When PPO gets stuck:
1. Try `attempt_repack()` - rearrange existing items
2. If still stuck, use MCTS search for escape path
3. If exhausted, episode ends (failure)

## Hierarchical Benefits

### 1. Search Space Reduction
- Without hierarchy: $L \times W \times Z \approx 59 \times 23 \times 10 = 13,570$ positions
- With hierarchy: 8 strategies × $\sim 100$ filtered positions = more focused search
- PPO can focus on fine-grained position optimization

### 2. Better Exploration
- High-level helps avoid local minima (e.g., forcing repack when stuck)
- Low-level learns to exploit valid positions within strategy

### 3. Clearer Semantics
- High-level = "WHAT strategy?"
- Low-level = "WHERE to place given strategy?"
- Each level has clear, interpretable responsibility

### 4. Better Training Stability
- Separated concerns = easier optimization
- Policy gradients less noisy (smaller action space per level)

## Deprecated Components

### LowLevelAgent (Old)
**File:** [models/low_level_agent.py](models/low_level_agent.py)

- **Status:** DEPRECATED - kept for backward compatibility only
- **Why Replaced:** Old implementation tried to do both strategy + position selection
- **Modern Replacement:** PPO + ActorCriticNetwork (better architecture)
- **Keep For:** Legacy reference, historical context

### Training with Old LowLevelAgent
```python
# OLD (DON'T USE):
position = low_level_agent(state)  # Both strategy + position

# NEW (USE THIS):
strategy = high_level_agent(state)  # Strategy
position = ppo_agent(state, strategy)  # Position given strategy
```

## File Organization

```
tugas-akhir/
├── train.py                    # Main training entry point
├── models/
│   ├── high_level_agent.py    # Strategy selection network
│   ├── actor_critic.py        # Position selection network (with CNN)
│   └── low_level_agent.py     # [DEPRECATED] Old implementation
├── agents/
│   ├── ppo.py                 # PPO training algorithm
│   ├── mcts.py                # MCTS search (planning)
│   └── oracle_policy.py       # Oracle/baseline policies
├── env/
│   ├── container_env.py       # 3D bin packing environment
│   ├── action_mask.py         # Valid action generation
│   ├── candidate_generator.py # Strategy-based position filtering
│   ├── height_map.py          # Container height map management
│   └── lbcp.py               # Load-Balanced Clustering
├── planning/
│   ├── mcts.py               # Monte Carlo Tree Search
│   └── repack.py             # Deadlock resolution (rearrangement)
├── common/
│   ├── memory.py             # Trajectory buffer
│   └── mcts_node.py          # MCTS tree node
└── utils/
    ├── metrics.py            # Training metrics
    └── logger.py             # Logging utilities
```

## Configuration & Hyperparameters

### PPO Hyperparameters (agents/ppo.py)
```python
learning_rate = 3e-4           # Adam optimizer LR
gamma = 0.99                   # Discount factor
gae_lambda = 0.95              # GAE lambda (bias-variance)
clip_ratio = 0.2               # PPO epsilon (ε)
entropy_coef = 0.01            # Entropy bonus coefficient
value_coef = 0.5               # Value loss coefficient
```

### Training Hyperparameters (train.py)
```python
num_episodes = 1000            # Total training episodes
steps_per_episode = 500        # Max steps per episode
batch_size = 32                # Mini-batch size for PPO
num_epochs = 3                 # PPO update epochs
gamma = 0.99                   # Reward discount
```

## Quick Start

### Training
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py
```

### Visualization
```bash
python visualization.py
```

## Troubleshooting

### Issue: "Module not found" errors
**Solution:** Check [__init__.py](env/__init__.py) files in env/, planning/, dataset/ folders - they must export all classes

### Issue: PPO not learning
**Solution:** Check action masking - invalid/masked positions should have logit forced to -inf

### Issue: Frequent deadlocks
**Solution:** Increase repack attempts before MCTS, or improve strategy selection in HighLevelAgent

### Issue: High variance in rewards
**Solution:** Increase `gae_lambda` toward 1.0 (reduce bias), or increase `entropy_coef` for more exploration

## References

### Paper: Hierarchical RL
- Kulkarni et al. "Hierarchical Deep RL with Deliberate Subgoals"
- Options Framework: Sutton et al.

### PPO
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- State-of-the-art policy gradient method with clipped objective

### 3D Bin Packing
- NP-complete problem (exponential complexity)
- Heuristics: FFD (First-Fit Decreasing), genetic algorithms, simulated annealing
- RL offers learning-based alternative

## Contact & Questions

For questions about architecture, consult:
1. Module docstrings (e.g., `train.py`, `agents/ppo.py`, `models/high_level_agent.py`)
2. This file for system-level overview
3. Individual file comments for implementation details
