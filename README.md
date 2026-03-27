![alt text](image.png)

## Ringkasan Proyek

Repositori ini berisi sistem online 3D container loading berbasis kontrol hierarkis.
Sistem menggabungkan:

1. High-level controller untuk keputusan strategi makro.
2. Low-level policy (actor-critic/PPO) untuk pemilihan aksi penempatan detail.
3. Rearrangement planner berbasis MCTS saat sistem mengalami deadlock (tidak ada posisi legal).

## 📐 Reorganisasi Codebase (v2.0)

Project ini telah diorganisasi ulang mengikuti prinsip **1 class = 1 file** untuk maintainability:

- **models/** → Neural network architectures (ActorCriticNetwork, HighLevelAgent)
- **agents/** → RL algorithms (PPO, MCTS)
- **common/** → Shared utilities (Memory, MCTSNode)
- **env/** → Environment + Physics (StabilityValidator, LBCPClusterer)

Backward compatibility dijaga 100%: folder `rl/`, `planning/`, dan `env/lbcp.py` hanya re-export dari lokasi baru.

👉 **Baca lebih lanjut:** Lihat section [Import Examples](#import-examples) di bawah.

## Alur Algoritma Utama

### 1. Hierarchical Control

Setiap item diproses dengan urutan berikut:

1. Amati state kontainer saat ini (height map + item saat ini).
2. High-level agent menghasilkan macro decision:
	- orientasi/arah strategi,
	- prioritas zona placement,
	- apakah repacking boleh dipicu.
3. Candidate generator membuat kandidat aksi berdasarkan macro decision.
4. Feasibility masking menghapus aksi tidak valid (out-of-bound, overflow, instabil).
5. Jika kandidat tersedia, low-level policy memilih aksi terbaik dari kandidat valid.
6. Jika kandidat kosong, masuk deadlock handler:
	- coba rearrangement/repacking,
	- jika masih gagal, fallback ke search MCTS aksi placement.
7. Environment diupdate, lanjut ke item berikutnya.

### 2. Low-Level Placement Policy (Actor-Critic)

Low-level policy berjalan dengan langkah:

1. Logits actor dimasking agar hanya aksi legal yang punya probabilitas.
2. Sampling aksi dilakukan dari distribusi masked policy.
3. Critic mengestimasi value state untuk update advantage.
4. PPO update dilakukan berkala dari trajectory yang terkumpul.

Intinya: actor memilih aksi placement, critic menilai kualitas state, dan PPO menjaga update tetap stabil.

### 3. Rearrangement dan Repacking (MCTS)

Saat deadlock terjadi, MCTS rearrangement dijalankan dengan 4 fase:

1. Selection: pilih node menggunakan UCB.
2. Expansion: buat child dari aksi unpack top-most k items.
3. Simulation: coba repack item yang di-unpack + item gagal, hitung reward (utilization + success).
4. Backpropagation: propagasi nilai reward sepanjang path tree.

Jika urutan terbaik valid, snapshot hasil bisa langsung diaplikasikan ke environment utama.

## Struktur Folder (1 Class = 1 File)

### 📌 Organized Folders (Implementasi Baru)

```
models/
├── actor_critic.py          # CNN-based Actor-Critic network
├── high_level_agent.py      # High-level strategy agent
└── low_level_agent.py       # Low-level placement agent (legacy reference)

agents/
├── ppo.py                   # PPO algorithm implementation
└── mcts.py                  # MCTS search algorithm

common/
├── memory.py                # Trajectory storage untuk PPO
└── mcts_node.py             # MCTS tree node structure

env/
├── stability_validator.py   # LBCP stability checking (NEW)
├── lbcp_clusterer.py        # Load-balanced clustering (NEW)
├── lbcp.py                  # Re-export layer (backward compat)
├── container_env.py         # Main container environment
├── height_map.py            # Height map management
├── action_mask.py           # Action masking utilities
└── ...
```

### 📌 Backward Compatibility Layers

```
rl/                          # Re-exports dari agents/ & models/
├── ppo.py                   # Re-exports PPO & ActorCriticNetwork
└── high_level_agent.py      # Re-exports HighLevelAgent

planning/
└── mcts.py                  # Re-exports MCTS & MCTSNode
```

### 📌 Utility Folders

```
dataset/                     # Item generation & datasets
utils/                       # Logging, metrics, helpers
tests/                       # Unit tests
```

## Konsep Reorganisasi

### RL Folder vs Agents Folder

- **`rl/`** = Backward compatibility layer (hanya re-exports)
- **`agents/`** = Actual implementation dengan logic lengkap

Analogi: `rl/` adalah "pintu ke toko baru", sedangkan `agents/` adalah "toko aslinya".

### LBCP: env/lbcp.py vs Modul Baru

- **`env/lbcp.py`** = Re-export layer
- **`env/stability_validator.py`** = `StabilityValidator` class (NEW)
- **`env/lbcp_clusterer.py`** = `LBCPClusterer` class (NEW)

Sebelumnya ada 2 class dalam 1 file → Sekarang masing-masing class di file sendiri.

## Import Examples

### ✅ Legacy Imports (Still Work)

Kode lama tetap berfungsi tanpa perubahan:

```python
from rl.ppo import PPO
from rl.high_level_agent import HighLevelAgent
from planning.mcts import MCTS, MCTSNode
from env.lbcp import StabilityValidator, LBCPClusterer
```

### ✨ New Recommended Imports

Untuk kode baru, gunakan direct imports:

```python
from agents.ppo import PPO
from agents.mcts import MCTS
from models.actor_critic import ActorCriticNetwork
from models.high_level_agent import HighLevelAgent
from common.mcts_node import MCTSNode
from common.memory import Memory
from env.stability_validator import StabilityValidator
from env.lbcp_clusterer import LBCPClusterer
```

### 📝 Catatan

- `train.py` dan `evaluate.py` tetap menggunakan legacy imports
- Auto-redirect ke implementasi baru terjadi secara transparan
- Backward compatibility dijamin 100% ✓

## Menjalankan dengan Make

Gunakan Makefile untuk perintah harian:

1. Lihat semua target:
	make help
2. Preview cleanup (dry-run):
	make clean
3. Terapkan cleanup:
	make clean-apply
4. Cleanup penuh + hapus folder output kosong:
	make clean-full
5. Jalankan unit test:
	make test
6. Smoke test evaluasi:
	make eval-smoke
7. Smoke test training:
	make train-smoke
8. Smoke test evaluasi (cutting stock):
	make eval-cutting-smoke
9. Smoke test training (cutting stock):
	make train-cutting-smoke
10. Jalankan semua smoke checks sekaligus:
	make all-smoke

## Cleanup Manual

Jika ingin langsung pakai script:

1. Preview:
	scripts/clean.sh
2. Apply:
	scripts/clean.sh --apply
3. Opsi tambahan:
	scripts/clean.sh --apply --no-outputs

---

## 🎓 Training Pipeline: What Gets Trained

### Overview

Sistem training hierarkis mengoptimalkan **tiga komponen utama secara paralel**:

1. **PPO Network** (Low-Level Policy) - ✅ TRAINED
2. **HighLevelAgent** (Strategy Selection) - ✅ TRAINED (NEW in v2.0)
3. **MCTS Search** (Deadlock Resolution) - Used as-is (no learning)

### Detailed Training Components

#### 1. PPO Network (Actor-Critic)

**Status**: ✅ Fully Trained

**What it does**:
- Takes current state (height map + item dimensions) as input
- Outputs action logits for all possible positions in container
- Selects placement position for current item

**Training via**:
- Trajectory collection from actual environment interactions
- GAE (Generalized Advantage Estimation) for return computation
- PPO update with clipped objectives for stable learning
- Entropy bonus to maintain exploration

**Configuration** (train.py):
```python
ppo = PPO(
    state_size=env.state_size,           # Height map + item dims
    action_size=env.action_size,         # L×W possible positions
    learning_rate=3e-4,
    gamma=0.99,                          # Discount factor
    gae_lambda=0.95,                     # GAE smoothing
    clip_ratio=0.2,                      # PPO clipping
    entropy_coef=0.01,                   # Exploration bonus
    ...
)
```

#### 2. HighLevelAgent (Strategy Selection)

**Status**: ✅ Now Trained (Task #1 completion)

**What it does**:
- Observes current container state and remaining items
- Selects one of 8 strategies:
  - 6 Orientations (left-to-right, right-to-left, front-to-back, etc.)
  - 1 Repack strategy (rearrange items if needed)
  - 1 No-op strategy
- Influences candidate generation and placement priority

**Training via** (NEW):
- Policy gradient loss: -log(π) × advantage
- Advantage comes from: `0.7 × episodic_reward + 0.3 × load_balance`
- Entropy bonus for exploration (coef=0.01)
- Gradient clipping (max_norm=1.0) for stability

**Training Loop Integration** (train.py):
```python
# In train_epoch():
episode_info, next_value, strategy_buffer = self.collect_steps()

# Update 1: Train HighLevelAgent
self._update_high_level_agent(strategy_buffer)

# Update 2: Train PPO
self.ppo.update(next_value=next_value, num_epochs=3)
```

**Key Features**:
- Strategy selection uses **sampling** during training (exploration)
- Strategy selection uses **greedy argmax** during evaluation (exploitation)
- Separate optimizer with lr=1e-4 (lower than PPO to prevent instability)
- Updated every epoch with 3 passes over strategy buffer

#### 3. MCTS (Deadlock Resolution)

**Status**: Used as-is (no parameter learning)

**What it does**:
- Activated only when system reaches deadlock
- Runs Monte Carlo Tree Search to find valid rearrangement
- Returns optimal sequence of items to unpack and repack

**Integration** (train.py):
```python
# When deadlock detected:
mcts = MCTS(self.env, budget=self.mcts_budget)
result = mcts.search_rearrangement(
    failed_item=current_item,
    max_unpack=3,
    apply_to_env=True
)
```

**Performance Tracking**:
- `deadlocks`: How many times deadlock occurred
- `rearrange_attempts`: How many rearrangement searches run
- `rearrange_success`: % of searches that found valid rearrangement
- `rearrange_applied`: % of searches applied to environment

### MCTS Enhanced: Active Blend with PPO

**Status**: ✅ Integrated (Task #2 completion)

**What's new**:
- MCTS can now influence normal action selection (not just deadlock recovery)
- During 20% of action selections, PPO blends with MCTS results
- When MCTS finds good action, system prefers it 70% of the time

**Code** (train.py - `_blend_ppo_mcts_decision()`):
```python
if np.random.random() < 0.2:  # 20% of time
    mcts_result = mcts.search(state, action_mask, depth_limit=5)
    if valid_action_found:
        if np.random.random() < 0.7:  # Use MCTS 70% of the time
            action = mcts_action
        else:
            action = ppo_action  # Blend: let PPO decide
```

**Metrics Tracked**:
- `mcts_active_used`: Count of times MCTS actively influenced decisions
- `mcts_fallback_used`: Count of deadlock fallback uses

### Training Statistics & Monitoring

**Per-Epoch Output** (printed during training):

```
Episode Reward:      X.XXXX ± X.XXXX    (Mean ± Std of last 100 episodes)
Episode Length:      X.X steps
Container Util:      XX.XX%             (Volume used)
Success Rate:        XX.X%              (Items placed / total items)
Deadlocks:           X                  (Deadlock occurrences)
Rearrange Attempts:  X
Rearrange Success:   XX.X%
Rearrange Applied:   XX.X%
MCTS Active Used:    X                  (Active blending count)
MCTS Fallback Used:  X                  (Deadlock recovery count)
```

### Training vs Evaluation Mode

**Training** (sample_strategy=True):
- HighLevelAgent: Samples strategies from distribution
- PPO: Maintains exploration via entropy bonus  
- MCTS: Actively blends with PPO (20% probability)
- Visualizations: Saved every 10 episodes

**Evaluation** (sample_strategy=False):
- HighLevelAgent: Deterministic greedy strategy selection
- PPO: Deterministic greedy action selection
- MCTS: Not used in normal flow (optional for comparison)
- Output: Metrics to CSV, visualizations on demand

### Troubleshooting Training

**If HighLevelAgent not learning**:
1. Check `train.py` line ~105: Ensure `.train()` mode (not `.eval()`)
2. Verify strategy_buffer is non-empty
3. Check optimizer step is called in `_update_high_level_agent()`

**If PPO performance plateaus**:
1. Adjust `entropy_coef` 0.01 → higher for more exploration
2. Try `clip_ratio` 0.2 → reduce to 0.1 for smaller updates
3. Check action masking is working: `effective_mask` should filter invalid positions

**If deadlocks too frequent**:
1. HighLevelAgent may not be learning good strategies
2. Try increasing MCTS budget: `self.mcts_budget = 20` → 30+
3. Ensure `candidate_generator` creates diverse candidates

### Policy Comparison & Evaluation

**Available Policies** (agents/oracle_policy.py):
1. ✅ **PPO** - Trained neural network policy
2. ✅ **HighLevelAgent+PPO** - Hierarchical control
3. 🔵 **Oracle LoadBalance** - Greedy heuristic (places to minimize CoG deviation)
4. 🔵 **Oracle Height** - First-fit-decreasing heuristic
5. 🔵 **Oracle Center** - Places items nearest to container center
6. ⚪ **Random** - Baseline (random valid placement)

**Run Comparison**:
```bash
python evaluate_comparison.py \
    --episodes 50 \
    --max-items 20 \
    --ppo-checkpoint models/ppo_final.pt \
    --high-level-checkpoint models/high_level_final.pt
```

**Output** (policy_comparison_results.csv):
- Utilization (mean ± std)
- Load balance coefficient
- Stability rate
- Success rate
- Total reward

---

## Summary: Training Flow

```
┌─────────────────────────────────────┐
│ TrainingLoop.train_epoch()          │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ collect_steps(n_steps)              │
│ - Interact with env                 │
│ - Collect PPO transitions           │
│ - Collect strategy info →buffer     │
└─────────────────────────────────────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
┌─────────┐  ┌──────────────┐
│ Update  │  │ Update       │
│ PPO     │  │ HighLevel    │
│ (3xGAE) │  │ Agent (3x)   │
└─────────┘  └──────────────┘
    ↓             ↓
    └──────┬──────┘
           ↓
    ┌──────────────┐
    │ Log Stats    │
    │ Save Viz     │
    │ Continue     │
    └──────────────┘
```

	scripts/clean.sh --apply --no-caches