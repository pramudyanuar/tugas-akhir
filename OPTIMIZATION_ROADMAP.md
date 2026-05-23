# Performance Optimization Roadmap

**Status:** ✅ Phase 1 (LBCP + A3C + HighLevelAgent + MCTS) complete  
**Current Speedup:** 2-5x per step (with recommended config)  
**Remaining Opportunities:** 1.5-3x more possible

---

## Phase 2: Additional Optimizations (Prioritized by Impact)

### ⭐ Priority 1: State Computation (Est. 1.5-2x)

**Problem:** State creation in `container_env._get_state_and_mask()` does repeated flatten/concatenate operations per step.

**Solutions:**

1. **Reuse state buffer (pre-allocated numpy array)**
   - Pre-allocate `state_buffer = np.zeros((state_size,), dtype=np.float32)` in `__init__`
   - Update in-place instead of flatten+concat every step
   - Impact: ~20-30% faster state creation

2. **Use float32 instead of float64**
   - Change `self.height_map = np.zeros(..., dtype=np.float32)`
   - Reduces memory bandwidth 2x, speeds up operations
   - Impact: ~15% faster overall (especially on non-GPU)

3. **Vectorize state normalization**
   - Current: `(height_map - min) / (max - min)` in Python
   - Better: Use numpy broadcasting: `(hm - np.min(hm)) / (np.max(hm) - np.min(hm) + 1e-6)`
   - Impact: negligible overhead

**Implementation effort:** 30 min, low risk, backward compatible

---

### ⭐ Priority 2: Candidate Sorting Cache (Est. 1.2-1.5x)

**Problem:** `CandidateGenerator.generate_from_macro()` resorts candidates every step even if mask unchanged.

**Solutions:**

1. **Cache sorted candidates by action_mask hash**
   - Store `(mask_hash → sorted_candidates)` LRU cache (256 entries)
   - Key: `hash(action_mask)`
   - Reuse if mask identical
   - Impact: ~50% of calls cache-hit in typical episodes

2. **Early filtering before sort**
   - Apply heuristic pre-filter (e.g., "keep only central 50%") before sort
   - Reduces sort input size
   - Impact: ~10-15% faster sort

**Implementation effort:** 15 min, low risk

**Code location:** `src/core/candidate_generator.py`, method `generate_from_macro()`

---

### ⭐ Priority 3: Batch Reward Calculation (Est. 1.1-1.3x if batch-train used)

**Problem:** Reward calculated per-step scalar in `container_env.step()`. Can vectorize if doing batch training.

**Solutions:**

1. **Pre-compute reward weights**
   - Store `volume_coef=8.0, util_coef=2.0, height_coef=1.0, step_penalty, skip_penalty` as attributes
   - Avoid repeated constant computation

2. **Vectorize for batch (optional, later)**
   - If batch training added: compute rewards for multiple (state, action) pairs at once
   - Impact: 10-20% faster batch updates

**Implementation effort:** 5 min (quick fix), 1-2 hours (full batch)

---

### Priority 4: Visualization/Logging Optimization (Est. 1.1-1.5x)

**Problem:** Visualization writes to disk synchronously per episode. Logging every N steps adds overhead.

**Solutions:**

1. **Async visualization (low-hanging fruit)**
   - Write visualization in background thread
   - Non-blocking after episode end
   - Impact: eliminates 50-100 ms per visualization (if vis_interval=1)

2. **Reduce logging frequency**
   - Current `progress_interval=25` (logs every 25 steps)
   - Suggest `progress_interval=50-100` for faster runs
   - Impact: ~10% less I/O overhead

3. **Batch visualization saves**
   - Store 5-10 episodes in memory, write in batch
   - Impact: ~30% faster disk I/O

**Implementation effort:** 20-30 min (async), 5 min (reduce logging)

---

### Priority 5: Algorithm-Level (Est. 1.3-2x, high complexity)

**If you want to go deeper:**

1. **Action space reduction**
   - Current: L×W = 59×23 = 1357 positions
   - Reduce to key positions only (e.g., 256 key points via K-means)
   - Requires retraining, ~2x fewer actions
   - Impact: 2x faster action selection (fewer candidates to evaluate)
   - Risk: Quality loss (medium-high)

2. **Numba JIT for hot loops**
   - Compile `update_feasibility_map()`, `compute_support_cells()` with `@njit`
   - Impact: 5-10x speedup for those functions
   - Impact: ~5-15% overall (if those are bottlenecks)
   - Effort: 30-60 min

3. **GPU acceleration**
   - Move feasibility_map updates to GPU
   - Batch height_map operations on GPU
   - Impact: 2-5x (but high setup cost)
   - Effort: 2-4 hours, requires CUDA/GPU setup

---

## Recommended Next Steps

### Quick wins (15-30 min, 1.3-1.5x total speedup):
1. Pre-allocate state buffer (float32) in container_env
2. Add candidate sorting cache in candidate_generator
3. Reduce logging frequency (progress_interval → 50)

### Medium effort (1-2 hours, +0.5-1x speedup):
4. Async visualization writes
5. Batch reward computation
6. Numba JIT for stability functions (optional)

### Only if needed:
7. Action space reduction (requires retraining)
8. GPU acceleration (requires infrastructure)

---

## Testing Strategy

After each optimization, measure:
- **Throughput:** steps/second (use `timing['env_step_s']` from train logs)
- **Quality:** episode reward, packing utilization (should stay same)
- **Memory:** peak RSS (should not increase)

```bash
# Profile before/after with single epoch:
python scripts/train.py --n_steps 512 --epochs 1 --mcts_budget 0 --no_mcts_prob 1.0
```

Compare `timing['env_step_s']` / 512 = avg ms/step

---

## Current Speedup Summary

| Phase | Optimization | Speedup | Status |
|-------|-------------|---------|--------|
| 1a | LBCP cache + parallel | 1.2-3x | ✅ Done |
| 1b | A3C forward cache | 1.2-1.5x | ✅ Done |
| 1c | HighLevelAgent cache | 1.1-1.2x | ✅ Done |
| 1d | MCTS early termination | 1.2-1.4x | ✅ Done |
| **Phase 1 Total** | **Combined** | **2-5x** | ✅ Done |
| 2a | State buffer + float32 | 1.2-1.5x | ⏳ Recommended |
| 2b | Candidate cache | 1.1-1.3x | ⏳ Recommended |
| 2c | Async visualization | 1.05-1.2x | ⏳ Optional |
| **Phase 2 Total** | **Combined** | **1.5-3x** | ⏳ Pending |
| **Grand Total (1+2)** | **All optimizations** | **~4-12x** | Potential |

---

## Memory Footprint

Current caches use:
- LBCP: ~10 MB (4096 entries)
- A3C: ~10 MB (512 entries)
- HighLevelAgent: ~5 MB (256 + 128 entries)
- Candidate (if added): ~5 MB
- **Total:** ~30 MB (acceptable on 8GB+ systems)

---

**Recommendation:** Start with Phase 2a (state buffer). Low effort, high impact, safe.
