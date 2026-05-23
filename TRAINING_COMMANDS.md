# Training Commands - 3 Options

## Option 1: QUICK TEST (5-10 min) - Verify Optimizations Work
```bash
python scripts/train.py \
  --num_epochs 1 \
  --n_steps 512 \
  --device cuda \
  --batched-envs 2 \
  --rollout-steps 16 \
  --fast-stability-mask \
  --candidate-top-k 64 \
  --mcts-budget 5 \
  --use-mcts-prob 0.05 \
  --vis-interval 1000
```

**What it does:**
- 1 epoch only = ~5-10 min on T4 GPU
- Reduced batch size (2 envs) to save memory
- Short rollout (16 steps) for quick iterations
- All optimizations ACTIVE
- Expected: ~1000-2000 steps/min (vs ~200 originally)

**Expected output:**
```
Collect rollout... |████] | steps: 512 | time: 0.25s | speed: 2048 steps/sec
Episode 1: reward=123.4, util=85%, time=0.25s
```

---

## Option 2: MEDIUM RUN (30-60 min) - Quick Benchmark
```bash
python scripts/train.py \
  --num_epochs 5 \
  --n_steps 2048 \
  --device cuda \
  --batched-envs 4 \
  --rollout-steps 32 \
  --fast-stability-mask \
  --candidate-top-k 128 \
  --mcts-budget 10 \
  --use-mcts-prob 0.05 \
  --vis-interval 100
```

**What it does:**
- 5 epochs = ~30-60 min on T4
- Full batch (4 envs)
- Balanced optimizations
- Reasonable benchmark

---

## Option 3: PRODUCTION FULL (8-24 hours) - Final Training
```bash
python scripts/train.py \
  --num_epochs 500 \
  --n_steps 2048 \
  --device cuda \
  --batched-envs 4 \
  --rollout-steps 32 \
  --fast-stability-mask \
  --candidate-top-k 256 \
  --mcts-budget 15 \
  --use-mcts-prob 0.1 \
  --vis-interval 50 \
  --checkpoint-steps 50000
```

**What it does:**
- 500 epochs = full training (~1M steps)
- Maximum batch size and configurations
- Save checkpoints every 50k steps
- Visualization every 50 episodes
- Expected: 3-5x speedup vs original = 8-24 hours (vs 3-5 days)

---

## Comparison: Speedup & Quality

| Metric | Original (Phase 0) | Phase 1 Only | Phase 1+2 (Current) |
|--------|-------------------|-------------|-------------------|
| Steps/sec | ~100-150 | 300-500 | 500-1000+ |
| Steps/hour | 360k-540k | 1.08M-1.8M | 1.8M-3.6M |
| 1M steps time | 2-3 hours | 33-55 min | 17-33 min |
| Epoch time (2048 steps) | ~20-30 sec | 4-7 sec | 2-4 sec |

---

## Recommendations

### For Quick Verification:
Run **Option 1** (1 epoch):
```bash
python scripts/train.py --num_epochs 1 --n_steps 512 --device cuda --batched-envs 2 --rollout-steps 16 --fast-stability-mask --candidate-top-k 64 --mcts-budget 5 --use-mcts-prob 0.05
```

Then check logs for speed improvement.

### For Kaggle Kernel:
- **Best:** Option 2 (5 epochs, ~60 min) fits typical Kaggle limit
- **Max:** Option 3 will need multi-session checkpoints

### For Local GPU:
- **Recommended:** Option 3 (full 500 epochs)
- Expected total time: 8-24 hours (vs 3-7 days originally)

---

## What to Monitor

After running, check `/logs/training/` for:
1. **Throughput:** `env_step_s` should be 2-5x lower (faster)
2. **Quality:** Episode rewards should stay same/better
3. **Memory:** Peak RSS should stay <4GB
4. **Success:** Higher utilization % per episode

Example good output:
```
Epoch 1: steps=2048, episodes=16, reward_mean=1050.5, util=82%, speed=512 steps/sec
Epoch 2: steps=2048, episodes=15, reward_mean=1125.3, util=85%, speed=498 steps/sec
...
```

---

## Next Steps After Testing

1. ✅ Run Option 1 to verify all optimizations work
2. ⏳ If speedup observed (2-5x), commit to GitHub
3. ⏳ Run Option 3 for final training (can run in background)
