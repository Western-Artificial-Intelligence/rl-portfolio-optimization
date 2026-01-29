# ReST: Reinforcement Learning from Self-Training

This document describes the ReST (Reinforcement learning from Self-Training) implementation for portfolio optimization.

## Overview

ReST is a two-loop training framework that combines:
- **Outer Loop**: Trajectory generation, scoring, and elite filtering
- **Inner Loop**: PPO training on elite trajectories only

The key idea is to generate many trading trajectories, score them based on risk-adjusted returns, keep only the "elite" (best-performing) trajectories, and train the policy to reproduce those elite behaviors.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    train_ReST.py                              │
├───────────────────────────────────────────────────────────────┤
│  1. Load data, initialize PPO model, setup logging            │
│  2. Pre-select fixed eval windows (seed-controlled)           │
│                                                               │
│  FOR iteration = 1 to 10:                                     │
│    ┌──────────────────────────────────────────────────────┐   │
│    │  GENERATE PHASE                                      │   │
│    │  • Set seed = base_seed + iteration                  │   │
│    │  • Sample M=300 start indices from [0, N-T]          │   │
│    │  • Run policy stochastically for T=126 steps each    │   │
│    │  • Record (s, a, r, portfolio_values) per trajectory │   │
│    └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│    ┌──────────────────────────────────────────────────────┐   │
│    │  SCORE & FILTER PHASE                                │   │
│    │  • Compute R(τ) = Sharpe(τ) - λ·MDD(τ) for each      │   │
│    │  • Elite percentile: 25% (iter 1-2) → 10% (iter 3+)  │   │
│    │  • Filter top percentile (min 30 elites)             │   │
│    └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│    ┌──────────────────────────────────────────────────────┐   │
│    │  PPO TRAINING PHASE                                  │   │
│    │  • Wrap elite trajectories in EliteReplayEnv         │   │
│    │  • PPO.learn() on replay env (5-10 epochs)           │   │
│    │  • SB3 computes advantages/values normally           │   │
│    └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│    ┌──────────────────────────────────────────────────────┐   │
│    │  LOGGING PHASE                                       │   │
│    │  • Mean R(τ) all vs elite, gap Δ                     │   │
│    │  • Action entropy / weight variance                  │   │
│    │  • Histogram of elite start dates                    │   │
│    └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│    ┌──────────────────────────────────────────────────────┐   │
│    │  EVALUATION PHASE                                    │   │
│    │  • Mini-eval every iteration (30-50 fixed windows)   │   │
│    │  • Full-eval every 3 iterations (full test set)      │   │
│    └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│    ┌──────────────────────────────────────────────────────┐   │
│    │  CHECKPOINT PHASE                                    │   │
│    │  • Save rest_iter_XX.zip + metrics                   │   │
│    │  • If held-out improved → update best_model.zip      │   │
│    └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│    Check exit criteria → next iteration or stop               │
│                                                               │
│  3. Save final model + summary                                │
└───────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Trajectory Scoring

Each trajectory τ is scored using:

```
R(τ) = Sharpe(τ) - λ · MDD(τ)
```

Where:
- **Sharpe(τ)**: Annualized Sharpe ratio computed over the full trajectory's returns
- **MDD(τ)**: Maximum drawdown from the portfolio value path
- **λ**: Drawdown penalty weight (default: 5.0)

### 2. Elite Filtering (Progressive)

| Iteration | Percentile | Description |
|-----------|------------|-------------|
| 1-2 | Top 25% | Lenient filtering early on |
| 3+ | Top 10% | Stricter filtering as training progresses |

Minimum of 30 elite trajectories is enforced regardless of percentile.

### 3. Replay Environment

The `EliteReplayEnv` streams elite trajectories to SB3 PPO:
- Does NOT simulate markets
- Returns stored rewards from original rollouts
- Lets SB3 compute advantages/values normally
- Episodes end exactly where stored trajectory ends

---

## Hyperparameters

### ReST Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 10 | Maximum ReST outer loop iterations |
| `rollouts_per_iter` | 300 | Trajectories generated per iteration (M) |
| `episode_length` | 126 | Trading days per trajectory (~6 months) |
| `lambda_mdd` | 5.0 | Drawdown penalty weight |
| `min_elites` | 30 | Minimum elite trajectories to keep |
| `base_seed` | 42 | Base random seed for reproducibility |

### PPO Parameters

Same as `train_ppo.py` for consistency:
- Learning rate: 1e-4
- Network: [256, 256] MLP
- Epochs per iteration: 10

---

## How to Run ReST Training

### Prerequisites

Before running ReST training, ensure the following are in place:

#### 1. Environment Setup

```bash
# Navigate to project directory
cd "C:\Users\micky\Desktop\Alpha RL\rl-portfolio-optimization"

# Activate virtual environment (if using uv)
.venv\Scripts\activate

# Or use uv run prefix for commands
```

#### 2. Required Data Files

| File | Description | How to Get |
|------|-------------|------------|
| `data/deepar_dataset.csv` | Market data with prices | Download from Google Drive (see README) |
| `data/super_states.npz` | Pre-computed 64-dim observations | Run `precompute_states.py` (Step 3) |
| `checkpoints/deepar/deepar_best.pt` | Trained DeepAR model | Run `train_deepar.py` first |
| `data/FRED/*.csv` | FRED macro data (VIX, etc.) | Run `fred_data.py` |

#### 3. Pre-compute Super-States (Required)

ReST requires pre-computed super-states for fast training. This only needs to be done once:

```bash
# This uses GPU for DeepAR inference (if available)
python -m ppo.precompute_states

# Or with custom output path
python -m ppo.precompute_states --output data/super_states.npz
```

**Expected output:**
```
Pre-computing Super-States for PPO Training
============================================================
Loading data from: data/deepar_dataset.csv
  Total dates: 1741
  Valid dates (after context): 1681

Computing 1681 super-states...
(This may take 10-30 minutes depending on your CPU)

Saving to: data/super_states.npz
  Shape: (1681, 64)
  File size: X.XX MB
```

---

### Running ReST Training

#### Step-by-Step

```bash
# Step 1: Ensure you're in the project directory
cd "C:\Users\micky\Desktop\Alpha RL\rl-portfolio-optimization"

# Step 2: Activate environment
.venv\Scripts\activate

# Step 3: Run ReST training (default settings)
python -m ppo.train_ReST
```

#### Basic Command

```bash
python -m ppo.train_ReST
```

#### With Custom Parameters

```bash
python -m ppo.train_ReST ^
    --max-iterations 10 ^
    --rollouts 300 ^
    --episode-length 126 ^
    --lambda-mdd 5.0 ^
    --seed 42
```

> **Note:** On Windows PowerShell, use backtick (`) instead of caret (^) for line continuation, or put everything on one line.

#### Using uv run (Alternative)

```bash
uv run python -m ppo.train_ReST
```

---

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `data/deepar_dataset.csv` | Training data path |
| `--cache` | `data/super_states.npz` | Pre-computed states |
| `--checkpoint-dir` | `checkpoints/rest` | Model save directory |
| `--log-dir` | `logs/rest` | Log directory |
| `--max-iterations` | 10 | ReST iterations |
| `--rollouts` | 300 | Trajectories per iteration |
| `--episode-length` | 126 | Steps per trajectory (~6 months) |
| `--lambda-mdd` | 5.0 | Drawdown penalty weight |
| `--ppo-epochs` | 10 | PPO epochs per iteration |
| `--seed` | 42 | Random seed |

---

### Expected Training Output

```
======================================================================
ReST Training for Portfolio Optimization
======================================================================
Start time: 2026-01-29 10:00:00
Max iterations: 10
Rollouts per iteration: 300
Episode length: 126 days
Lambda (MDD penalty): 5.0
======================================================================

Loading training data from: data/deepar_dataset.csv
  Loaded 15,669 rows
  Securities: 9
  Date range: 2018-01-31 to 2024-12-31

Loading cached super-states from: data/super_states.npz
  Loaded 1681 pre-computed observations

Data split (train_ratio=0.8):
  Training: 1392 days (2018-01-31 to 2022-12-30)
  Test: 349 days (2023-01-03 to 2024-12-31)

======================================================================
ReST Iteration 1/10
======================================================================

[1/5] Generating 300 trajectories...
  Collecting trajectory 50/300...
  Collecting trajectory 100/300...
  ...
  Mean score: 0.523 +/- 1.245

[2/5] Filtering elite trajectories...
  Selected 75 elites (25% threshold)
  Elite mean score: 1.892
  Elite mean Sharpe: 2.134
  Elite mean MDD: 0.048

[3/5] Creating replay environment...
  Total elite steps: 9450

[4/5] Training PPO on elite trajectories...
  Completed PPO training (94500 timesteps)

[5/5] Evaluating and logging...
  Held-out evaluation (30 windows):
    Sharpe: 1.245
    Return: 12.34%
    Max DD: 8.56%

============================================================
ReST Iteration 1 Summary
============================================================
  Rollouts: 300 | Elites: 75 (25%)
  Score (all):   +0.523 ± 1.245
  Score (elite): +1.892 ± 0.567
  Gap Δ:         +1.369
  ...
============================================================
```

---

### Monitoring Training Progress

#### Real-time Console Output

The training script prints progress for each iteration including:
- Trajectory generation progress
- Elite filtering statistics
- PPO training completion
- Held-out evaluation metrics
- Failure warnings (if any)

#### Log Files

Check `logs/rest/` for detailed metrics:

```bash
# View latest iteration metrics
type logs\rest\rest_iter_01_metrics.json

# View complete training log (after completion)
type logs\rest\rest_training_log.json
```

#### Checkpoints

Models are saved to `checkpoints/rest/`:

```bash
# List saved models
dir checkpoints\rest\*.zip
```

---

### Troubleshooting

#### "Cache not found" Error

```
Loading cached super-states from: data/super_states.npz
  Cache not found: data/super_states.npz
```

**Solution:** Run pre-computation first:
```bash
python -m ppo.precompute_states
```

#### "DeepAR checkpoint not found" Error

**Solution:** Train DeepAR model first:
```bash
python -m deepAR.train_deepar
```

#### Low Score Gap (Δ ≈ 0)

If the score gap between elite and average trajectories is near zero:
- Increase rollouts: `--rollouts 500`
- Adjust lambda: `--lambda-mdd 3.0` or `--lambda-mdd 8.0`

#### Entropy Collapse Warning

If action entropy drops below 0.3:
- This indicates the policy is becoming too deterministic
- Consider increasing exploration in PPO (modify `ent_coef` in code)

#### Out of Memory

If you run out of memory:
- Reduce rollouts: `--rollouts 200`
- Reduce episode length: `--episode-length 63` (3 months)

---

### Quick Start Summary

```bash
# 1. Setup (one-time)
cd "C:\Users\micky\Desktop\Alpha RL\rl-portfolio-optimization"
.venv\Scripts\activate

# 2. Pre-compute states (one-time, ~10-30 min)
python -m ppo.precompute_states

# 3. Run ReST training
python -m ppo.train_ReST

# 4. Check results
dir checkpoints\rest\
type logs\rest\rest_training_log.json
```

---

## Output Structure

```
checkpoints/rest/
├── rest_iter_01.zip          # Model after iteration 1
├── rest_iter_01_metrics.json # Metrics for iteration 1
├── rest_iter_02.zip
├── rest_iter_02_metrics.json
├── ...
├── rest_iter_10.zip
├── rest_iter_10_metrics.json
├── rest_final.zip            # Final model
├── best_model.zip            # Best held-out performance
└── best_metrics.json         # Metrics for best model

logs/rest/
├── rest_iter_01_metrics.json
├── rest_iter_02_metrics.json
├── ...
└── rest_training_log.json    # Complete training log
```

---

## Diagnostics & Failure Detection

### Metrics Tracked Per Iteration

1. **Score Statistics**
   - Mean R(τ) over all rollouts
   - Mean R(τ) over elite set
   - Score gap: Δ = E[R(τ)|elite] - E[R(τ)]

2. **Policy Behavior**
   - Action entropy (exploration measure)
   - Weight variance across time

3. **Elite Distribution**
   - Histogram of elite start dates (regime coverage)

4. **Held-out Performance**
   - Sharpe ratio on test set
   - Total return
   - Maximum drawdown

### Failure Indicators

| Indicator | Symptom | Possible Fix |
|-----------|---------|--------------|
| Δ ≈ 0 | Score gap near zero | More rollouts or better reward function |
| Low entropy | Entropy < 0.3 | Increase exploration (higher ent_coef) |
| Static weights | Weight variance < 0.0005 | Agent not adapting to regimes |
| Elite clustering | Elites from single period | More rollouts or wider window sampling |

---

## Files

| File | Description |
|------|-------------|
| `ppo/train_ReST.py` | Main training script |
| `ppo/rest_trajectory.py` | Trajectory dataclass, collector, filter |
| `ppo/rest_utils.py` | Scoring functions, logging utilities |
| `ppo/windowed_env.py` | Windowed environment wrapper |
| `ppo/elite_replay_env.py` | Replay environment for PPO |

---

## Theory Reference

ReST is based on the principle of **reinforcement learning from synthetic data**. The algorithm:

1. Generates diverse trajectories using the current policy (exploration via stochastic sampling)
2. Evaluates trajectories using a reward function that balances return and risk
3. Filters to keep only "elite" (high-scoring) trajectories
4. Trains the policy via maximum likelihood on elite (state, action) pairs

This is similar to:
- Expert iteration (with self-generated experts)
- Reward-weighted regression
- Best-of-N sampling + fine-tuning

The key innovation is using the **same policy** to both generate and filter trajectories, creating a self-improvement loop.

---

## Constraints & Safety

### CPU vs GPU Usage

| Component | Device | Reason |
|-----------|--------|--------|
| **PPO/ReST Training** | CPU | Strict CPU-only for stability and consistency |
| **DeepAR Pre-computation**  | GPU (if available) | Pre-computed in `precompute_states.py`, cached to `.npz` |

PPO training uses `device="cpu"` explicitly. DeepAR inference (for super-state building) can use GPU when running `precompute_states.py`, but the cached states are loaded as numpy arrays for ReST training.

### Data Leakage Prevention

1. **DeepAR features**: Pre-computed with proper temporal filtering (data at time t uses only data ≤ t)
2. **Train/test split**: Strict chronological split (no future data in training)
3. **Evaluation windows**: Fixed seed ensures same windows across iterations

### Reproducibility

1. **Seed per iteration**: `seed = base_seed + iteration`
2. **Fixed eval windows**: Same test windows every evaluation
3. **Checkpointing**: Every iteration saved for debugging

### Interface Stability

1. **PortfolioEnv.step()**: Return signature unchanged
2. **Windowing**: Added via `reset(options=...)` only
3. **PPO model**: Standard SB3 interface preserved
