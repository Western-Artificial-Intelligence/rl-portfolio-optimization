# PPO Training Documentation

> **For Western AI Developers** - How our PPO agent for portfolio optimization works.

---

## Table of Contents

1. [What is PPO?](#what-is-ppo)
2. [Our Architecture](#our-architecture)
3. [Super-State Observation](#super-state-observation)
4. [Training Pipeline](#training-pipeline)
5. [Hyperparameters](#hyperparameters)
6. [Reward Functions](#reward-functions)
7. [Running Training](#running-training)
8. [Troubleshooting](#troubleshooting)

---

## What is PPO?

**Proximal Policy Optimization (PPO)** is a reinforcement learning algorithm that learns through trial and error.

### Core Concepts

| Term | Definition |
|------|------------|
| **Agent** | The neural network that makes portfolio decisions |
| **Environment** | Simulated stock market (`PortfolioEnv`) |
| **Observation** | What the agent "sees" (64-dim Super-State) |
| **Action** | Portfolio weights for each asset |
| **Reward** | Risk-adjusted return (Sharpe Ratio) |
| **Episode** | One complete run through the training data |

### Training Loop

```
1. Agent observes market state (Super-State)
2. Agent outputs portfolio weights [AAPL: 0.3, MSFT: 0.2, ...]
3. Environment simulates next trading day
4. Agent receives reward based on performance
5. Agent updates its policy to improve future decisions
6. Repeat for millions of timesteps
```

### Why PPO?

- **Stable**: Prevents large policy updates that break training
- **Sample Efficient**: Reuses data within each update
- **Continuous Actions**: Handles portfolio weights (0-1 range)

---

## Our Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO AGENT (Brain)                        │
│              Neural Network: [256, 256] MLP                 │
│         Input: 64-dim observation → Output: 9 weights       │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Observation (64-dim)
┌─────────────────────────────────────────────────────────────┐
│                    SUPER-STATE BUILDER                      │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   DeepAR    │  │  FRED API   │  │  FinBERT    │         │
│  │ (Forecasts) │  │ (Macro Data)│  │ (Sentiment) │         │
│  │  54 dims    │  │   6 dims    │  │   4 dims    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Market Data
┌─────────────────────────────────────────────────────────────┐
│                    PORTFOLIO ENV                            │
│         Simulates trading with transaction costs            │
│         Computes rewards (Sharpe, returns, etc.)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Super-State Observation

The agent sees a **64-dimensional vector** at each timestep:

### Feature Breakdown

| Index | Features | Count | Source |
|-------|----------|-------|--------|
| 0-53 | DeepAR forecasts per stock | 54 | DeepAR model |
| 54-59 | Macro indicators | 6 | FRED API |
| 60-63 | Sentiment (placeholder) | 4 | FinBERT (TBD) |

### DeepAR Features (per stock × 9 stocks = 54)

| Feature | Description |
|---------|-------------|
| `forecast_mean` | Expected next-day return |
| `forecast_std` | Uncertainty in forecast |
| `forecast_skew` | Asymmetry of return distribution |
| `confidence` | Model confidence (0-1) |
| `quantile_10` | 10th percentile (downside risk) |
| `quantile_90` | 90th percentile (upside potential) |

### FRED Macro Features (6)

| Feature | Description |
|---------|-------------|
| `VIX` | Volatility index (market fear) |
| `VIX_regime` | Regime: Low/Normal/High/Extreme |
| `T10Y2Y` | Yield curve spread |
| `yield_inverted` | Is yield curve inverted? (recession signal) |
| `FEDFUNDS` | Federal funds rate |
| `fed_direction` | Is Fed hiking/cutting/stable? |

### Normalization

All features are normalized to **[-1, 1]** range for stable training.

---

## Training Pipeline

### Files

| File | Purpose |
|------|---------|
| `ppo/train_ppo.py` | Main training script |
| `ppo/super_state.py` | Builds 64-dim observations |
| `backtesting/core/PortfolioEnv.py` | Trading environment |

### Data Flow

```python
# 1. Load market data
df = pd.read_csv("data/deepar_dataset.csv")

# 2. Create environment
env = PortfolioEnv(df, use_super_state=True, reward_mode="sharpe")

# 3. Initialize PPO agent
model = PPO("MlpPolicy", env, learning_rate=1e-4, ...)

# 4. Train
model.learn(total_timesteps=1_000_000)

# 5. Save
model.save("checkpoints/ppo/ppo_final.zip")
```

### Train/Eval Split

- **Training**: First 80% of dates (2018-01-31 to 2023-08-11)
- **Evaluation**: Last 20% of dates (2023-08-14 to 2024-12-31)

---

## Hyperparameters

Our PPO is tuned for financial time-series:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 1e-4 | Conservative for noisy data |
| `n_steps` | 2048 | Steps per policy update |
| `batch_size` | 64 | Smaller batches for noise |
| `n_epochs` | 10 | Optimization passes per update |
| `gamma` | 0.99 | High discount = long-term focus |
| `gae_lambda` | 0.95 | Advantage estimation trade-off |
| `clip_range` | 0.2 | Policy change limit per update |
| `ent_coef` | 0.01 | Exploration encouragement |
| `net_arch` | [256, 256] | Network layers |

### Tuning Guidelines

- **Training unstable?** → Lower `learning_rate`, `clip_range`
- **Agent explores too little?** → Increase `ent_coef`
- **Short-term focus?** → Increase `gamma` toward 0.999
- **Overfitting?** → Reduce `n_epochs`, add dropout

---

## Reward Functions

The environment supports multiple reward modes:

| Mode | Formula | Best For |
|------|---------|----------|
| `sharpe` | `(mean_return - rf) / std * sqrt(252)` | Risk-adjusted performance |
| `simple_return` | `(new_value - old_value) / old_value` | Maximum profit |
| `log_return` | `log(new_value / old_value)` | Penalizes large losses |
| `risk_adjusted` | `excess_return / volatility` | Per-step risk adjustment |

### Why Sharpe Ratio?

```
Sharpe Ratio = (Returns - Risk-Free Rate) / Volatility
```

- Prevents agent from taking excessive risks
- Encourages consistent, smooth returns
- Industry-standard metric for portfolio performance

---

## Running Training

### ⚡ Fast Training with Cached Super-States (Recommended)

Pre-computing super-states gives **100x speedup** by avoiding repeated DeepAR inference.

```bash
# Step 1: Pre-compute super-states (one-time, ~15-30 min)
python -m ppo.precompute_states

# Step 2: Train with cache (fast!)
python -m ppo.train_ppo --use-cache --timesteps 50000 --checkpoint-freq 10000
```

### Why Cache?

| Mode | Time per step | 50k steps takes |
|------|---------------|-----------------|
| **Without cache** | ~1-2 seconds | ~15-30 hours |
| **With cache** | ~0.01 seconds | ~5 minutes |

### Quick Start (Without Cache - Slow)

```bash
# Activate environment
.\.venv\Scripts\activate

# Quick test (50k steps, will take hours without cache!)
python -m ppo.train_ppo --timesteps 50000 --checkpoint-freq 10000

# Full training (1M steps)
python -m ppo.train_ppo --timesteps 1000000
```

### Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | 1,000,000 | Total training steps |
| `--checkpoint-freq` | 100,000 | Save every N steps |
| `--data` | `data/deepar_dataset.csv` | Training data path |
| `--resume` | None | Resume from checkpoint |
| `--use-cache` | False | Use pre-computed super-states |
| `--cache-path` | `data/super_states.npz` | Path to cache file |

### Output Files

| Path | Contents |
|------|----------|
| `checkpoints/ppo/ppo_final.zip` | Final trained model |
| `checkpoints/ppo/ppo_checkpoint_*.zip` | Periodic checkpoints |
| `checkpoints/ppo/best_model.zip` | Best eval performance |
| `data/super_states.npz` | Pre-computed observations |

---

## Troubleshooting

### NaN Values in Training

**Error**: `ValueError: found invalid values: tensor([[nan, nan, ...]])`

**Cause**: Observations contain NaN/Inf from DeepAR forecasts

**Fix**: Already handled in `super_state.py`:
```python
super_state = np.nan_to_num(super_state, nan=0.0, posinf=1.0, neginf=-1.0)
```

### Training Too Slow

**Symptom**: ~1 second per step

**Cause**: DeepAR inference runs every step

**Fix**: Pre-compute super-states (see optimization section)

### TensorBoard Errors on OneDrive

**Error**: `FailedPreconditionError: ... is not a directory`

**Cause**: OneDrive sync conflicts with TensorFlow

**Fix**: TensorBoard is disabled in current version

---

## Future Improvements

1. **Pre-computed Super-States** - Cache observations for 100x faster training
2. **FinBERT Integration** - Replace sentiment placeholders with real NLP
3. **GPU Training** - Move from CPU to CUDA for faster model updates
4. **Multiple Reward Experiments** - Compare Sharpe vs pure returns

---

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [DeepAR Paper](https://arxiv.org/abs/1704.04110) - Amazon, 2017
- [ReST Paper](https://arxiv.org/abs/2308.08998) - Our training methodology inspiration

---

*Last updated: January 2026 | Western AI Research Team*
