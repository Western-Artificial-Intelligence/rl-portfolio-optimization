# Backtesting Documentation

> **For Western AI Developers** - How to evaluate the trained PPO agent.

---

## Quick Start

```bash
# Activate environment
.\.venv\Scripts\activate

# Run backtest (uses cached super-states, ~30 seconds)
python -m ppo.backtest_ppo
```

---

## What is Backtesting?

Backtesting evaluates a trading strategy on **historical data it has never seen**. We use the last 20% of our dataset (Aug 2023 → Dec 2024) as a holdout evaluation period.

```
Training Period (80%)         Evaluation Period (20%)
2018-01 ────────────────────► 2023-08 ─────────────────► 2024-12
     PPO learns here              PPO tested here (backtest)
```

---

## How Our Backtest Works

### 1. Load Trained Model

```python
from stable_baselines3 import PPO
model = PPO.load("checkpoints/ppo/ppo_final.zip")
```

### 2. Run on Evaluation Data

- Load cached super-states (`data/super_states.npz`) for fast inference
- Step through each trading day in the eval period
- Model outputs portfolio weights → Execute trades → Track value

### 3. Compute Metrics

| Metric | Formula | Goal |
|--------|---------|------|
| **Total Return** | `(Final - Initial) / Initial` | Higher is better |
| **Sharpe Ratio** | `mean(excess_returns) / std(returns) × √252` | Higher is better |
| **Max Drawdown** | `max((peak - trough) / peak)` | Lower is better |

### 4. Compare Against Baseline

We compare against **S&P 500 buy-and-hold** which represents a passive investment strategy.

---

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `checkpoints/ppo/ppo_final.zip` | Path to trained model |
| `--data` | `data/deepar_dataset.csv` | Dataset path |
| `--cache` | `data/super_states.npz` | Cached states path |
| `--output` | `results/` | Output directory |
| `--initial-cash` | `1,000,000` | Starting portfolio value |

---

## Output Files

| Path | Description |
|------|-------------|
| `results/backtest_metrics.json` | All metrics in JSON format |
| `results/backtest_equity_curve.png` | Portfolio value over time |
| `results/backtest_weights.png` | Weight allocations over time |

---

## Interpreting Results

### Good Results

- **Sharpe > 1.0**: Risk-adjusted returns are good
- **Total Return > Baseline**: PPO outperforms passive investing
- **Max Drawdown < 20%**: Capital protection is reasonable

### Warning Signs

- **Sharpe < 0**: Agent is losing money on average
- **Drawdown > 30%**: High risk of significant losses
- **Flat weights**: Agent isn't adapting to market conditions

---

## Monte Carlo Simulation (Advanced)

For more robust evaluation, run multiple backtests with noise:

```python
# Future implementation
results = []
for i in range(100):
    result = run_backtest_with_noise(model, eval_df, noise_scale=0.01)
    results.append(result["metrics"])

mean_sharpe = np.mean([r["sharpe_ratio"] for r in results])
std_sharpe = np.std([r["sharpe_ratio"] for r in results])
print(f"Sharpe: {mean_sharpe:.3f} ± {std_sharpe:.3f}")
```

---

## Related Documentation

- [PPO Training Documentation](PPODoc.md)
- [DeepAR Documentation](../deepAR/DeepARDoc.md)

---

*Last updated: January 2026 | Western AI Research Team*
