# Next Steps - PPO Portfolio Optimization

> **For Western AI Developers** - What to do after initial backtest results.

---

## Current Status ✅

- PPO agent trained on 80% of data (2018 → Aug 2023)
- Backtest completed on 20% holdout (Aug 2023 → Dec 2024)
- **Result**: +45.9% return vs S&P 500's +31.0%

---

## Priority 1: Add Cash Position 💰

### Why Cash Matters

Currently the agent trades 9 assets and is **always 100% invested**. This is unrealistic because:

| Problem | Impact |
|---------|--------|
| Can't de-risk | Agent holds stocks during crashes |
| Forced to allocate | Must put money somewhere even when uncertain |
| Higher drawdowns | Our 13% drawdown could be reduced |

### Implementation Steps

1. **Modify `PortfolioEnv`**:
   - Add "CASH" as the 10th asset with constant price of $1.00
   - Update action space: `shape=(10,)` instead of `(9,)`

2. **Update Super-State**:
   - Keep 64-dim observation (don't need price features for cash)
   - Or add a cash weight feature

3. **Re-run `precompute_states.py`**:
   - Cache needs to be regenerated

4. **Retrain PPO**:
   - Model needs to learn when to hold cash
   - May require tuning exploration (`ent_coef`)

5. **Backtest again** and compare:
   - Expected: Lower drawdown, possibly better Sharpe

---

## Priority 2: Monte Carlo Simulation 🎲

### What is Monte Carlo?

Instead of running **one** backtest, run **many** backtests with small random variations to measure robustness.

### How It Works for Our Agent

```
┌─────────────────────────────────────────────────────────────┐
│                    MONTE CARLO FRAMEWORK                    │
│                                                             │
│  For i = 1 to N (e.g., 100 runs):                          │
│    1. Add small noise to observations (±1-2%)              │
│    2. Add execution noise (slippage variation)             │
│    3. Run full backtest                                     │
│    4. Record: Sharpe, Return, Drawdown                      │
│                                                             │
│  Output: Mean ± Std for each metric                         │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Sketch

```python
def monte_carlo_backtest(model, price_df, eval_dates, n_runs=100, noise_scale=0.01):
    results = []
    
    for i in range(n_runs):
        # Add observation noise
        noisy_states = cached_states.copy()
        noisy_states['states'] += np.random.normal(0, noise_scale, noisy_states['states'].shape)
        
        # Add slippage noise (±5 bps variation)
        slippage = 5 + np.random.randint(-3, 4)  # 2-8 bps
        
        # Run backtest
        result = run_ppo_backtest(model, price_df, eval_dates, noisy_states)
        results.append(result['metrics'])
    
    # Aggregate
    sharpes = [r['sharpe_ratio'] for r in results]
    returns = [r['total_return'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    
    return {
        'sharpe': f"{np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}",
        'return': f"{np.mean(returns)*100:.1f}% ± {np.std(returns)*100:.1f}%",
        'drawdown': f"{np.mean(drawdowns)*100:.1f}% ± {np.std(drawdowns)*100:.1f}%",
    }
```

### Expected Output

```
Monte Carlo Results (100 runs):
  Sharpe Ratio: 1.56 ± 0.12
  Total Return: 45.9% ± 3.2%
  Max Drawdown: 13.0% ± 1.8%
```

### Why This Matters

- **Confidence intervals** show how sensitive results are
- **Low variance** = robust strategy
- **High variance** = may be overfitting to specific price path

---

## Priority 3: Multi-Regime Testing 📊

Test the agent on different market conditions:

| Regime | Date Range | Characteristic |
|--------|------------|----------------|
| COVID Crash | Feb-Mar 2020 | Sharp drawdown |
| 2022 Bear Market | Jan-Oct 2022 | Prolonged decline |
| 2023-24 Bull Run | Jan 2023 → now | Strong recovery |

This reveals if the agent adapts or fails in specific conditions.

---

## Stretch Goal: Temporal Portfolio Graph (TPG)

From the original paper proposal - add a GNN module to capture stock correlations:

```
PPO + DeepAR + FRED + TPG = "Super Agent"
```

This would model relationships like:
- AAPL ↔ MSFT (tech correlation)
- SPX ↔ individual stocks (market factor)
- VIX ↔ PSQ (inverse volatility play)

---

## Quick Reference: Commands

```bash
# Re-precompute states (after modifying env)
python -m ppo.precompute_states

# Retrain with new action space
python -m ppo.train_ppo --use-cache --timesteps 100000

# Run backtest
python -m ppo.backtest_ppo

# Future: Monte Carlo (not yet implemented)
python -m ppo.monte_carlo_backtest --runs 100
```

---

*Last updated: January 2026 | Western AI Research Team*
