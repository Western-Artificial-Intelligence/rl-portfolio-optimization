# DeepAR Documentation

> **For Western AI Developers** - How our DeepAR probabilistic forecasting model works.

---

## Table of Contents

1. [What is DeepAR?](#what-is-deepar)
2. [Architecture](#architecture)
3. [Outputs & Features](#outputs--features)
4. [Training Pipeline](#training-pipeline)
5. [Integration with PPO](#integration-with-ppo)
6. [Next Steps: Implied Volatility](#next-steps-implied-volatility)

---

## What is DeepAR?

**DeepAR** is a probabilistic forecasting model developed by Amazon. Unlike traditional forecasting that predicts a single value ("stock will be $150"), DeepAR predicts a **probability distribution** ("90% chance between $145-$155").

### Why Probabilistic Forecasting?

| Traditional Forecast | DeepAR Forecast |
|---------------------|-----------------|
| "Price = $150" | "Price ~ N(μ=$150, σ=$5)" |
| No uncertainty info | Tells us confidence level |
| Agent can't assess risk | Agent knows when to be cautious |

### Key Benefits for RL

1. **Uncertainty-Aware**: PPO agent can take smaller positions when uncertainty (σ) is high
2. **Risk Management**: Quantiles (10th, 90th percentile) show tail risk
3. **Regime Detection**: High variance indicates volatile market conditions

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     DeepAR Model                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Input: [60 days of log returns] + [series embedding]      │
│                          ↓                                   │
│   ┌────────────────────────────────────────────────────┐    │
│   │        LSTM (2 layers, 64 hidden units)            │    │
│   │              with dropout (0.1)                     │    │
│   └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│   ┌────────────────────────────────────────────────────┐    │
│   │            Gaussian Output Layer                    │    │
│   │     Outputs: μ (mean) and σ (std deviation)        │    │
│   └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│   Output: N(μ, σ²) - Gaussian distribution of returns       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | Purpose |
|-----------|---------|
| **Series Embedding** | Learns unique patterns for each stock (AAPL, MSFT, etc.) |
| **LSTM Encoder** | Captures temporal patterns in price history |
| **Gaussian Output** | Maps hidden state to distribution parameters |

### Configuration

```python
DeepARModel(
    input_size=1,        # Univariate (log returns)
    hidden_size=64,      # LSTM hidden dimension
    num_layers=2,        # LSTM depth
    dropout=0.1,         # Regularization
    embedding_dim=16,    # Series embedding size
    num_series=9,        # Number of tracked stocks
)
```

---

## Outputs & Features

DeepAR provides **6 features per stock** for the PPO agent:

| Feature | Description | Range |
|---------|-------------|-------|
| `forecast_mean` | Expected next-day return | [-0.1, 0.1] |
| `forecast_std` | Uncertainty in forecast | [0, 0.1] |
| `forecast_skew` | Asymmetry of distribution | [-2, 2] |
| `confidence` | Model confidence score | [0, 1] |
| `quantile_10` | 10th percentile (downside) | [-0.15, 0.05] |
| `quantile_90` | 90th percentile (upside) | [-0.05, 0.15] |

### Interpretation

```
High forecast_mean + Low forecast_std = Strong bullish signal
High forecast_mean + High forecast_std = Risky bullish bet
Low forecast_mean + Low forecast_std = Clear bearish signal
Wide (q90 - q10) spread = High uncertainty
```

---

## Training Pipeline

### Files

| File | Purpose |
|------|---------|
| `deepAR/model.py` | Model architecture (DeepARModel, DeepARForecaster) |
| `deepAR/train_deepar.py` | Training script |
| `deepAR/preprocessing.py` | Data utilities |

### Training Command

```bash
uv run python deepAR/train_deepar.py --epochs 30
```

### Loss Function

DeepAR uses **Negative Log-Likelihood (NLL)** loss:

```
L = -log P(y | μ, σ) = 0.5 * log(2π) + log(σ) + (y - μ)² / (2σ²)
```

This trains the model to:
1. Predict accurate means (μ close to actual)
2. Calibrate uncertainty (σ reflects true variance)

### Checkpoints

| File | Contents |
|------|----------|
| `checkpoints/deepar/deepar_best.pt` | Best validation model weights |
| `checkpoints/deepar/training_summary.json` | Training config and metrics |
| `checkpoints/deepar/series_to_idx.json` | Stock → index mapping |

---

## Integration with PPO

DeepAR is NOT trained end-to-end with PPO. Instead:

1. **DeepAR trains separately** on historical returns
2. **DeepAR is frozen** during PPO training
3. **PPO uses DeepAR forecasts** as part of its observation

```
Historical Data
      ↓
┌─────────────────┐
│ DeepAR (frozen) │ ← Pre-trained, not updated during PPO
└─────────────────┘
      ↓
   Forecasts
      ↓
┌─────────────────┐
│ SuperStateBuilder │ ← Combines DeepAR + FRED + Sentiment
└─────────────────┘
      ↓
 64-dim observation
      ↓
┌─────────────────┐
│   PPO Agent     │ ← Learning portfolio weights
└─────────────────┘
```

---

## Next Steps: Implied Volatility

### Current Limitation

DeepAR currently uses only **historical log returns** as input. This captures past behavior but misses forward-looking market expectations.

### The Opportunity

**Implied Volatility (IV)** from options prices captures **what the market EXPECTS** volatility to be. This is forward-looking information that DeepAR doesn't currently have.

### Bloomberg Terminal Integration

We can extract IV data from Bloomberg using the following approach:

#### Step 1: Data Extraction

```python
# Bloomberg API pseudocode
from blpapi import Session

def get_implied_volatility(ticker: str) -> dict:
    """
    Pull implied volatility metrics from Bloomberg.
    
    Key fields to extract:
    - IVOL_DELTA_25_CALL_3M: 3-month 25-delta call IV
    - IVOL_DELTA_25_PUT_3M: 3-month 25-delta put IV
    - IVOL_ATM_3M: At-the-money 3-month IV
    - SKEW_3M: Put-call IV skew (fear indicator)
    """
    session.sendRequest({
        "securities": [f"{ticker} US Equity"],
        "fields": ["IVOL_ATM_3M", "IVOL_DELTA_25_PUT_3M", "IVOL_DELTA_25_CALL_3M"]
    })
    return response
```

#### Step 2: Feature Engineering

| IV Feature | Meaning | Signal |
|------------|---------|--------|
| `ATM_IV` | Overall expected volatility | High = caution |
| `IV_skew` | Put IV - Call IV | High = fear, bearish |
| `IV_term_structure` | 3M IV - 1M IV | High = expecting future volatility |
| `IV_rank` | IV percentile vs. 52-week range | High = expensive options |

#### Step 3: DeepAR Integration

Update DeepAR to accept IV as a **dynamic feature**:

```python
# Current: Only log returns
DeepARModel(input_size=1, num_dynamic_features=0)

# Future: Log returns + IV features
DeepARModel(input_size=1, num_dynamic_features=4)  # 4 IV features
```

#### Implementation Tasks

1. **Create Bloomberg extractor** (`deepAR/bloomberg-data-extraction/iv_extractor.py`)
2. **Update preprocessing** to include IV in training data
3. **Retrain DeepAR** with IV as dynamic features
4. **Update SuperStateBuilder** to expect new forecast format

### Expected Benefits

| Without IV | With IV |
|------------|---------|
| React to volatility after it happens | Anticipate volatility spikes |
| Miss pre-earnings uncertainty | Capture options market fear |
| Uniform confidence across events | Event-aware forecasts |

### Research Questions (for Dr. Zhan)

1. Does IV improve DeepAR forecast accuracy?
2. How far ahead should we look (1M, 3M, 6M IV)?
3. Does IV-enhanced DeepAR improve PPO Sharpe Ratio?
4. What correlation exists between IV skew and next-day returns?

---

## Securities Tracked

| Symbol | Name | Bloomberg Ticker |
|--------|------|------------------|
| AAPL | Apple Inc. | AAPL US Equity |
| AMZN | Amazon.com Inc. | AMZN US Equity |
| META | Meta Platforms Inc. | META US Equity |
| MSFT | Microsoft Corp. | MSFT US Equity |
| NVDA | NVIDIA Corp. | NVDA US Equity |
| TSLA | Tesla Inc. | TSLA US Equity |
| NDX | NASDAQ-100 Index | NDX Index |
| SPX | S&P 500 Index | SPX Index |
| PSQ | ProShares Short QQQ | PSQ US Equity |

---

## References

- [DeepAR Paper](https://arxiv.org/abs/1704.04110) - Salinas et al., Amazon, 2017
- [PyTorch LSTM Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Bloomberg API Docs](https://www.bloomberg.com/professional/support/api-library/)

---

*Last updated: January 2026 | Western AI Research Team*
