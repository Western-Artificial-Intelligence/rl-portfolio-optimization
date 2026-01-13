# 🏗️ Project Alpha-RL: Architecture Documentation

> **An Informed RL Agent for Portfolio Optimization**
>
> This document provides a comprehensive technical overview of the project architecture, component interactions, and implementation details.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Component Deep Dive](#3-component-deep-dive)
4. [Data Flow](#4-data-flow)
5. [Directory Structure](#5-directory-structure)
6. [Training Pipeline](#6-training-pipeline)
7. [Configuration](#7-configuration)
8. [Future Roadmap](#8-future-roadmap)

---

## 1. Project Overview

### 1.1 Mission

Build an autonomous **Reinforcement Learning (RL) agent** that intelligently manages a stock portfolio to maximize **risk-adjusted returns (Sharpe Ratio)**.

### 1.2 Core Innovation

Unlike traditional trading bots that only analyze price data, our agent uses a **"Super-State"** that combines:

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **DeepAR** | Historical prices | Probability distribution (μ, σ) | Uncertainty-aware forecasting |
| **FinBERT/NLP** | Financial news | Sentiment vector | Market mood detection |
| **FRED API** | Economic indicators | Macro vector | Regime awareness |
| **PPO Agent** | All of the above | Portfolio weights | Decision making |

### 1.3 Key Differentiators

1. **Probabilistic Forecasting**: Not just "price will be $150" but "90% chance between $145-$155"
2. **Regime Awareness**: Agent knows if we're in a bull market, recession, or crisis
3. **ReST Training**: Novel "Grow/Improve" methodology adapted from language models

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PPO AGENT (Brain)                           │
│                     Outputs: Portfolio Weights                      │
│                   [AAPL: 0.3, MSFT: 0.5, CASH: 0.2]                │
└─────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────┴───────┐ ┌─────┴─────┐ ┌───────┴───────┐
            │   DeepAR      │ │  FinBERT  │ │  FRED API     │
            │  (Forecaster) │ │   (NLP)   │ │ (Macro Data)  │
            └───────────────┘ └───────────┘ └───────────────┘
                    ▲               ▲               ▲
                    │               │               │
            ┌───────┴───────┐ ┌─────┴─────┐ ┌───────┴───────┐
            │ Price History │ │   News    │ │ Fed Rates,    │
            │ OHLCV Data    │ │  Articles │ │ VIX, Yields   │
            └───────────────┘ └───────────┘ └───────────────┘
```

### 2.2 Data Flow Diagram

```
       ┌──────────────────────────────────────────────────────────────┐
       │                      DATA SOURCES                            │
       └──────────────────────────────────────────────────────────────┘
                    │                   │                   │
                    ▼                   ▼                   ▼
       ┌────────────────────┐ ┌─────────────────┐ ┌───────────────────┐
       │ Bloomberg/Yahoo    │ │ Financial News  │ │ Federal Reserve   │
       │ Price Data         │ │ APIs            │ │ (FRED API)        │
       └────────────────────┘ └─────────────────┘ └───────────────────┘
                    │                   │                   │
                    ▼                   ▼                   ▼
       ┌────────────────────┐ ┌─────────────────┐ ┌───────────────────┐
       │ build_deepar_      │ │ FinBERT         │ │ fred_data.py      │
       │ dataset.py         │ │ Pipeline        │ │                   │
       └────────────────────┘ └─────────────────┘ └───────────────────┘
                    │                   │                   │
                    ▼                   ▼                   ▼
       ┌────────────────────┐ ┌─────────────────┐ ┌───────────────────┐
       │ deepar_dataset.csv │ │ Sentiment       │ │ fred_macro_data   │
       │ (Engineered        │ │ Scores          │ │ .csv              │
       │  Features)         │ │                 │ │                   │
       └────────────────────┘ └─────────────────┘ └───────────────────┘
                    │                   │                   │
                    ▼                   ▼                   ▼
       ┌──────────────────────────────────────────────────────────────┐
       │                    SUPER-STATE VECTOR                        │
       │  [Forecast μ, σ] + [Sentiment] + [Macro: VIX, Yield, Rate]  │
       └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────────┐
       │                      PPO AGENT                               │
       │              Proximal Policy Optimization                    │
       └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────────┐
       │                   PORTFOLIO WEIGHTS                          │
       │           Action: Rebalance to new allocations               │
       └──────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep Dive

### 3.1 DeepAR Forecaster (`deepAR/`)

**Purpose**: Predict the probability distribution of future stock returns.

#### Architecture

```
Input: [Price History, Dynamic Features, Series Embedding]
                            │
                            ▼
                   ┌────────────────┐
                   │  LSTM Network  │
                   │  (2 layers,    │
                   │   64 hidden)   │
                   └────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Gaussian Head  │
                   │   μ layer      │
                   │   σ layer      │
                   └────────────────┘
                            │
                            ▼
            Output: (mean, variance) distribution
```

#### Key Files

| File | Purpose |
|------|---------|
| `deepAR/model.py` | DeepARModel class with LSTM + Gaussian output |
| `deepAR/train_deepar.py` | Training pipeline with early stopping |
| `deepAR/bloomberg-data-extraction/build_deepar_dataset.py` | Feature engineering |

#### Features Engineered

| Feature | Formula | Purpose |
|---------|---------|---------|
| `log_return` | ln(Pₜ / Pₜ₋₁) | Stationarity |
| `roll_vol_10` | std(returns, 10d) | Volatility signal |
| `ret_5d` | (Pₜ - Pₜ₋₅) / Pₜ₋₅ | Short-term momentum |
| `ret_20d` | (Pₜ - Pₜ₋₂₀) / Pₜ₋₂₀ | Medium-term momentum |
| `zvol` | (Vol - μ) / σ | Volume anomaly |

#### Training Configuration

```python
DEFAULT_CONFIG = {
    "hidden_size": 64,        # LSTM hidden units
    "num_layers": 2,          # LSTM depth
    "dropout": 0.1,           # Regularization
    "context_length": 60,     # 60 days of history
    "prediction_length": 5,   # Predict 5 days ahead
    "batch_size": 64,
    "learning_rate": 1e-3,
    "patience": 10,           # Early stopping
}
```

---

### 3.2 FRED Macro Monitor (`fred_data_extraction/`)

**Purpose**: Track economic regime indicators from the Federal Reserve.

#### Key Indicators

| Indicator | FRED Code | Interpretation |
|-----------|-----------|----------------|
| **Yield Curve** | T10Y2Y | < 0 = Recession signal |
| **VIX** | VIXCLS | > 25 = High fear |
| **Fed Funds Rate** | FEDFUNDS | Rising = Tightening |

#### Derived Features

```python
# Yield curve inversion (recession predictor)
yield_curve_inverted = (T10Y2Y < 0).astype(int)

# VIX regime classification
# 0: Low (<15), 1: Normal (15-25), 2: High (25-35), 3: Extreme (>35)
vix_regime = pd.cut(VIXCLS, bins=[0, 15, 25, 35, inf])

# Fed policy direction
fed_hiking = (fed_rate_change > 0).astype(int)
fed_cutting = (fed_rate_change < 0).astype(int)
```

---

### 3.3 NLP Sentiment Analyzer (`naturalLanguageProcessing/`)

**Purpose**: Extract market sentiment from financial news and reports.

#### Pipeline

```
Raw Text → FinBERT → Sentiment Score [-1, +1]
                           │
                           ▼
              Aggregated Sentiment Vector
              (per stock, per day)
```

---

### 3.4 PPO Agent (`backtesting/`)

**Purpose**: Make portfolio allocation decisions based on the super-state.

#### Environment Interface

```python
class PortfolioEnv:
    # State: [market_data, forecast_vector, sentiment, macro_vector]
    # Action: portfolio weights [w₁, w₂, ..., wₙ] where Σwᵢ = 1
    # Reward: Sharpe Ratio of portfolio returns
```

#### Reward Function

```
Reward = (Rₚ - Rₑ) / σₚ   (Sharpe Ratio)

Where:
  Rₚ = Portfolio return
  Rₑ = Risk-free rate
  σₚ = Portfolio volatility
```

---

## 4. Data Flow

### 4.1 Training Pipeline

```
Step 1: Data Extraction
    bloomberg_prices.csv ─────┐
    nasdaq100_prices.csv ─────┼──► build_deepar_dataset.py
                              │
                              ▼
                     deepar_dataset.csv
                     (15,669 samples, 9 stocks)

Step 2: Model Training
    deepar_dataset.csv ──► train_deepar.py
                                  │
                                  ▼
                     checkpoints/deepar/
                     ├── deepar_best.pt
                     ├── deepar_final.pt
                     └── training_summary.json

Step 3: Integration (Future)
    Trained DeepAR ──┐
    FinBERT        ──┼──► PPO Agent Training
    FRED Data      ──┘
```

### 4.2 Inference Pipeline (Future)

```
Live Data ──► Preprocessor ──► DeepAR   ──┐
                              FinBERT  ──┼──► PPO ──► Trade Execution
                              FRED     ──┘
```

---

## 5. Directory Structure

```
Portfolio-Optimizer/
│
├── 📁 data/                          # Data storage
│   ├── bloomberg_prices.csv          # Raw Bloomberg price data
│   ├── nasdaq100_prices.csv          # NASDAQ-100 constituents
│   ├── sp500_prices.csv              # S&P 500 constituents
│   ├── nasdaq100_static.csv          # Static features (sector, beta)
│   └── deepar_dataset.csv            # Processed training data
│
├── 📁 deepAR/                        # DeepAR forecasting module
│   ├── model.py                      # DeepARModel architecture
│   ├── train_deepar.py               # Training pipeline
│   ├── preprocessing.py              # Data utilities
│   └── bloomberg-data-extraction/    # Data extraction scripts
│       ├── build_deepar_dataset.py   # Feature engineering
│       ├── download_nasdaq100_data.py
│       ├── download_sp500_data.py
│       └── documentation.md
│
├── 📁 fred_data_extraction/          # FRED macro data module
│   └── fred_data.py                  # FRED API integration
│
├── 📁 naturalLanguageProcessing/     # NLP sentiment module
│   └── (FinBERT implementation)
│
├── 📁 backtesting/                   # RL environment & backtesting
│   └── core/
│       └── PortfolioEnv.py           # Gym environment
│
├── 📁 checkpoints/                   # Saved model weights
│   └── deepar/
│       ├── deepar_best.pt            # Best validation model
│       ├── deepar_final.pt           # Final epoch model
│       └── training_summary.json     # Metrics & config
│
├── 📁 docs/                          # Documentation
│   └── ARCHITECTURE.md               # This file
│
├── main.py                           # Entry point
├── pyproject.toml                    # Dependencies
└── README.md                         # Quick start guide
```

---

## 6. Training Pipeline

### 6.1 DeepAR Training Steps

```bash
# Step 1: Prepare dataset (combines Bloomberg + NASDAQ-100)
python deepAR/bloomberg-data-extraction/build_deepar_dataset.py

# Step 2: Train the model
uv run python deepAR/train_deepar.py --data data/deepar_dataset.csv --epochs 30

# Step 3: Check results
cat checkpoints/deepar/training_summary.json
```

### 6.2 Understanding Training Output

```
Epoch   1/30 | Train Loss: -1.67 | Val Loss: -2.23 | LR: 1.00e-03
              ↑                    ↑                  ↑
              │                    │                  └─ Learning rate
              │                    └─ Validation loss (more negative = better)
              └─ Training loss (more negative = better)
```

**For Gaussian NLL loss:**
- More negative = Model assigns higher probability to correct values
- Less negative = Worse predictions

### 6.3 Evaluation Metrics

| Metric | Description | Good Values |
|--------|-------------|-------------|
| **MAE** | Mean Absolute Error | < 0.02 |
| **RMSE** | Root Mean Squared Error | < 0.03 |
| **Coverage 95%** | % of true values in 95% CI | ~0.95 |
| **CRPS** | Probabilistic accuracy | < 1.0 |

---

## 7. Configuration

### 7.1 Environment Variables

```bash
# .env file
FRED_API_KEY=your_fred_api_key_here
```

### 7.2 Model Hyperparameters

Edit in `deepAR/train_deepar.py`:

```python
DEFAULT_CONFIG = {
    # Architecture
    "hidden_size": 64,       # Increase for more capacity
    "num_layers": 2,         # More layers = deeper model
    "dropout": 0.1,          # Regularization strength
    
    # Training
    "batch_size": 64,        # Larger = faster but more memory
    "learning_rate": 1e-3,   # Lower for fine-tuning
    "patience": 10,          # Early stopping threshold
    
    # Data windows
    "context_length": 60,    # Days of history to use
    "prediction_length": 5,  # Days to forecast
}
```

---

## 8. Future Roadmap

### Phase 1: DeepAR (✅ Complete)
- [x] Data extraction pipeline
- [x] Feature engineering
- [x] Model architecture
- [x] Training pipeline
- [x] Evaluation metrics

### Phase 2: FRED Integration (🔄 In Progress)
- [x] FRED API wrapper
- [ ] Macro regime detection
- [ ] Integration with PPO state

### Phase 3: NLP Sentiment (📋 Planned)
- [ ] FinBERT integration
- [ ] News data pipeline
- [ ] Sentiment aggregation

### Phase 4: PPO Agent (📋 Planned)
- [ ] Super-state construction
- [ ] PPO training with ReST methodology
- [ ] Backtesting framework

### Phase 5: Production (📋 Planned)
- [ ] Live data feeds
- [ ] Paper trading
- [ ] Dashboard/monitoring

---

## 📚 References

1. **DeepAR**: Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
2. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms"
3. **ReST**: Gulcehre et al., "Reinforced Self-Training (ReST) for Language Modeling"
4. **FinBERT**: Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"

---

*Last Updated: January 2026*
