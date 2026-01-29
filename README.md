# 🚀 Project Alpha-RL: Regime-Aware Portfolio Optimization

<div align="center">

**An Informed RL Agent for Portfolio Optimization**

Western AI – 2025-2026 Project

Dataset: https://drive.google.com/drive/folders/1DzsK6fLDA-q-fbjGWCoMtdj4BDn_JrkO?usp=sharing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

*Western AI – 2025-2026 Research Project*

</div>

---

##  Overview

This project develops a **state-of-the-art AI trading agent** that makes portfolio allocation decisions using:

| Module | Technology | Purpose |
|--------|------------|---------|
| 🔮 **DeepAR** | Probabilistic LSTM | Forecasts returns with uncertainty estimates |
| 🌐 **FRED API** | Federal Reserve Data | Tracks macro-economic regimes (VIX, yield curve, Fed rate) |
| 📰 **FinBERT** | Transformer NLP | Extracts sentiment from financial news *(planned)* |
| 🧠 **PPO Agent** | Reinforcement Learning | Makes portfolio allocation decisions |

The agent observes a **64-dimensional "Super-State"** combining forecasts, macro data, and sentiment to make regime-aware investment decisions.

---

##  Key Features

- **Probabilistic Forecasting**: Not just "price will be $150" but "90% chance between $145-$155"
- **Regime Awareness**: Agent adapts strategy based on economic conditions (bull/bear/crisis)
- **Uncertainty-Aware**: Takes smaller positions when forecasts are uncertain
- **ReST Training**: Novel "Grow/Improve" methodology adapted from language modeling

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PPO AGENT (Brain)                           │
│                     Outputs: Portfolio Weights                      │
│                   [AAPL: 0.3, MSFT: 0.5, CASH: 0.2]                │
└─────────────────────────────────────────────────────────────────────┘
                                    ▲
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────┴───────┐ ┌─────┴─────┐ ┌───────┴───────┐
            │   DeepAR      │ │  FinBERT  │ │  FRED API     │
            │  (Forecaster) │ │   (NLP)   │ │ (Macro Data)  │
            └───────┬───────┘ └─────┬─────┘ └───────┬───────┘
                    │               │               │
            ┌───────┴───────┐ ┌─────┴─────┐ ┌───────┴───────┐
            │ Price History │ │   News    │ │ VIX, Yields,  │
            │ OHLCV Data    │ │ Articles  │ │ Fed Rates     │
            └───────────────┘ └───────────┘ └───────────────┘
```

---

##  Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| DeepAR Model | ✅ Complete | Trained on 9 securities, 60-day context |
| FRED Data | ✅ Complete | VIX, Yield Curve, Fed Funds Rate |
| SuperStateBuilder | ✅ Complete | 64-dim observation vector |
| PortfolioEnv | ✅ Complete | Gymnasium-compliant trading environment |
| PPO Training | 🔄 In Progress | ReST training methodology |
| FinBERT Sentiment | 📋 Planned | NLP module |
| Dashboard | 📋 Planned | React/Streamlit visualization |

---

##  Quick Start

### Prerequisites

- Python 3.10+
- [UV](https://pythdocs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Western-Artificial-Intelligence/rl-portfolio-optimization.git
cd rl-portfolio-optimization

# Create virtual environment with UV
uv venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
uv sync
```

### Run DeepAR Training

```bash
# Train the forecasting model
uv run python deepAR/train_deepar.py --epochs 30
```

### Test the Environment

```bash
# Test SuperStateBuilder
python -m ppo.super_state

# Test PortfolioEnv
python -c "
import pandas as pd
from backtesting.core.PortfolioEnv import PortfolioEnv

df = pd.read_csv('data/deepar_dataset.csv')
env = PortfolioEnv(df=df, use_super_state=True)
obs, info = env.reset()
print(f'Observation shape: {obs.shape}')  # (64,)
print('✓ Environment ready!')
"
```

---

##  Project Structure

```
Portfolio-Optimizer/
├── 📁 data/                     # Market data
│   ├── FRED/                    # Macro-economic data
│   │   ├── VIXCLS.csv          # VIX volatility index
│   │   ├── T10Y2Y.csv          # Yield curve spread
│   │   └── FEDFUNDS.csv        # Federal funds rate
│   ├── deepar_dataset.csv      # Processed training data
│   └── *.csv                   # Price data files
│
├── 📁 deepAR/                   # Forecasting module
│   ├── model.py                # DeepARModel + DeepARForecaster
│   ├── train_deepar.py         # Training pipeline
│   └── preprocessing.py        # Data utilities
│
├── 📁 ppo/                      # RL Agent module
│   ├── __init__.py
│   └── super_state.py          # SuperStateBuilder class
│
├── 📁 backtesting/              # Trading environment
│   └── core/
│       └── PortfolioEnv.py     # Gymnasium environment
│
├── 📁 checkpoints/              # Saved models
│   └── deepar/
│       ├── deepar_best.pt      # Best validation model
│       └── training_summary.json
│
├── 📁 tests/                    # Unit tests
│   ├── test_super_state.py
│   └── test_portfolio_env.py
│
└── 📁 docs/                     # Documentation
    └── ARCHITECTURE.md
```

---

##  Super-State Vector

The agent observes a **64-dimensional vector** at each step:

| Index | Features | Count | Source |
|-------|----------|-------|--------|
| 0-53 | Per-stock forecasts (mean, std, skew, confidence, q10, q90) | 54 | DeepAR |
| 54-59 | Macro indicators (VIX, yield curve, fed rate) | 6 | FRED |
| 60-63 | Sentiment placeholders | 4 | FinBERT (TBD) |

All values are normalized to **[-1, 1]** range for stable training.

---

##  Securities Tracked

The DeepAR model is trained on 9 securities:

| Symbol | Name | Type |
|--------|------|------|
| AAPL | Apple Inc. | Stock |
| AMZN | Amazon.com Inc. | Stock |
| META | Meta Platforms Inc. | Stock |
| MSFT | Microsoft Corp. | Stock |
| NVDA | NVIDIA Corp. | Stock |
| TSLA | Tesla Inc. | Stock |
| NDX | NASDAQ-100 Index | Index |
| SPX | S&P 500 Index | Index |
| PSQ | ProShares Short QQQ | Inverse ETF |

---

##  Tech Stack

| Category | Technologies |
|----------|--------------|
| **ML/RL** | PyTorch, Stable-Baselines3, Gymnasium |
| **Data** | Pandas, NumPy, SciPy |
| **Finance** | Bloomberg API, FRED API |
| **NLP** | FinBERT, Transformers |
| **DevOps** | UV, pytest, Git |

---

##  References

1. **DeepAR**: Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
2. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms"
3. **ReST**: Gulcehre et al., "Reinforced Self-Training (ReST) for Language Modeling"
4. **FinBERT**: Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"

---

##  Team

**Western AI Research Group** – 2025-2026

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**[Documentation](docs/ARCHITECTURE.md)** · **[Report Bug](https://github.com/Western-Artificial-Intelligence/rl-portfolio-optimization/issues)** · **[Request Feature](https://github.com/Western-Artificial-Intelligence/rl-portfolio-optimization/issues)**

</div>
