# Portfolio Optimization with Reinforcement Learning and Real-Time Financial Feeds  
**Western AI – 2025 Project**

## 📈 Overview  
This project proposes a **hybrid AI-driven portfolio optimization system** that combines **Reinforcement Learning (RL)** for adaptive asset allocation with **Natural Language Processing (NLP)** for real-time sentiment analysis.  

By integrating **quantitative market data** with **qualitative sentiment signals** from financial news sources (via models like **FinBERT**) and training an **RL agent (e.g., PPO, Actor-Critic)** on this data, our goal is to achieve **superior risk-adjusted returns** compared to traditional approaches based solely on historical market data.

---

## 🧠 Core Objectives  
- **Develop an RL-based trading agent** capable of dynamic portfolio rebalancing.  
- **Integrate real-time financial sentiment data** into the agent’s decision-making process.  
- **Backtest and compare** model performance against classical benchmarks (e.g., mean-variance optimization).  
- **Visualize results** through a modern, interactive dashboard showing live portfolio performance, sentiment feed, and allocation breakdowns.

---

## 🧩 Tech Stack  
**Backend & Modeling**
- Python, PyTorch / TensorFlow  
- Stable-Baselines3 (for RL algorithms)  
- FinBERT (for financial sentiment analysis)  
- Bloomberg or Yahoo Finance API (for data collection)  
- Pandas, NumPy, Scikit-learn  

**Frontend**
- React.js + D3.js for data visualization  
- Streamlit prototype (for early testing)  

**Infrastructure**
- Docker for containerization  
- Google Cloud / AWS (for model training and hosting)  

---

## 🧪 System Architecture  
[Bloomberg/Yahoo API] → [Data Processing Layer] → [NLP Model (FinBERT)]
↓
[RL Agent (PPO/Actor-Critic)]
↓
[Portfolio Management + Simulation Engine]
↓
[React/Streamlit Dashboard Visualization]


## 🚀 Getting Started

### 1. Environment Setup

It is **strongly recommended** you set up your development environment **before our first team meeting** to ensure everything installs correctly and to flag any setup issues early.

---

### 🧩 Prerequisites
Before you begin, make sure you have the following installed:
- **Python 3.10+**
- **Node.js** (for frontend/dashboard components)
- **Git & GitHub account**
- **Access to Bloomberg or Yahoo Finance API** (for financial data)

---

### ⚙️ Step 1: Install [UV](https://docs.astral.sh/uv/)

UV is a **blazing-fast Python package manager and environment tool** developed by Astral (creators of Ruff & Rye).  
It replaces `pip`, `venv`, and `pip-tools` — managing dependencies, environments, and Python versions in one tool.

#### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

### ⚙️ Step 2: Clone the Repo
git clone https://github.com/hAlcLeite/Portfolio-Optimizer.git
cd Portfolio-Optimizer


### ⚙️ Step 3: Create and Activate a Virtual Environment
uv venv

Activate the environment
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows

### ⚙️ Step 4: Install Dependencies
uv pip install -r pyproject.toml

or simply:
uv sync

### ⚙️ Step 5: Run the Project
uv run streamlit run app.py

Or, to test a specific script:

uv run main.py


#### Clone Repository  
```bash
git clone https://github.com/hAlcLeite/Portfolio-Optimizer.git
cd Portfolio-Optimizer
