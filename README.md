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
It is **recommended you set up your environment** before our first team meeting to identify any issues early.  

#### Prerequisites  
- Python 3.10+  
- Node.js (for frontend)  
- Git & GitHub account  
- Access to Bloomberg or Yahoo Finance API  

#### Clone Repository  
```bash
git clone https://github.com/hAlcLeite/Portfolio-Optimizer.git
cd Portfolio-Optimizer
