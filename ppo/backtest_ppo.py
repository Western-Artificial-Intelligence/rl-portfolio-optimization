# =======================
# File: ppo/backtest_ppo.py
# =======================
# Backtesting Script for PPO Portfolio Agent
#
# Evaluates the trained PPO agent on holdout data and compares
# against baseline strategies (S&P 500 Buy-and-Hold).
#
# Usage:
#   python -m ppo.backtest_ppo
#   python -m ppo.backtest_ppo --model checkpoints/ppo/ppo_final.zip
#   python -m ppo.backtest_ppo --model checkpoints/rest/best_model.zip --name "PPO+ReST"
# =======================

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

# Assets matching the training environment
ASSETS = [
    "AAPL US Equity",
    "AMZN US Equity",
    "META US Equity",
    "MSFT US Equity",
    "NDX Index",
    "NVDA US Equity",
    "PSQ US Equity",
    "SPX Index",
    "TSLA US Equity",
]


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the dataset."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "security"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['security'].nunique()} securities")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def load_cached_states(cache_path: str) -> dict:
    """Load pre-computed super-states for fast inference."""
    print(f"Loading cached super-states from: {cache_path}")
    cache_data = np.load(cache_path, allow_pickle=True)
    cached_states = {
        'states': cache_data['states'],
        'dates': list(cache_data['dates']),
    }
    print(f"  Loaded {len(cached_states['dates'])} pre-computed observations")
    return cached_states


def get_eval_dates(df: pd.DataFrame, train_ratio: float = 0.8) -> list:
    """Get the evaluation (holdout) dates."""
    all_dates = sorted(df["date"].unique())
    split_idx = int(len(all_dates) * train_ratio)
    eval_dates = all_dates[split_idx:]
    print(f"\nEvaluation period:")
    print(f"  {len(eval_dates)} trading days")
    print(f"  {pd.Timestamp(eval_dates[0]).date()} to {pd.Timestamp(eval_dates[-1]).date()}")
    return eval_dates


def pivot_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form data to wide-form (date x security prices)."""
    pivot = df.pivot(index="date", columns="security", values="close")
    pivot = pivot.sort_index()
    return pivot


def run_ppo_backtest(
    model: PPO,
    price_df: pd.DataFrame,
    eval_dates: list,
    cached_states: dict = None,
    initial_cash: float = 1_000_000,
) -> dict:
    """
    Run the trained PPO agent through the evaluation period.
    
    This is a simplified backtest that:
    1. Gets observation from cached states
    2. Gets action (portfolio weights) from PPO
    3. Simulates portfolio rebalancing
    4. Tracks portfolio value
    """
    print("\nRunning PPO backtest...")
    
    # Build date-to-state index lookup
    date_to_idx = {}
    if cached_states:
        for idx, d in enumerate(cached_states['dates']):
            date_to_idx[str(d)[:10]] = idx
    
    # Initialize portfolio simulation
    portfolio_value = initial_cash
    cash = initial_cash
    holdings = {asset: 0.0 for asset in ASSETS}  # shares held
    
    # Track results
    portfolio_values = [initial_cash]
    weight_history = []
    dates_tracked = []
    
    # Filter to eval dates that exist in both price data and cache
    valid_dates = []
    for d in eval_dates:
        date_str = str(d)[:10]
        if date_str in date_to_idx and d in price_df.index:
            valid_dates.append(d)
    
    print(f"  Valid eval dates: {len(valid_dates)}")
    
    for i, date in enumerate(valid_dates):
        date_str = str(date)[:10]
        
        # Get observation from cache
        if date_str in date_to_idx:
            obs = cached_states['states'][date_to_idx[date_str]].astype(np.float32)
        else:
            obs = np.zeros(64, dtype=np.float32)
        
        # Get action from trained model (deterministic)
        action, _ = model.predict(obs, deterministic=True)
        
        # Normalize weights to sum to 1
        weights = np.clip(action, 0, 1)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(len(ASSETS)) / len(ASSETS)
        
        # Get current prices
        try:
            prices = {asset: price_df.loc[date, asset] for asset in ASSETS}
        except KeyError:
            continue  # Skip if prices not available
        
        # Calculate current portfolio value
        holdings_value = sum(holdings[a] * prices[a] for a in ASSETS)
        portfolio_value = cash + holdings_value
        
        # Rebalance to target weights
        target_values = {a: weights[j] * portfolio_value for j, a in enumerate(ASSETS)}
        
        # Execute trades (simplified, no slippage for now)
        for j, asset in enumerate(ASSETS):
            current_value = holdings[asset] * prices[asset]
            target_value = target_values[asset]
            diff = target_value - current_value
            
            if abs(diff) > 1:  # Min trade threshold
                shares_to_trade = diff / prices[asset]
                holdings[asset] += shares_to_trade
                cash -= diff
        
        # Record new portfolio value after trades
        holdings_value = sum(holdings[a] * prices[a] for a in ASSETS)
        portfolio_value = cash + holdings_value
        
        portfolio_values.append(portfolio_value)
        weight_history.append(weights.copy())
        dates_tracked.append(date)
        
        if (i + 1) % 50 == 0:
            print(f"  Day {i+1}/{len(valid_dates)}: ${portfolio_value:,.2f}")
    
    # Compute metrics
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        total_return = (portfolio_values[-1] - initial_cash) / initial_cash
        
        # Sharpe (annualized)
        if np.std(returns) > 1e-8:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        peak = initial_cash
        max_dd = 0
        for val in portfolio_values:
            peak = max(peak, val)
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
    else:
        total_return = 0
        sharpe = 0
        max_dd = 0
    
    metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "final_value": portfolio_values[-1] if portfolio_values else initial_cash,
        "num_days": len(portfolio_values) - 1,
    }
    
    print(f"\nPPO Backtest Complete:")
    print(f"  Final Value: ${metrics['final_value']:,.2f}")
    print(f"  Total Return: {metrics['total_return'] * 100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    
    return {
        "portfolio_values": portfolio_values,
        "weight_history": weight_history,
        "dates": dates_tracked,
        "metrics": metrics,
    }


def run_baseline_backtest(
    price_df: pd.DataFrame,
    eval_dates: list,
    initial_cash: float = 1_000_000,
    benchmark_ticker: str = "SPX Index",
) -> dict:
    """
    Run S&P 500 buy-and-hold baseline strategy.
    """
    print(f"\nRunning baseline backtest ({benchmark_ticker})...")
    
    # Filter to dates in price data
    valid_dates = [d for d in eval_dates if d in price_df.index]
    
    if benchmark_ticker not in price_df.columns:
        print(f"  Warning: {benchmark_ticker} not found!")
        return {"portfolio_values": [initial_cash], "dates": [], "metrics": {}}
    
    # Get benchmark prices for eval period
    prices = price_df.loc[valid_dates, benchmark_ticker].values
    
    # Buy and hold simulation
    initial_price = prices[0]
    shares = initial_cash / initial_price
    portfolio_values = [shares * p for p in prices]
    
    # Compute metrics
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        total_return = (portfolio_values[-1] - initial_cash) / initial_cash
        
        if np.std(returns) > 1e-8:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        peak = initial_cash
        max_dd = 0
        for val in portfolio_values:
            peak = max(peak, val)
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
    else:
        total_return = 0
        sharpe = 0
        max_dd = 0
    
    metrics = {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "final_value": portfolio_values[-1] if portfolio_values else initial_cash,
    }
    
    print(f"\nBaseline Backtest Complete ({benchmark_ticker}):")
    print(f"  Final Value: ${metrics['final_value']:,.2f}")
    print(f"  Total Return: {metrics['total_return'] * 100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    
    return {
        "portfolio_values": portfolio_values,
        "dates": valid_dates,
        "metrics": metrics,
    }


def plot_results(
    ppo_results: dict,
    baseline_results: dict,
    output_dir: str,
    model_name: str = "PPO Agent",
    file_prefix: str = "backtest",
):
    """Generate and save visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========== Equity Curve ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ppo_values = ppo_results["portfolio_values"]
    baseline_values = baseline_results["portfolio_values"]
    
    # Align lengths
    min_len = min(len(ppo_values), len(baseline_values))
    
    ax.plot(range(min_len), ppo_values[:min_len], 
            label=model_name, linewidth=2, color="blue")
    ax.plot(range(min_len), baseline_values[:min_len], 
            label="S&P 500 (Buy & Hold)", linewidth=2, color="gray", linestyle="--")
    
    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title(f"{model_name} vs S&P 500 Baseline - Equity Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.tight_layout()
    equity_path = output_path / f"{file_prefix}_equity_curve.png"
    plt.savefig(equity_path, dpi=150)
    plt.close()
    print(f"\nSaved equity curve: {equity_path}")
    
    # ========== Weight Allocation Over Time ==========
    if ppo_results.get("weight_history"):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        weights = np.array(ppo_results["weight_history"])
        x = range(len(weights))
        
        # Use short names for legend
        short_names = ["AAPL", "AMZN", "META", "MSFT", "NDX", "NVDA", "PSQ", "SPX", "TSLA"]
        
        ax.stackplot(x, weights.T, labels=short_names, alpha=0.8)
        
        ax.set_xlabel("Trading Days", fontsize=12)
        ax.set_ylabel("Portfolio Weight", fontsize=12)
        ax.set_title(f"{model_name} - Portfolio Weight Allocation Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        weights_path = output_path / f"{file_prefix}_weights.png"
        plt.savefig(weights_path, dpi=150)
        plt.close()
        print(f"Saved weight allocation: {weights_path}")


def print_comparison_table(ppo_metrics: dict, baseline_metrics: dict, model_name: str = "PPO Agent"):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {model_name:>18} {'S&P 500':>15} {'Winner':>8}")
    print("-" * 70)
    
    ppo_ret = ppo_metrics.get("total_return", 0) * 100
    base_ret = baseline_metrics.get("total_return", 0) * 100
    winner = "[*]" if ppo_ret > base_ret else ""
    print(f"{'Total Return (%)':<25} {ppo_ret:>17.2f}% {base_ret:>14.2f}% {winner:>8}")
    
    ppo_sharpe = ppo_metrics.get("sharpe_ratio", 0)
    base_sharpe = baseline_metrics.get("sharpe_ratio", 0)
    winner = "[*]" if ppo_sharpe > base_sharpe else ""
    print(f"{'Sharpe Ratio':<25} {ppo_sharpe:>18.3f} {base_sharpe:>15.3f} {winner:>8}")
    
    ppo_dd = ppo_metrics.get("max_drawdown", 0) * 100
    base_dd = baseline_metrics.get("max_drawdown", 0) * 100
    winner = "[*]" if ppo_dd < base_dd else ""
    print(f"{'Max Drawdown (%)':<25} {ppo_dd:>17.2f}% {base_dd:>14.2f}% {winner:>8}")
    
    ppo_val = ppo_metrics.get("final_value", 0)
    base_val = baseline_metrics.get("final_value", 0)
    print(f"{'Final Value ($)':<25} {ppo_val:>17,.0f} {base_val:>14,.0f}")
    
    print("=" * 70)


def save_metrics(
    ppo_results: dict, 
    baseline_results: dict, 
    output_dir: str,
    model_name: str = "PPO Agent",
    file_prefix: str = "backtest",
):
    """Save metrics to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "agent": ppo_results["metrics"],
        "baseline_sp500": baseline_results["metrics"],
    }
    
    json_path = output_path / f"{file_prefix}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    
    print(f"Saved metrics: {json_path}")


def main():
    """Main backtest entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest trained PPO agent for portfolio optimization",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "ppo" / "ppo_final.zip"),
        help="Path to trained PPO model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name for display (e.g., 'PPO', 'PPO+ReST'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "deepar_dataset.csv"),
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(PROJECT_ROOT / "data" / "super_states.npz"),
        help="Path to cached super-states",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Output directory for plots and metrics",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000,
        help="Initial portfolio value",
    )
    
    args = parser.parse_args()
    
    # Auto-detect model name from path if not provided
    if args.name is None:
        model_path = args.model.lower()
        if "rest" in model_path:
            model_name = "PPO+ReST"
            file_prefix = "backtest_rest"
        else:
            model_name = "PPO"
            file_prefix = "backtest_ppo"
    else:
        model_name = args.name
        # Create safe filename from model name
        file_prefix = "backtest_" + args.name.lower().replace("+", "_").replace(" ", "_")
    
    print("\n" + "=" * 70)
    print("PORTFOLIO BACKTESTING")
    print("=" * 70)
    print(f"Model Name: {model_name}")
    print(f"Model Path: {args.model}")
    print(f"Initial Cash: ${args.initial_cash:,.0f}")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model...")
    model = PPO.load(args.model)
    print("  Model loaded!")
    
    # Load data
    df = load_data(args.data)
    
    # Load cached states
    cached_states = None
    if Path(args.cache).exists():
        cached_states = load_cached_states(args.cache)
    else:
        print(f"\n[WARNING] Cache not found at {args.cache}")
        return
    
    # Pivot data to wide format (date x security)
    price_df = pivot_prices(df)
    print(f"\nPrice matrix: {price_df.shape[0]} dates x {price_df.shape[1]} securities")
    
    # Get eval dates (last 20%)
    eval_dates = get_eval_dates(df)
    
    # Run model backtest
    ppo_results = run_ppo_backtest(
        model=model,
        price_df=price_df,
        eval_dates=eval_dates,
        cached_states=cached_states,
        initial_cash=args.initial_cash,
    )
    
    # Run baseline backtest
    baseline_results = run_baseline_backtest(
        price_df=price_df,
        eval_dates=eval_dates,
        initial_cash=args.initial_cash,
    )
    
    # Print comparison
    print_comparison_table(ppo_results["metrics"], baseline_results["metrics"], model_name)
    
    # Generate plots
    plot_results(ppo_results, baseline_results, args.output, model_name, file_prefix)
    
    # Save metrics
    save_metrics(ppo_results, baseline_results, args.output, model_name, file_prefix)
    
    print("\n[OK] Backtest complete!")


if __name__ == "__main__":
    main()
