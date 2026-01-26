# =======================
# File: ppo/monte_carlo_backtest.py
# =======================
# Monte Carlo Simulation for PPO Portfolio Agent
#
# Runs multiple backtests with noise to measure robustness and
# generate confidence intervals for performance metrics.
#
# Usage:
#   python -m ppo.monte_carlo_backtest
#   python -m ppo.monte_carlo_backtest --runs 100 --noise 0.02
# =======================

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "security"]).reset_index(drop=True)
    return df


def load_cached_states(cache_path: str) -> dict:
    """Load pre-computed super-states."""
    cache_data = np.load(cache_path, allow_pickle=True)
    return {
        'states': cache_data['states'].copy(),
        'dates': list(cache_data['dates']),
    }


def get_eval_dates(df: pd.DataFrame, train_ratio: float = 0.8) -> list:
    """Get the evaluation (holdout) dates."""
    all_dates = sorted(df["date"].unique())
    split_idx = int(len(all_dates) * train_ratio)
    return all_dates[split_idx:]


def pivot_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form data to wide-form."""
    pivot = df.pivot(index="date", columns="security", values="close")
    return pivot.sort_index()


def run_single_backtest(
    model: PPO,
    price_df: pd.DataFrame,
    eval_dates: list,
    cached_states: dict,
    obs_noise_scale: float = 0.0,
    slippage_bps: float = 5.0,
    initial_cash: float = 1_000_000,
) -> dict:
    """
    Run a single backtest with optional noise.
    
    Args:
        model: Trained PPO model
        price_df: Price matrix (date x security)
        eval_dates: List of evaluation dates
        cached_states: Pre-computed observations
        obs_noise_scale: Std of Gaussian noise added to observations
        slippage_bps: Slippage in basis points
        initial_cash: Starting capital
    
    Returns:
        Dictionary with metrics
    """
    # Build date-to-state index lookup
    date_to_idx = {}
    for idx, d in enumerate(cached_states['dates']):
        date_to_idx[str(d)[:10]] = idx
    
    # Add noise to states if specified
    states = cached_states['states'].copy()
    if obs_noise_scale > 0:
        noise = np.random.normal(0, obs_noise_scale, states.shape)
        states = states + noise
        # Clip to valid range
        states = np.clip(states, -1, 1)
    
    # Initialize portfolio
    portfolio_value = initial_cash
    cash = initial_cash
    holdings = {asset: 0.0 for asset in ASSETS}
    
    portfolio_values = [initial_cash]
    slippage_rate = slippage_bps / 10_000
    
    # Filter valid dates
    valid_dates = [d for d in eval_dates 
                   if str(d)[:10] in date_to_idx and d in price_df.index]
    
    for date in valid_dates:
        date_str = str(date)[:10]
        
        # Get observation (with noise already applied)
        obs = states[date_to_idx[date_str]].astype(np.float32)
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Normalize weights
        weights = np.clip(action, 0, 1)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(len(ASSETS)) / len(ASSETS)
        
        # Get prices
        try:
            prices = {asset: price_df.loc[date, asset] for asset in ASSETS}
        except KeyError:
            continue
        
        # Current portfolio value
        holdings_value = sum(holdings[a] * prices[a] for a in ASSETS)
        portfolio_value = cash + holdings_value
        
        # Target allocation
        target_values = {a: weights[j] * portfolio_value for j, a in enumerate(ASSETS)}
        
        # Execute trades with slippage
        for j, asset in enumerate(ASSETS):
            current_value = holdings[asset] * prices[asset]
            target_value = target_values[asset]
            diff = target_value - current_value
            
            if abs(diff) > 1:
                # Apply slippage
                if diff > 0:  # Buy
                    effective_price = prices[asset] * (1 + slippage_rate)
                else:  # Sell
                    effective_price = prices[asset] * (1 - slippage_rate)
                
                shares_to_trade = diff / effective_price
                holdings[asset] += shares_to_trade
                cash -= shares_to_trade * effective_price
        
        # Update portfolio value
        holdings_value = sum(holdings[a] * prices[a] for a in ASSETS)
        portfolio_value = cash + holdings_value
        portfolio_values.append(portfolio_value)
    
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
    
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "final_value": portfolio_values[-1] if portfolio_values else initial_cash,
    }


def run_monte_carlo(
    model: PPO,
    price_df: pd.DataFrame,
    eval_dates: list,
    cached_states: dict,
    n_runs: int = 100,
    obs_noise_scale: float = 0.01,
    slippage_noise_bps: float = 3.0,
    initial_cash: float = 1_000_000,
) -> Dict[str, List[float]]:
    """
    Run Monte Carlo simulation with multiple backtests.
    
    Args:
        model: Trained PPO model
        n_runs: Number of simulation runs
        obs_noise_scale: Std of observation noise (default 1%)
        slippage_noise_bps: Random variation in slippage (±X bps)
    
    Returns:
        Dictionary of metric lists
    """
    print(f"\n🎲 Running Monte Carlo Simulation ({n_runs} runs)")
    print(f"   Observation noise: ±{obs_noise_scale*100:.1f}%")
    print(f"   Slippage noise: 5 ± {slippage_noise_bps:.0f} bps")
    
    results = {
        "total_return": [],
        "sharpe_ratio": [],
        "max_drawdown": [],
        "final_value": [],
    }
    
    for i in tqdm(range(n_runs), desc="Simulations"):
        # Random slippage between (5 - noise) and (5 + noise) bps
        slippage = 5 + np.random.uniform(-slippage_noise_bps, slippage_noise_bps)
        
        # Run backtest with noise
        metrics = run_single_backtest(
            model=model,
            price_df=price_df,
            eval_dates=eval_dates,
            cached_states=cached_states,
            obs_noise_scale=obs_noise_scale,
            slippage_bps=slippage,
            initial_cash=initial_cash,
        )
        
        for key in results:
            results[key].append(metrics[key])
    
    return results


def print_monte_carlo_results(results: Dict[str, List[float]], baseline_metrics: dict):
    """Print formatted Monte Carlo results."""
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Mean':>12} {'Std':>12} {'95% CI':>20} {'Baseline':>12}")
    print("-" * 70)
    
    for metric, values in results.items():
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        ci_low = np.percentile(arr, 2.5)
        ci_high = np.percentile(arr, 97.5)
        baseline = baseline_metrics.get(metric, 0)
        
        if "return" in metric or "drawdown" in metric:
            print(f"{metric:<20} {mean*100:>11.2f}% {std*100:>11.2f}% "
                  f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%] {baseline*100:>11.2f}%")
        elif "value" in metric:
            print(f"{metric:<20} ${mean:>10,.0f} ${std:>10,.0f} "
                  f"[${ci_low:,.0f}, ${ci_high:,.0f}] ${baseline:>10,.0f}")
        else:
            print(f"{metric:<20} {mean:>12.3f} {std:>12.3f} "
                  f"[{ci_low:.2f}, {ci_high:.2f}] {baseline:>12.3f}")
    
    print("=" * 70)
    
    # Statistical significance test
    sharpe_arr = np.array(results["sharpe_ratio"])
    baseline_sharpe = baseline_metrics.get("sharpe_ratio", 0)
    pct_beat_baseline = np.mean(sharpe_arr > baseline_sharpe) * 100
    
    print(f"\n📊 PPO beats S&P 500 Sharpe in {pct_beat_baseline:.1f}% of simulations")


def plot_monte_carlo_results(results: Dict[str, List[float]], output_dir: str):
    """Generate Monte Carlo visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_config = [
        ("total_return", "Total Return", True),
        ("sharpe_ratio", "Sharpe Ratio", False),
        ("max_drawdown", "Max Drawdown", True),
        ("final_value", "Final Value ($)", False),
    ]
    
    for ax, (metric, title, is_pct) in zip(axes.flat, metrics_config):
        values = np.array(results[metric])
        if is_pct:
            values = values * 100
        
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(np.percentile(values, 2.5), color='orange', linestyle=':', linewidth=2, label='95% CI')
        ax.axvline(np.percentile(values, 97.5), color='orange', linestyle=':', linewidth=2)
        
        ax.set_xlabel(f"{title} {'(%)' if is_pct else ''}", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"Distribution of {title}", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Monte Carlo Simulation Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_path / "monte_carlo_distributions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved distribution plot: {plot_path}")


def save_monte_carlo_results(results: Dict[str, List[float]], output_dir: str):
    """Save Monte Carlo results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_runs": len(results["total_return"]),
        "metrics": {}
    }
    
    for metric, values in results.items():
        arr = np.array(values)
        summary["metrics"][metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "ci_2.5": float(np.percentile(arr, 2.5)),
            "ci_97.5": float(np.percentile(arr, 97.5)),
        }
    
    json_path = output_path / "monte_carlo_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved results: {json_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation for PPO portfolio agent",
    )
    
    parser.add_argument("--model", type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "ppo" / "ppo_final.zip"))
    parser.add_argument("--data", type=str,
        default=str(PROJECT_ROOT / "data" / "deepar_dataset.csv"))
    parser.add_argument("--cache", type=str,
        default=str(PROJECT_ROOT / "data" / "super_states.npz"))
    parser.add_argument("--output", type=str,
        default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--runs", type=int, default=100,
        help="Number of Monte Carlo runs")
    parser.add_argument("--noise", type=float, default=0.01,
        help="Observation noise scale (0.01 = 1%)")
    parser.add_argument("--slippage-noise", type=float, default=3.0,
        help="Slippage variation in bps")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MONTE CARLO BACKTESTING")
    print("=" * 60)
    print(f"Runs: {args.runs}")
    print(f"Observation noise: {args.noise*100:.1f}%")
    print(f"Slippage noise: ±{args.slippage_noise:.0f} bps")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(args.model)
    
    # Load data
    print("Loading data...")
    df = load_data(args.data)
    cached_states = load_cached_states(args.cache)
    price_df = pivot_prices(df)
    eval_dates = get_eval_dates(df)
    
    print(f"Eval period: {len(eval_dates)} days")
    
    # Run Monte Carlo
    results = run_monte_carlo(
        model=model,
        price_df=price_df,
        eval_dates=eval_dates,
        cached_states=cached_states,
        n_runs=args.runs,
        obs_noise_scale=args.noise,
        slippage_noise_bps=args.slippage_noise,
    )
    
    # Baseline for comparison
    baseline_metrics = {
        "total_return": 0.310,
        "sharpe_ratio": 1.625,
        "max_drawdown": 0.088,
        "final_value": 1310022,
    }
    
    # Print results
    print_monte_carlo_results(results, baseline_metrics)
    
    # Generate plots
    plot_monte_carlo_results(results, args.output)
    
    # Save results
    save_monte_carlo_results(results, args.output)
    
    print("\n✓ Monte Carlo simulation complete!")


if __name__ == "__main__":
    main()
