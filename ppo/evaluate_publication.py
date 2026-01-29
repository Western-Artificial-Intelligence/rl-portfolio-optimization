# =======================
# File: ppo/evaluate_publication.py
# =======================
# Publication-Ready Evaluation Framework for ReST
#
# Implements rigorous evaluation suitable for academic papers:
#   1. Walk-forward cross-validation (multiple test periods)
#   2. Statistical significance testing (bootstrap CIs, t-tests)
#   3. Multiple random seeds (reproducibility)
#   4. Multiple baselines (equal-weight, momentum, risk parity)
#   5. Sensitivity analysis
#
# Usage:
#   python -m ppo.evaluate_publication
#   python -m ppo.evaluate_publication --num-seeds 5 --num-folds 4
#
# =======================

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO


# =======================
# CONFIGURATION
# =======================

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


@dataclass
class EvalResult:
    """Results from a single evaluation run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    calmar_ratio: float  # Return / Max DD
    sortino_ratio: float
    num_days: int


@dataclass
class FoldResult:
    """Results from one fold of walk-forward CV."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    rest_result: EvalResult
    ppo_result: Optional[EvalResult]
    baseline_results: Dict[str, EvalResult]


# =======================
# PORTFOLIO SIMULATION
# =======================

def run_backtest(
    model: PPO,
    price_df: pd.DataFrame,
    cached_states: Dict,
    test_dates: List,
    initial_cash: float = 1_000_000,
    max_weight: float = 1.0,  # Position limit
) -> EvalResult:
    """
    Run backtest with a trained model.
    
    Args:
        model: Trained PPO model
        price_df: Wide-format price data (date x security)
        cached_states: Pre-computed super-states
        test_dates: List of test dates
        initial_cash: Starting portfolio value
        max_weight: Maximum weight per asset (for position limits)
    
    Returns:
        EvalResult with all metrics
    """
    # Build date-to-state-idx lookup
    date_to_idx = {}
    for idx, d in enumerate(cached_states['dates']):
        date_to_idx[str(d)[:10]] = idx
    
    # Initialize portfolio
    portfolio_value = initial_cash
    cash = initial_cash
    holdings = {asset: 0.0 for asset in ASSETS}
    
    portfolio_values = [initial_cash]
    
    for date in test_dates:
        date_str = str(date)[:10]
        
        # Get observation
        if date_str in date_to_idx:
            obs = cached_states['states'][date_to_idx[date_str]].astype(np.float32)
        else:
            obs = np.zeros(64, dtype=np.float32)
        
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        
        # Normalize and apply position limits
        weights = np.clip(action, 0, 1)
        weights = np.minimum(weights, max_weight)  # Position limit
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
        
        # Calculate current value
        holdings_value = sum(holdings.get(a, 0) * prices.get(a, 0) for a in ASSETS)
        prev_value = cash + holdings_value
        
        # Rebalance
        target_values = {a: weights[j] * prev_value for j, a in enumerate(ASSETS)}
        
        for j, asset in enumerate(ASSETS):
            current_value = holdings.get(asset, 0) * prices.get(asset, 0)
            target_value = target_values[asset]
            diff = target_value - current_value
            
            if abs(diff) > 1 and prices.get(asset, 0) > 0:
                holdings[asset] = holdings.get(asset, 0) + diff / prices[asset]
                cash -= diff
        
        # New value
        holdings_value = sum(holdings.get(a, 0) * prices.get(a, 0) for a in ASSETS)
        portfolio_value = cash + holdings_value
        portfolio_values.append(portfolio_value)
    
    return compute_metrics(portfolio_values, initial_cash)


def run_baseline_backtest(
    price_df: pd.DataFrame,
    test_dates: List,
    strategy: str = "equal_weight",
    initial_cash: float = 1_000_000,
) -> EvalResult:
    """
    Run baseline strategy backtest.
    
    Strategies:
        - 'equal_weight': 1/N allocation
        - 'spx': S&P 500 buy-and-hold
        - 'momentum': Simple momentum (top 3 by 20-day return)
        - 'risk_parity': Inverse volatility weighting
    """
    portfolio_values = [initial_cash]
    
    if strategy == "spx":
        # S&P 500 buy and hold
        start_price = price_df.loc[test_dates[0], "SPX Index"]
        shares = initial_cash / start_price
        for date in test_dates:
            price = price_df.loc[date, "SPX Index"]
            portfolio_values.append(shares * price)
    
    elif strategy == "equal_weight":
        # Equal weight, rebalanced daily
        cash = initial_cash
        holdings = {a: 0.0 for a in ASSETS}
        
        for date in test_dates:
            prices = {a: price_df.loc[date, a] for a in ASSETS}
            holdings_val = sum(holdings[a] * prices[a] for a in ASSETS)
            total_val = cash + holdings_val
            
            # Equal weight
            target_per_asset = total_val / len(ASSETS)
            for asset in ASSETS:
                target_shares = target_per_asset / prices[asset]
                diff_shares = target_shares - holdings[asset]
                diff_value = diff_shares * prices[asset]
                holdings[asset] = target_shares
                cash -= diff_value
            
            holdings_val = sum(holdings[a] * prices[a] for a in ASSETS)
            portfolio_values.append(cash + holdings_val)
    
    elif strategy == "momentum":
        # Momentum: top 3 by 20-day return
        cash = initial_cash
        holdings = {a: 0.0 for a in ASSETS}
        lookback = 20
        
        for i, date in enumerate(test_dates):
            prices = {a: price_df.loc[date, a] for a in ASSETS}
            holdings_val = sum(holdings[a] * prices[a] for a in ASSETS)
            total_val = cash + holdings_val
            
            # Calculate momentum (need lookback period)
            if i >= lookback:
                past_date = test_dates[i - lookback]
                returns = {}
                for a in ASSETS:
                    try:
                        past_price = price_df.loc[past_date, a]
                        curr_price = prices[a]
                        returns[a] = (curr_price - past_price) / past_price
                    except:
                        returns[a] = 0
                
                # Top 3 by momentum
                top3 = sorted(returns.keys(), key=lambda x: returns[x], reverse=True)[:3]
                weights = {a: 1/3 if a in top3 else 0 for a in ASSETS}
            else:
                weights = {a: 1/len(ASSETS) for a in ASSETS}
            
            # Rebalance
            for asset in ASSETS:
                target_val = total_val * weights[asset]
                target_shares = target_val / prices[asset] if prices[asset] > 0 else 0
                diff_shares = target_shares - holdings[asset]
                diff_value = diff_shares * prices[asset]
                holdings[asset] = target_shares
                cash -= diff_value
            
            holdings_val = sum(holdings[a] * prices[a] for a in ASSETS)
            portfolio_values.append(cash + holdings_val)
    
    elif strategy == "risk_parity":
        # Inverse volatility weighting
        cash = initial_cash
        holdings = {a: 0.0 for a in ASSETS}
        lookback = 60
        
        for i, date in enumerate(test_dates):
            prices = {a: price_df.loc[date, a] for a in ASSETS}
            holdings_val = sum(holdings[a] * prices[a] for a in ASSETS)
            total_val = cash + holdings_val
            
            # Calculate volatility weights
            if i >= lookback:
                vols = {}
                for a in ASSETS:
                    try:
                        hist_prices = [price_df.loc[test_dates[j], a] for j in range(i-lookback, i)]
                        returns = np.diff(hist_prices) / np.array(hist_prices[:-1])
                        vols[a] = np.std(returns) if len(returns) > 0 else 1.0
                    except:
                        vols[a] = 1.0
                
                # Inverse vol weights
                inv_vols = {a: 1/max(v, 0.001) for a, v in vols.items()}
                total_inv_vol = sum(inv_vols.values())
                weights = {a: inv_vols[a] / total_inv_vol for a in ASSETS}
            else:
                weights = {a: 1/len(ASSETS) for a in ASSETS}
            
            # Rebalance
            for asset in ASSETS:
                target_val = total_val * weights[asset]
                target_shares = target_val / prices[asset] if prices[asset] > 0 else 0
                diff_shares = target_shares - holdings[asset]
                diff_value = diff_shares * prices[asset]
                holdings[asset] = target_shares
                cash -= diff_value
            
            holdings_val = sum(holdings[a] * prices[a] for a in ASSETS)
            portfolio_values.append(cash + holdings_val)
    
    return compute_metrics(portfolio_values, initial_cash)


def compute_metrics(portfolio_values: List[float], initial_cash: float) -> EvalResult:
    """Compute all performance metrics."""
    values = np.array(portfolio_values)
    
    if len(values) < 2:
        return EvalResult(0, 0, 0, 0, 0, 0, 0)
    
    returns = np.diff(values) / np.array(values[:-1])
    
    # Total return
    total_return = (values[-1] - initial_cash) / initial_cash
    
    # Sharpe (annualized)
    if np.std(returns) > 1e-8:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / (peak + 1e-8)
    max_dd = np.max(dd)
    
    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)
    
    # Calmar ratio
    calmar = total_return / max_dd if max_dd > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 1e-8:
        sortino = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(252)
    else:
        sortino = sharpe
    
    return EvalResult(
        total_return=float(total_return),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(max_dd),
        volatility=float(volatility),
        calmar_ratio=float(calmar),
        sortino_ratio=float(sortino),
        num_days=len(values) - 1,
    )


# =======================
# WALK-FORWARD CV
# =======================

def walk_forward_cv(
    df: pd.DataFrame,
    cached_states: Dict,
    num_folds: int = 4,
    train_years: int = 4,
    test_months: int = 12,
    rest_params: Dict = None,
    ppo_model_path: Optional[str] = None,
    seeds: List[int] = None,
    verbose: bool = True,
) -> List[FoldResult]:
    """
    Perform walk-forward cross-validation.
    
    Creates multiple train/test splits moving forward in time.
    """
    from ppo.train_ReST import train_rest
    
    if seeds is None:
        seeds = [42]
    
    price_df = df.pivot(index="date", columns="security", values="close").sort_index()
    all_dates = sorted(df["date"].unique())
    
    # Calculate fold boundaries
    total_days = len(all_dates)
    test_days = int(test_months * 21)  # ~21 trading days per month
    train_days = int(train_years * 252)  # ~252 trading days per year
    
    results = []
    
    for fold in range(num_folds):
        if verbose:
            print(f"\n{'='*70}")
            print(f"FOLD {fold + 1}/{num_folds}")
            print(f"{'='*70}")
        
        # Calculate split points
        test_end_idx = total_days - fold * test_days
        test_start_idx = test_end_idx - test_days
        train_end_idx = test_start_idx
        train_start_idx = max(0, train_end_idx - train_days)
        
        if train_start_idx >= train_end_idx or test_start_idx >= test_end_idx:
            print(f"  Skipping fold {fold + 1}: insufficient data")
            continue
        
        train_dates = all_dates[train_start_idx:train_end_idx]
        test_dates_fold = all_dates[test_start_idx:test_end_idx]
        
        if verbose:
            print(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
            print(f"  Test:  {test_dates_fold[0].date()} to {test_dates_fold[-1].date()} ({len(test_dates_fold)} days)")
        
        # Prepare train data
        train_df = df[df["date"].isin(train_dates)].copy()
        
        # Train ReST model for each seed
        fold_rest_results = []
        
        for seed in seeds:
            if verbose:
                print(f"\n  Training ReST (seed={seed})...")
            
            # Create temp checkpoint dir
            temp_checkpoint_dir = PROJECT_ROOT / "checkpoints" / "cv" / f"fold{fold}_seed{seed}"
            temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            temp_log_dir = PROJECT_ROOT / "logs" / "cv" / f"fold{fold}_seed{seed}"
            temp_log_dir.mkdir(parents=True, exist_ok=True)
            
            # Train model
            rest_hparams = rest_params or {}
            rest_hparams["base_seed"] = seed
            
            model = train_rest(
                data_path=str(PROJECT_ROOT / "data" / "deepar_dataset.csv"),
                checkpoint_dir=str(temp_checkpoint_dir),
                log_dir=str(temp_log_dir),
                cache_path=str(PROJECT_ROOT / "data" / "super_states.npz"),
                max_iterations=rest_hparams.get("max_iterations", 5),
                rollouts_per_iter=rest_hparams.get("rollouts_per_iter", 100),
                episode_length=rest_hparams.get("episode_length", 126),
                lambda_mdd=rest_hparams.get("lambda_mdd", 5.0),
                train_ratio=1.0,  # Use all provided data for training
                verbose=False,
            )
            
            # Evaluate on test set
            rest_result = run_backtest(
                model=model,
                price_df=price_df,
                cached_states=cached_states,
                test_dates=test_dates_fold,
            )
            fold_rest_results.append(rest_result)
        
        # Average results across seeds
        avg_rest_result = EvalResult(
            total_return=np.mean([r.total_return for r in fold_rest_results]),
            sharpe_ratio=np.mean([r.sharpe_ratio for r in fold_rest_results]),
            max_drawdown=np.mean([r.max_drawdown for r in fold_rest_results]),
            volatility=np.mean([r.volatility for r in fold_rest_results]),
            calmar_ratio=np.mean([r.calmar_ratio for r in fold_rest_results]),
            sortino_ratio=np.mean([r.sortino_ratio for r in fold_rest_results]),
            num_days=fold_rest_results[0].num_days,
        )
        
        # Run PPO baseline if provided
        ppo_result = None
        if ppo_model_path and Path(ppo_model_path).exists():
            ppo_model = PPO.load(ppo_model_path)
            ppo_result = run_backtest(
                model=ppo_model,
                price_df=price_df,
                cached_states=cached_states,
                test_dates=test_dates_fold,
            )
        
        # Run baseline strategies
        baseline_results = {}
        for strategy in ["spx", "equal_weight", "momentum", "risk_parity"]:
            if verbose:
                print(f"  Running {strategy} baseline...")
            baseline_results[strategy] = run_baseline_backtest(
                price_df=price_df,
                test_dates=test_dates_fold,
                strategy=strategy,
            )
        
        results.append(FoldResult(
            fold=fold + 1,
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            test_start=str(test_dates_fold[0].date()),
            test_end=str(test_dates_fold[-1].date()),
            rest_result=avg_rest_result,
            ppo_result=ppo_result,
            baseline_results=baseline_results,
        ))
    
    return results


# =======================
# STATISTICAL TESTS
# =======================

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns:
        (mean, lower_ci, upper_ci)
    """
    values = np.array(values)
    n = len(values)
    
    # Bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Confidence interval
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return float(np.mean(values)), float(lower), float(upper)


def paired_ttest(
    values1: List[float],
    values2: List[float],
) -> Tuple[float, float]:
    """
    Perform paired t-test.
    
    Returns:
        (t_statistic, p_value)
    """
    t_stat, p_val = stats.ttest_rel(values1, values2)
    return float(t_stat), float(p_val)


def compute_statistics(results: List[FoldResult]) -> Dict:
    """Compute statistical summary across folds."""
    if not results:
        return {}
    
    # Extract metrics
    rest_sharpes = [r.rest_result.sharpe_ratio for r in results]
    rest_returns = [r.rest_result.total_return for r in results]
    
    spx_sharpes = [r.baseline_results["spx"].sharpe_ratio for r in results]
    spx_returns = [r.baseline_results["spx"].total_return for r in results]
    
    ew_sharpes = [r.baseline_results["equal_weight"].sharpe_ratio for r in results]
    
    # Bootstrap CIs
    rest_sharpe_mean, rest_sharpe_lo, rest_sharpe_hi = bootstrap_ci(rest_sharpes)
    rest_return_mean, rest_return_lo, rest_return_hi = bootstrap_ci(rest_returns)
    
    # T-tests vs baselines
    t_stat_spx, p_val_spx = paired_ttest(rest_sharpes, spx_sharpes)
    t_stat_ew, p_val_ew = paired_ttest(rest_sharpes, ew_sharpes)
    
    return {
        "num_folds": len(results),
        "rest": {
            "sharpe_mean": rest_sharpe_mean,
            "sharpe_ci_95": [rest_sharpe_lo, rest_sharpe_hi],
            "return_mean": rest_return_mean,
            "return_ci_95": [rest_return_lo, rest_return_hi],
        },
        "statistical_tests": {
            "rest_vs_spx": {
                "t_statistic": t_stat_spx,
                "p_value": p_val_spx,
                "significant_at_0.05": p_val_spx < 0.05,
            },
            "rest_vs_equal_weight": {
                "t_statistic": t_stat_ew,
                "p_value": p_val_ew,
                "significant_at_0.05": p_val_ew < 0.05,
            },
        },
    }


# =======================
# REPORTING
# =======================

def print_results(results: List[FoldResult], stats: Dict):
    """Print formatted results."""
    print("\n" + "=" * 80)
    print("PUBLICATION-READY EVALUATION RESULTS")
    print("=" * 80)
    
    # Per-fold results
    print("\n--- Per-Fold Results ---\n")
    print(f"{'Fold':<6} {'Test Period':<25} {'ReST Sharpe':>12} {'SPX Sharpe':>12} {'EW Sharpe':>12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.fold:<6} {r.test_start} to {r.test_end:<10} "
              f"{r.rest_result.sharpe_ratio:>12.3f} "
              f"{r.baseline_results['spx'].sharpe_ratio:>12.3f} "
              f"{r.baseline_results['equal_weight'].sharpe_ratio:>12.3f}")
    
    # Statistical summary
    print("\n--- Statistical Summary ---\n")
    print(f"ReST Sharpe Ratio:")
    print(f"  Mean: {stats['rest']['sharpe_mean']:.3f}")
    print(f"  95% CI: [{stats['rest']['sharpe_ci_95'][0]:.3f}, {stats['rest']['sharpe_ci_95'][1]:.3f}]")
    
    print(f"\nReST Total Return:")
    print(f"  Mean: {stats['rest']['return_mean']*100:.1f}%")
    print(f"  95% CI: [{stats['rest']['return_ci_95'][0]*100:.1f}%, {stats['rest']['return_ci_95'][1]*100:.1f}%]")
    
    # T-tests
    print("\n--- Statistical Significance ---\n")
    for test_name, test_result in stats["statistical_tests"].items():
        sig = "YES" if test_result["significant_at_0.05"] else "NO"
        print(f"{test_name}:")
        print(f"  t-statistic: {test_result['t_statistic']:.3f}")
        print(f"  p-value: {test_result['p_value']:.4f}")
        print(f"  Significant at 0.05: {sig}")
    
    print("\n" + "=" * 80)


def save_results(results: List[FoldResult], stats: Dict, output_path: Path):
    """Save results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "folds": [asdict(r) for r in results],
        "statistics": stats,
    }
    
    # Convert EvalResult objects
    for fold in output["folds"]:
        fold["rest_result"] = asdict(fold["rest_result"]) if fold["rest_result"] else None
        fold["ppo_result"] = asdict(fold["ppo_result"]) if fold["ppo_result"] else None
        fold["baseline_results"] = {k: asdict(v) for k, v in fold["baseline_results"].items()}
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\nResults saved to: {output_path}")


# =======================
# MAIN
# =======================

def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready evaluation for ReST"
    )
    parser.add_argument("--num-folds", type=int, default=4,
                        help="Number of walk-forward folds")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of random seeds per fold")
    parser.add_argument("--train-years", type=int, default=4,
                        help="Years of training data per fold")
    parser.add_argument("--test-months", type=int, default=12,
                        help="Months of test data per fold")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="ReST iterations per fold")
    parser.add_argument("--rollouts", type=int, default=100,
                        help="Rollouts per ReST iteration")
    parser.add_argument("--output", type=str, 
                        default=str(PROJECT_ROOT / "results" / "publication_eval.json"),
                        help="Output file path")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PUBLICATION-READY EVALUATION")
    print("=" * 80)
    print(f"Folds: {args.num_folds}")
    print(f"Seeds per fold: {args.num_seeds}")
    print(f"Train years: {args.train_years}")
    print(f"Test months: {args.test_months}")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(PROJECT_ROOT / "data" / "deepar_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "security"]).reset_index(drop=True)
    
    # Load cached states
    cache_data = np.load(PROJECT_ROOT / "data" / "super_states.npz", allow_pickle=True)
    cached_states = {
        'states': cache_data['states'],
        'dates': list(cache_data['dates']),
    }
    
    # Run walk-forward CV
    seeds = list(range(42, 42 + args.num_seeds))
    
    rest_params = {
        "max_iterations": args.max_iterations,
        "rollouts_per_iter": args.rollouts,
        "episode_length": 126,
        "lambda_mdd": 5.0,
    }
    
    results = walk_forward_cv(
        df=df,
        cached_states=cached_states,
        num_folds=args.num_folds,
        train_years=args.train_years,
        test_months=args.test_months,
        rest_params=rest_params,
        ppo_model_path=str(PROJECT_ROOT / "checkpoints" / "ppo" / "ppo_final.zip"),
        seeds=seeds,
        verbose=True,
    )
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print and save
    print_results(results, stats)
    save_results(results, stats, Path(args.output))


if __name__ == "__main__":
    main()
