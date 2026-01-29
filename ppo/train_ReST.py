# =======================
# File: ppo/train_ReST.py
# =======================
# ReST (Reinforcement learning from Self-Training) Training Script
#
# This script implements ReST for portfolio optimization:
#   - Outer loop: Generate trajectories, score, filter elites
#   - Inner loop: Train PPO on elite trajectories only
#
# IMPORTANT: PPO/ReST training runs strictly on CPU.
# DeepAR (used for pre-computing super-states in precompute_states.py)
# can use GPU separately.
#
# Usage:
#   python -m ppo.train_ReST
#   python -m ppo.train_ReST --max-iterations 10 --rollouts 300
#
# Dependencies: See pyproject.toml (managed by uv sync)
#
# =======================

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

# Ensure project root is in path for local imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from backtesting.core.PortfolioEnv import PortfolioEnv
from ppo.windowed_env import WindowedPortfolioEnv, create_windowed_env
from ppo.elite_replay_env import EliteReplayEnv, create_elite_replay_env
from ppo.rest_trajectory import (
    Trajectory,
    TrajectoryCollector,
    EliteFilter,
    extract_elite_data,
    get_elite_statistics,
)
from ppo.rest_utils import (
    RESTLogger,
    RESTIterationMetrics,
    compute_trajectory_score,
)


# =======================
# DEFAULT HYPERPARAMETERS
# =======================

REST_HYPERPARAMS = {
    # ReST outer loop
    "max_iterations": 10,           # Maximum ReST iterations
    "rollouts_per_iter": 300,       # M = trajectories per iteration
    "episode_length": 126,          # T = 6 months per trajectory
    
    # Trajectory scoring
    "lambda_mdd": 5.0,              # Drawdown penalty weight
    "risk_free_rate": 0.04,         # Annual risk-free rate
    
    # Elite filtering (progressive)
    # Iterations 1-2: 25%, Iterations 3+: 10%
    "min_elites": 30,               # Minimum elite trajectories
    
    # PPO inner loop (per ReST iteration)
    "ppo_epochs": 10,               # PPO epochs per iteration
    "ppo_n_steps": 2048,            # Steps per PPO update
    "ppo_batch_size": 64,           # Mini-batch size
    "ppo_learning_rate": 1e-4,      # Learning rate
    
    # Reproducibility
    "base_seed": 42,                # Base random seed
}

# PPO model hyperparameters (same as train_ppo.py for consistency)
# IMPORTANT: PPO/ReST training runs strictly on CPU.
# DeepAR (used for pre-computing super-states) can use GPU separately.
PPO_MODEL_PARAMS = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [256, 256]},
    "device": "cpu",  # Strictly CPU for PPO/ReST training
    "verbose": 0,
}


# =======================
# DATA LOADING
# =======================

def load_training_data(data_path: str) -> pd.DataFrame:
    """Load and prepare training data."""
    print(f"Loading training data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate columns
    required_cols = ["security", "date", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Sort by date and security
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "security"]).reset_index(drop=True)
    
    print(f"  Loaded {len(df):,} rows")
    print(f"  Securities: {df['security'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def load_cached_states(cache_path: str) -> Optional[Dict]:
    """Load pre-computed super-states if available."""
    if not Path(cache_path).exists():
        print(f"  Cache not found: {cache_path}")
        return None
    
    print(f"Loading cached super-states from: {cache_path}")
    cache_data = np.load(cache_path, allow_pickle=True)
    cached_states = {
        'states': cache_data['states'],
        'dates': list(cache_data['dates']),
    }
    print(f"  Loaded {len(cached_states['dates'])} pre-computed observations")
    return cached_states


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple:
    """Split data into train and test sets by date."""
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * train_ratio)
    
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    
    print(f"\nData split (train_ratio={train_ratio}):")
    print(f"  Training: {len(train_dates)} days ({train_dates[0].date()} to {train_dates[-1].date()})")
    print(f"  Test: {len(test_dates)} days ({test_dates[0].date()} to {test_dates[-1].date()})")
    
    return train_df, test_df, len(train_dates)


# =======================
# EVALUATION
# =======================

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


def _pivot_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form data to wide-form (date x security prices)."""
    pivot = df.pivot(index="date", columns="security", values="close")
    pivot = pivot.sort_index()
    return pivot


def _run_single_eval_window(
    model: PPO,
    price_df: pd.DataFrame,
    cached_states: Dict,
    date_to_idx: Dict[str, int],
    start_date_idx: int,
    episode_length: int,
    initial_cash: float = 1_000_000,
) -> Dict[str, float]:
    """
    Run a single evaluation window with correct price handling.
    
    Uses wide-format prices (date x security) and proper portfolio simulation.
    """
    eval_dates = list(price_df.index[start_date_idx:start_date_idx + episode_length])
    
    # Initialize portfolio
    portfolio_value = initial_cash
    cash = initial_cash
    holdings = {asset: 0.0 for asset in ASSETS}
    
    portfolio_values = [initial_cash]
    
    for date in eval_dates:
        date_str = str(date)[:10]
        
        # Get observation from cache
        if date_str in date_to_idx:
            obs = cached_states['states'][date_to_idx[date_str]].astype(np.float32)
        else:
            obs = np.zeros(64, dtype=np.float32)
        
        # Get action from model (deterministic for eval)
        action, _ = model.predict(obs, deterministic=True)
        
        # Normalize weights
        weights = np.clip(action, 0, 1)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(len(ASSETS)) / len(ASSETS)
        
        # Get prices for this date
        try:
            prices = {asset: price_df.loc[date, asset] for asset in ASSETS}
        except KeyError:
            continue
        
        # Calculate current portfolio value
        holdings_value = sum(holdings.get(a, 0) * prices.get(a, 0) for a in ASSETS)
        portfolio_value = cash + holdings_value
        
        # Rebalance to target weights
        target_values = {a: weights[j] * portfolio_value for j, a in enumerate(ASSETS)}
        
        # Execute trades (simplified)
        for j, asset in enumerate(ASSETS):
            current_value = holdings.get(asset, 0) * prices.get(asset, 0)
            target_value = target_values[asset]
            diff = target_value - current_value
            
            if abs(diff) > 1 and prices.get(asset, 0) > 0:
                shares_to_trade = diff / prices[asset]
                holdings[asset] = holdings.get(asset, 0) + shares_to_trade
                cash -= diff
        
        # Record portfolio value after trades
        holdings_value = sum(holdings.get(a, 0) * prices.get(a, 0) for a in ASSETS)
        portfolio_value = cash + holdings_value
        portfolio_values.append(portfolio_value)
    
    # Compute metrics
    values = np.array(portfolio_values)
    if len(values) < 2:
        return {"total_return": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    
    returns = np.diff(values) / (np.array(values[:-1]) + 1e-8)
    total_ret = (values[-1] - values[0]) / (values[0] + 1e-8)
    
    # Sharpe (annualized)
    if len(returns) > 1 and np.std(returns) > 1e-8:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / (peak + 1e-8)
    max_dd = np.max(dd)
    
    return {"total_return": total_ret, "sharpe": sharpe, "max_dd": max_dd}


def evaluate_on_test(
    model: PPO,
    test_df: pd.DataFrame,
    cached_states: Optional[Dict],
    num_eval_windows: int = 50,
    episode_length: int = 63,  # 3 months for faster eval
    seed: int = 12345,  # Fixed seed for comparability
) -> Dict[str, float]:
    """
    Evaluate model on held-out test data using correct price handling.
    
    Uses wide-format prices (date x security) like backtest_ppo.py,
    avoiding the DataHandler bug with long-format data.
    
    Args:
        model: Trained PPO model
        test_df: Test DataFrame (long format)
        cached_states: Pre-computed states
        num_eval_windows: Number of evaluation windows
        episode_length: Steps per evaluation episode
        seed: Fixed seed for reproducibility
        
    Returns:
        Dictionary of evaluation metrics
    """
    if cached_states is None:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "num_eval_episodes": 0,
        }
    
    rng = np.random.RandomState(seed)
    
    # Pivot to wide format (date x security)
    price_df = _pivot_prices(test_df)
    
    # Build date-to-state index lookup
    date_to_idx = {}
    for idx, d in enumerate(cached_states['dates']):
        date_to_idx[str(d)[:10]] = idx
    
    # Get valid start indices
    num_dates = len(price_df)
    max_start = max(1, num_dates - episode_length)
    
    # Sample evaluation windows (fixed for comparability)
    num_samples = min(num_eval_windows, max_start)
    eval_starts = rng.choice(max_start, size=num_samples, replace=False)
    
    # Run evaluations
    total_returns = []
    sharpes = []
    max_drawdowns = []
    
    for start_idx in eval_starts:
        metrics = _run_single_eval_window(
            model=model,
            price_df=price_df,
            cached_states=cached_states,
            date_to_idx=date_to_idx,
            start_date_idx=start_idx,
            episode_length=episode_length,
        )
        total_returns.append(metrics["total_return"])
        sharpes.append(metrics["sharpe"])
        max_drawdowns.append(metrics["max_dd"])
    
    return {
        "total_return": float(np.mean(total_returns)),
        "sharpe_ratio": float(np.mean(sharpes)),
        "max_drawdown": float(np.mean(max_drawdowns)),
        "num_eval_episodes": len(eval_starts),
    }


# =======================
# MAIN TRAINING LOOP
# =======================

def train_rest(
    data_path: str,
    checkpoint_dir: str,
    log_dir: str,
    cache_path: Optional[str] = None,
    # ReST hyperparameters
    max_iterations: int = 10,
    rollouts_per_iter: int = 300,
    episode_length: int = 126,
    lambda_mdd: float = 5.0,
    # PPO hyperparameters
    ppo_epochs: int = 10,
    # Seeds
    base_seed: int = 42,
    # Evaluation
    eval_frequency: int = 1,  # Evaluate every N iterations
    full_eval_frequency: int = 3,  # Full eval every N iterations
):
    """
    Main ReST training loop.
    
    Args:
        data_path: Path to training data CSV
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for logs
        cache_path: Path to pre-computed super-states
        max_iterations: Maximum ReST iterations
        rollouts_per_iter: Trajectories per iteration (M)
        episode_length: Steps per trajectory (T)
        lambda_mdd: Drawdown penalty weight
        ppo_epochs: PPO epochs per ReST iteration
        base_seed: Base random seed
        eval_frequency: Mini-eval every N iterations
        full_eval_frequency: Full eval every N iterations
    """
    print("\n" + "=" * 70)
    print("ReST Training for Portfolio Optimization")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max iterations: {max_iterations}")
    print(f"Rollouts per iteration: {rollouts_per_iter}")
    print(f"Episode length: {episode_length} days")
    print(f"Lambda (MDD penalty): {lambda_mdd}")
    print("=" * 70 + "\n")
    
    # Create directories
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = RESTLogger(log_dir=str(log_dir))
    
    # Load data
    df = load_training_data(data_path)
    cached_states = load_cached_states(cache_path) if cache_path else None
    
    # Split train/test
    train_df, test_df, num_train_days = split_train_test(df, train_ratio=0.8)
    
    # Create windowed environment for trajectory collection
    train_env = create_windowed_env(
        df=train_df,
        cached_states=cached_states,
        episode_length=episode_length,
        reward_mode="sharpe",
        risk_free_rate=0.04,
    )
    
    # Initialize PPO model
    print("\nInitializing PPO model...")
    
    # Create a dummy env for PPO initialization
    dummy_env = DummyVecEnv([lambda: Monitor(PortfolioEnv(
        df=train_df,
        cached_states=cached_states,
        reward_mode="sharpe",
    ))])
    
    model = PPO(
        policy="MlpPolicy",
        env=dummy_env,
        **PPO_MODEL_PARAMS,
    )
    print(f"  Model initialized with architecture: {PPO_MODEL_PARAMS['policy_kwargs']}")
    
    # Create trajectory collector with correct price handling
    collector = TrajectoryCollector(
        df=train_df,
        cached_states=cached_states,
        policy=model,
        episode_length=episode_length,
    )
    
    elite_filter = EliteFilter(min_elites=30)
    
    # Track best model
    best_eval_score = float('-inf')
    
    # =======================
    # ReST OUTER LOOP
    # =======================
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ReST Iteration {iteration}/{max_iterations}")
        print(f"{'='*70}")
        
        iteration_start = datetime.now()
        
        # Set seed for reproducibility
        iter_seed = base_seed + iteration
        collector.set_seed(iter_seed)
        
        # -----------------------
        # 1. GENERATE TRAJECTORIES
        # -----------------------
        print(f"\n[1/5] Generating {rollouts_per_iter} trajectories...")
        
        trajectories = collector.collect(
            num_rollouts=rollouts_per_iter,
            lambda_mdd=lambda_mdd,
            risk_free_rate=0.04,
            verbose=True,
        )
        
        all_scores = [t.score for t in trajectories]
        print(f"  Mean score: {np.mean(all_scores):.3f} +/- {np.std(all_scores):.3f}")
        
        # -----------------------
        # 2. FILTER ELITE SET
        # -----------------------
        print(f"\n[2/5] Filtering elite trajectories...")
        
        elites, percentile_used = elite_filter.filter(trajectories, iteration=iteration)
        
        elite_stats = get_elite_statistics(elites)
        print(f"  Selected {len(elites)} elites ({percentile_used*100:.0f}% threshold)")
        print(f"  Elite mean score: {elite_stats['mean_score']:.3f}")
        print(f"  Elite mean Sharpe: {elite_stats['mean_sharpe']:.3f}")
        print(f"  Elite mean MDD: {elite_stats['mean_mdd']:.3f}")
        
        # -----------------------
        # 3. CREATE REPLAY ENV
        # -----------------------
        print(f"\n[3/5] Creating replay environment...")
        
        replay_env = create_elite_replay_env(
            elite_trajectories=elites,
            shuffle=True,
            seed=iter_seed,
        )
        
        # Wrap for SB3
        replay_vec_env = DummyVecEnv([lambda: replay_env])
        
        total_elite_steps = replay_env.get_total_timesteps()
        print(f"  Total elite steps: {total_elite_steps}")
        
        # -----------------------
        # 4. PPO TRAINING ON ELITES
        # -----------------------
        print(f"\n[4/5] Training PPO on elite trajectories...")
        
        # Update model's environment to replay env
        model.set_env(replay_vec_env)
        
        # Train PPO
        # We want to go through the elite data multiple times (epochs)
        # n_steps * n_epochs gives us the effective training
        train_timesteps = total_elite_steps * ppo_epochs
        
        model.learn(
            total_timesteps=train_timesteps,
            reset_num_timesteps=True,
            progress_bar=True,
        )
        
        print(f"  Completed PPO training ({train_timesteps} timesteps)")
        
        # -----------------------
        # 5. EVALUATION & LOGGING
        # -----------------------
        print(f"\n[5/5] Evaluating and logging...")
        
        # Determine eval type
        do_full_eval = (iteration % full_eval_frequency == 0) or (iteration == max_iterations)
        num_eval_windows = 50 if do_full_eval else 30
        eval_episode_length = episode_length if do_full_eval else 63  # 3 months for mini-eval
        
        # Run evaluation
        eval_metrics = evaluate_on_test(
            model=model,
            test_df=test_df,
            cached_states=cached_states,
            num_eval_windows=num_eval_windows,
            episode_length=eval_episode_length,
            seed=12345,  # Fixed seed for comparability
        )
        
        print(f"  Held-out evaluation ({num_eval_windows} windows):")
        print(f"    Sharpe: {eval_metrics['sharpe_ratio']:.3f}")
        print(f"    Return: {eval_metrics['total_return']*100:.2f}%")
        print(f"    Max DD: {eval_metrics['max_drawdown']*100:.2f}%")
        
        # Log iteration
        logger.log_iteration(
            iteration=iteration,
            all_scores=all_scores,
            elite_scores=[t.score for t in elites],
            elite_sharpes=[t.sharpe for t in elites],
            elite_mdds=[t.max_drawdown for t in elites],
            elite_actions=[t.actions for t in elites],
            elite_start_indices=[t.start_idx for t in elites],
            elite_percentile=percentile_used,
            eval_metrics=eval_metrics,
        )
        
        # -----------------------
        # CHECKPOINTING
        # -----------------------
        
        # Save iteration checkpoint
        iter_checkpoint_path = checkpoint_dir / f"rest_iter_{iteration:02d}.zip"
        model.save(str(iter_checkpoint_path))
        
        iter_metrics_path = checkpoint_dir / f"rest_iter_{iteration:02d}_metrics.json"
        with open(iter_metrics_path, 'w') as f:
            json.dump({
                "iteration": iteration,
                "eval_metrics": eval_metrics,
                "elite_stats": {
                    "num_elites": len(elites),
                    "mean_score": elite_stats['mean_score'],
                    "mean_sharpe": elite_stats['mean_sharpe'],
                    "mean_mdd": elite_stats['mean_mdd'],
                },
                "all_scores_mean": float(np.mean(all_scores)),
                "all_scores_std": float(np.std(all_scores)),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        
        print(f"  Saved checkpoint: {iter_checkpoint_path}")
        
        # Check if this is best model
        eval_score = eval_metrics['sharpe_ratio']
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            
            # Save best model
            best_model_path = checkpoint_dir / "best_model.zip"
            model.save(str(best_model_path))
            
            best_metrics_path = checkpoint_dir / "best_metrics.json"
            with open(best_metrics_path, 'w') as f:
                json.dump({
                    "iteration": iteration,
                    "eval_metrics": eval_metrics,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)
            
            print(f"  New best model! Sharpe: {eval_score:.3f}")
        
        # Check failure indicators
        metrics = logger.iteration_metrics[-1]
        failures = logger.check_failure_indicators(metrics)
        if failures:
            print(f"\n  FAILURE INDICATORS DETECTED:")
            for f in failures:
                print(f"    - {f}")
        
        iteration_time = (datetime.now() - iteration_start).total_seconds()
        print(f"\n  Iteration completed in {iteration_time:.1f}s")
        
        # Update collector's policy reference (in case model was updated)
        collector.policy = model
    
    # =======================
    # TRAINING COMPLETE
    # =======================
    
    # Save final model
    final_model_path = checkpoint_dir / "rest_final.zip"
    model.save(str(final_model_path))
    
    # Save full log
    logger.save_full_log()
    
    print("\n" + "=" * 70)
    print("ReST Training Complete!")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best eval Sharpe: {best_eval_score:.3f}")
    print(f"Final model: {final_model_path}")
    print(f"Best model: {checkpoint_dir / 'best_model.zip'}")
    print(f"Logs: {log_dir}")
    print("=" * 70)
    
    return model


# =======================
# MAIN ENTRY POINT
# =======================

def main():
    """Parse arguments and run ReST training."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent using ReST for portfolio optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument(
        "--data", type=str,
        default=str(PROJECT_ROOT / "data" / "deepar_dataset.csv"),
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--cache", type=str,
        default=str(PROJECT_ROOT / "data" / "super_states.npz"),
        help="Path to pre-computed super-states",
    )
    
    # Output paths
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "rest"),
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=str,
        default=str(PROJECT_ROOT / "logs" / "rest"),
        help="Directory for logs",
    )
    
    # ReST hyperparameters
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum ReST iterations",
    )
    parser.add_argument(
        "--rollouts", type=int, default=300,
        help="Trajectories per iteration (M)",
    )
    parser.add_argument(
        "--episode-length", type=int, default=126,
        help="Steps per trajectory (T = ~6 months)",
    )
    parser.add_argument(
        "--lambda-mdd", type=float, default=5.0,
        help="Drawdown penalty weight",
    )
    
    # PPO hyperparameters
    parser.add_argument(
        "--ppo-epochs", type=int, default=10,
        help="PPO epochs per ReST iteration",
    )
    
    # Seeds
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed",
    )
    
    args = parser.parse_args()
    
    train_rest(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        cache_path=args.cache,
        max_iterations=args.max_iterations,
        rollouts_per_iter=args.rollouts,
        episode_length=args.episode_length,
        lambda_mdd=args.lambda_mdd,
        ppo_epochs=args.ppo_epochs,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
