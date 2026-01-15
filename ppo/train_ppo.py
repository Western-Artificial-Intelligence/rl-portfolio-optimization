# =======================
# File: ppo/train_ppo.py
# =======================
# PPO Training Script for Portfolio Optimization
# 
# This script trains a Proximal Policy Optimization (PPO) agent
# to manage a portfolio using the Super-State observations from
# DeepAR forecasts + FRED macro data.
#
# Usage:
#   python -m ppo.train_ppo
#   python -m ppo.train_ppo --timesteps 500000 --checkpoint-freq 50000
# =======================

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure project root is in path for local imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from backtesting.core.PortfolioEnv import PortfolioEnv

# =======================
# HYPERPARAMETERS
# =======================
# These are tuned for financial time-series with 64-dim Super-State observations.
# Reference: SB3 docs, "RL for Finance" best practices

PPO_HYPERPARAMS = {
    # Learning rate: Lower for noisy financial data (default is 3e-4)
    # Start conservative to avoid divergence
    "learning_rate": 1e-4,
    
    # Number of steps to collect before each policy update
    # Higher = more stable but slower updates
    # 2048 is standard, works well for episodic tasks
    "n_steps": 2048,
    
    # Mini-batch size for optimization
    # Smaller batches (64-128) help with noisy financial data
    # Must divide n_steps evenly: 2048 / 64 = 32 mini-batches
    "batch_size": 64,
    
    # Number of epochs when optimizing the surrogate loss
    # 10 is standard, lower values (5) for more stable but slower learning
    "n_epochs": 10,
    
    # Discount factor (gamma): How much to value future rewards
    # Higher (0.99) = more long-term focus, good for finance
    "gamma": 0.99,
    
    # GAE lambda: Trade-off between bias and variance in advantage estimation
    # 0.95 is standard, slightly lower for noisy data
    "gae_lambda": 0.95,
    
    # Clipping parameter (epsilon): How much policy can change per update
    # 0.2 is standard, lower (0.1) for more stable but slower updates
    "clip_range": 0.2,
    
    # Clipping for value function (optional, helps stability)
    "clip_range_vf": 0.2,
    
    # Entropy coefficient: Encourages exploration
    # Higher (0.01-0.02) for more exploration, lower (0.0) for exploitation
    # Start with 0.01 to encourage exploration of different portfolio weights
    "ent_coef": 0.01,
    
    # Value function coefficient in loss
    "vf_coef": 0.5,
    
    # Max gradient norm for clipping (prevents exploding gradients)
    "max_grad_norm": 0.5,
    
    # Network architecture: Shared MLP for policy and value
    # Two hidden layers of 256 units each
    "policy_kwargs": {
        "net_arch": [256, 256],
    },
    
    # Use GPU if available
    "device": "auto",
    
    # Verbosity level (1 = training info)
    "verbose": 1,
}


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate the training dataset.
    
    Args:
        data_path: Path to deepar_dataset.csv
        
    Returns:
        DataFrame with columns ['security', 'date', 'close', ...]
    """
    print(f"Loading training data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_cols = ["security", "date", "close"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by security and date for consistent ordering
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["security", "date"]).reset_index(drop=True)
    
    print(f"  Loaded {len(df):,} rows")
    print(f"  Securities: {df['security'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def create_environment(
    df: pd.DataFrame,
    use_super_state: bool = True,
    cached_states: dict = None,
) -> PortfolioEnv:
    """
    Create and wrap the Portfolio Environment.
    
    Args:
        df: Training DataFrame
        use_super_state: Whether to use 64-dim Super-State observations
        cached_states: Pre-computed super-states dict (optional, for fast training)
        
    Returns:
        Wrapped environment
    """
    env = PortfolioEnv(
        df=df,
        use_super_state=use_super_state,
        reward_mode="sharpe",  # Optimize for risk-adjusted returns
        reward_window=30,      # 30-day rolling window for Sharpe
        initial_cash=1_000_000,
        context_length=60,     # 60 days of history for DeepAR
        cached_states=cached_states,  # Use pre-computed states if provided
    )
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    return env


def create_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    eval_env: PortfolioEnv,
    checkpoint_freq: int = 100_000,
) -> CallbackList:
    """
    Create training callbacks for checkpointing and evaluation.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        eval_env: Environment for evaluation
        checkpoint_freq: Save checkpoint every N timesteps
        
    Returns:
        CallbackList with all callbacks
    """
    # Checkpoint callback: Save model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,  # PPO doesn't use replay buffer
        save_vecnormalize=True,
    )
    
    # Eval callback: Evaluate on held-out data every N steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=checkpoint_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    return CallbackList([checkpoint_callback, eval_callback])


def train(
    data_path: str,
    checkpoint_dir: str,
    log_dir: str,
    total_timesteps: int = 1_000_000,
    checkpoint_freq: int = 100_000,
    resume_from: str = None,
    cache_path: str = None,  # Path to pre-computed super-states
):
    """
    Main training loop for PPO agent.
    
    Args:
        data_path: Path to training data CSV
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        total_timesteps: Total training timesteps
        checkpoint_freq: Checkpoint frequency
        resume_from: Path to checkpoint to resume from (optional)
        cache_path: Path to pre-computed super-states .npz file (optional)
    """
    print("\n" + "=" * 60)
    print("PPO Training for Portfolio Optimization")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Checkpoint frequency: {checkpoint_freq:,}")
    print()
    
    # Create directories (pre-create TensorBoard run dir to avoid OneDrive issues)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create specific run directory for TensorBoard (avoids OneDrive sync issues)
    run_name = f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_log_dir = Path(log_dir) / run_name
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard log directory: {tb_log_dir}")
    
    # Load cached super-states if provided
    cached_states = None
    if cache_path and Path(cache_path).exists():
        print(f"\n✓ Loading cached super-states from: {cache_path}")
        cache_data = np.load(cache_path, allow_pickle=True)
        cached_states = {
            'states': cache_data['states'],
            'dates': cache_data['dates'],
        }
        print(f"  Loaded {len(cached_states['dates'])} pre-computed observations")
        print(f"  Shape: {cached_states['states'].shape}")
    elif cache_path:
        print(f"\n⚠️ Cache file not found: {cache_path}")
        print("  Run: python -m ppo.precompute_states")
        print("  Falling back to slow mode (computing DeepAR each step)...")
    
    # Load data
    df = load_training_data(data_path)
    
    # Split data for training and evaluation (80/20)
    dates = df["date"].unique()
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    eval_dates = dates[split_idx:]
    
    train_df = df[df["date"].isin(train_dates)]
    eval_df = df[df["date"].isin(eval_dates)]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_dates)} days ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Evaluation: {len(eval_dates)} days ({eval_df['date'].min()} to {eval_df['date'].max()})")
    
    # Create environments (pass cached_states for fast training)
    print("\nCreating environments...")
    train_env = create_environment(train_df, use_super_state=True, cached_states=cached_states)
    eval_env = create_environment(eval_df, use_super_state=True, cached_states=cached_states)
    
    # Wrap in DummyVecEnv for SB3 compatibility
    train_env = DummyVecEnv([lambda: train_env])
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create or load model
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming training from: {resume_from}")
        model = PPO.load(resume_from, env=train_env)
    else:
        print("\nInitializing new PPO model...")
        print(f"Hyperparameters:")
        for key, value in PPO_HYPERPARAMS.items():
            print(f"  {key}: {value}")
        
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            tensorboard_log=None,  # Disabled - OneDrive conflicts with TensorBoard
            **PPO_HYPERPARAMS,
        )
    
    # Print model architecture
    print(f"\nModel architecture:")
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")
    
    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=str(tb_log_dir),  # Use pre-created dir
        eval_env=eval_env,
        checkpoint_freq=checkpoint_freq,
    )
    
    # Train!
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False if resume_from else True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Save final model
    final_model_path = Path(checkpoint_dir) / "ppo_final.zip"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final model: {final_model_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir={log_dir}")
    

def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for portfolio optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "deepar_dataset.csv"),
        help="Path to training data CSV",
    )
    
    # Output paths
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "ppo"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "ppo"),
        help="Directory for TensorBoard logs",
    )
    
    # Training settings
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="Save checkpoint every N timesteps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    # Caching options
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use pre-computed super-states for 100x faster training",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "super_states.npz"),
        help="Path to pre-computed super-states file",
    )
    
    args = parser.parse_args()
    
    # Determine cache path
    cache_path = args.cache_path if args.use_cache else None
    
    train(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume,
        cache_path=cache_path,
    )


if __name__ == "__main__":
    main()
