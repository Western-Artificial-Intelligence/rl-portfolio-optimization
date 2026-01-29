# =======================
# File: ppo/windowed_env.py
# =======================
# Windowed Portfolio Environment Wrapper for ReST
#
# This is a thin wrapper around PortfolioEnv that adds:
#   - Windowed episode reset (start at specific index)
#   - Episode length limiting
#
# IMPORTANT: PortfolioEnv.step() return signature is UNCHANGED.
#
# Note: CPU-based environment (no GPU operations).
# Dependencies: gymnasium, numpy, pandas (all in pyproject.toml via stable-baselines3)
#
# =======================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.core.PortfolioEnv import PortfolioEnv


class WindowedPortfolioEnv(gym.Env):
    """
    Thin wrapper around PortfolioEnv for windowed episode reset.
    
    Adds capability to reset to specific windows of training data,
    which is required for ReST trajectory generation.
    
    Key features:
    - reset(options={"start_idx": int, "episode_length": int})
    - step() returns same signature as PortfolioEnv (unchanged)
    - Episode terminates after episode_length steps
    
    This keeps the underlying PortfolioEnv interface stable for
    team compatibility.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        cached_states: Optional[Dict] = None,
        default_episode_length: int = 126,
        **env_kwargs,
    ):
        """
        Initialize the windowed environment.
        
        Args:
            df: Full training DataFrame with columns ['security', 'date', 'close', ...]
            cached_states: Pre-computed super-states dict (optional)
            default_episode_length: Default episode length if not specified in reset
            **env_kwargs: Additional arguments passed to PortfolioEnv
        """
        super().__init__()
        
        # Store full data
        self.full_df = df.copy()
        self.full_df['date'] = pd.to_datetime(self.full_df['date'])
        self.full_df = self.full_df.sort_values(['date', 'security']).reset_index(drop=True)
        
        # Get unique dates
        self.unique_dates = sorted(self.full_df['date'].unique())
        self.num_dates = len(self.unique_dates)
        
        # Store cached states and env kwargs
        self.cached_states = cached_states
        self.env_kwargs = env_kwargs
        self.default_episode_length = default_episode_length
        
        # Current episode settings
        self.current_start_idx = 0
        self.current_episode_length = default_episode_length
        self.current_step = 0
        
        # Create underlying environment (will be recreated on reset)
        self._create_env(0, default_episode_length)
        
        # Copy spaces from underlying env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def _create_env(self, start_idx: int, episode_length: int):
        """
        Create underlying PortfolioEnv for specific window.
        
        Args:
            start_idx: Starting date index
            episode_length: Number of trading days in episode
        """
        # Clamp indices
        start_idx = max(0, min(start_idx, self.num_dates - episode_length - 1))
        end_idx = min(start_idx + episode_length, self.num_dates)
        
        # Get date range
        start_date = self.unique_dates[start_idx]
        end_date = self.unique_dates[end_idx - 1] if end_idx <= self.num_dates else self.unique_dates[-1]
        
        # Filter data to window
        window_df = self.full_df[
            (self.full_df['date'] >= start_date) &
            (self.full_df['date'] <= end_date)
        ].copy().reset_index(drop=True)
        
        # Create environment
        self.env = PortfolioEnv(
            df=window_df,
            cached_states=self.cached_states,
            **self.env_kwargs,
        )
        
        # Store settings
        self.current_start_idx = start_idx
        self.current_episode_length = episode_length
        self.current_step = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment, optionally to specific window.
        
        Args:
            seed: Random seed (passed to underlying env)
            options: Dictionary with optional keys:
                - 'start_idx': Starting date index (int)
                - 'episode_length': Episode length (int)
                
        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)
        
        # Parse options
        if options is None:
            options = {}
        
        start_idx = options.get('start_idx', self.current_start_idx)
        episode_length = options.get('episode_length', self.default_episode_length)
        
        # Recreate environment for new window
        self._create_env(start_idx, episode_length)
        
        # Reset underlying env
        obs, info = self.env.reset(seed=seed)
        
        # Add window info
        info['start_idx'] = self.current_start_idx
        info['episode_length'] = self.current_episode_length
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Returns:
            Same signature as PortfolioEnv.step():
            (observation, reward, terminated, truncated, info)
        """
        # Delegate to underlying env
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track step count
        self.current_step += 1
        
        # Truncate if we've exceeded episode length
        if self.current_step >= self.current_episode_length:
            truncated = True
        
        # Add step info
        info['step_in_episode'] = self.current_step
        info['start_idx'] = self.current_start_idx
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        return self.env.render(mode)
    
    @property
    def unwrapped(self):
        """Return the underlying PortfolioEnv."""
        return self.env
    
    @property
    def portfolio_value(self) -> float:
        """Get current portfolio value from underlying env."""
        return self.env.portfolio_value
    
    def get_portfolio_metrics(self) -> Dict:
        """Get portfolio metrics from underlying env."""
        return self.env.get_portfolio_metrics()


def create_windowed_env(
    df: pd.DataFrame,
    cached_states: Optional[Dict] = None,
    episode_length: int = 126,
    reward_mode: str = "sharpe",
    reward_window: int = 30,
    initial_cash: float = 1_000_000,
    risk_free_rate: float = 0.04,
    use_super_state: bool = True,
    context_length: int = 60,
) -> WindowedPortfolioEnv:
    """
    Factory function to create a windowed environment.
    
    Args:
        df: Training DataFrame
        cached_states: Pre-computed super-states
        episode_length: Default episode length
        reward_mode: Reward computation mode
        reward_window: Window for rolling statistics
        initial_cash: Starting portfolio value
        risk_free_rate: Annual risk-free rate
        use_super_state: Whether to use 64-dim observations
        context_length: DeepAR context length
        
    Returns:
        WindowedPortfolioEnv instance
    """
    return WindowedPortfolioEnv(
        df=df,
        cached_states=cached_states,
        default_episode_length=episode_length,
        reward_mode=reward_mode,
        reward_window=reward_window,
        initial_cash=initial_cash,
        risk_free_rate=risk_free_rate,
        use_super_state=use_super_state,
        context_length=context_length,
    )
