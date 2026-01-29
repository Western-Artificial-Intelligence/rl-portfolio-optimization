# =======================
# File: ppo/rest_trajectory.py
# =======================
# Trajectory management for ReST (Reinforcement learning from Self-Training)
#
# Contains:
#   - Trajectory dataclass for storing rollout data
#   - TrajectoryCollector for generating rollouts (FIXED version with correct prices)
#   - EliteFilter for selecting top trajectories
#
# Note: All computations in this file are CPU-based (numpy).
# Dependencies: numpy, pandas, dataclasses, typing, pathlib
#
# =======================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from .rest_utils import (
    compute_trajectory_score,
    compute_trajectory_sharpe,
    compute_max_drawdown,
    compute_returns_from_values,
)

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


# =======================
# TRAJECTORY DATA STRUCTURE
# =======================

@dataclass
class Trajectory:
    """
    Complete record of a single rollout/episode.
    
    Contains all data needed for:
    - Elite scoring: portfolio_values, returns
    - PPO training: states, actions, rewards
    - Logging: start_idx for regime analysis
    """
    
    # Core trajectory data
    states: List[np.ndarray] = field(default_factory=list)       # Observations [T, 64]
    actions: List[np.ndarray] = field(default_factory=list)      # Portfolio weights [T, 9]
    rewards: List[float] = field(default_factory=list)           # Step rewards (for PPO)
    portfolio_values: List[float] = field(default_factory=list)  # For MDD calculation
    
    # Metadata
    start_idx: int = 0                    # Start index in training data
    episode_length: int = 0               # Actual episode length
    
    # Computed scores (populated after trajectory completes)
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    score: float = 0.0  # R(tau) = Sharpe - lambda*MDD
    
    def compute_score(
        self,
        lambda_mdd: float = 5.0,
        risk_free_rate: float = 0.04,
    ) -> float:
        """
        Compute trajectory score R(tau) = Sharpe(tau) - lambda*MDD(tau).
        
        Args:
            lambda_mdd: Drawdown penalty weight
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Trajectory score
        """
        if len(self.portfolio_values) < 2:
            self.sharpe = 0.0
            self.max_drawdown = 0.0
            self.score = 0.0
            return 0.0
        
        # Compute returns from portfolio values
        values = np.array(self.portfolio_values)
        returns = compute_returns_from_values(values)
        
        # Compute components
        self.sharpe = compute_trajectory_sharpe(returns, risk_free_rate)
        self.max_drawdown = compute_max_drawdown(values)
        
        # Final score
        self.score = self.sharpe - lambda_mdd * self.max_drawdown
        return self.score
    
    def get_training_tuples(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Extract (state, action, reward) tuples for PPO training.
        
        Returns:
            List of (s_t, a_t, r_t) tuples
        """
        tuples = []
        min_len = min(len(self.states), len(self.actions), len(self.rewards))
        for i in range(min_len):
            tuples.append((
                self.states[i],
                self.actions[i],
                self.rewards[i],
            ))
        return tuples
    
    def __len__(self) -> int:
        return len(self.states)


# =======================
# TRAJECTORY COLLECTOR (FIXED)
# =======================

class TrajectoryCollector:
    """
    Collects trajectories using correct wide-format price handling.
    
    FIXED VERSION: Uses pivoted price data (date x security) instead of
    the buggy DataHandler which iterates row-by-row through long-format data.
    
    For ReST:
    - Samples random start windows from training data
    - Runs policy stochastically for T steps
    - Records full trajectory (s, a, r, portfolio_values)
    - Uses correct portfolio simulation with per-asset prices
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        cached_states: Dict,
        policy,
        episode_length: int = 126,
        initial_cash: float = 1_000_000,
        seed: int = None,
    ):
        """
        Initialize the trajectory collector with correct price handling.
        
        Args:
            df: Training DataFrame (long format)
            cached_states: Pre-computed super-states dict with 'states' and 'dates'
            policy: SB3 PPO model for action selection
            episode_length: T = steps per episode (default 126 = 6 months)
            initial_cash: Starting portfolio value
            seed: Random seed for reproducibility
        """
        self.policy = policy
        self.episode_length = episode_length
        self.initial_cash = initial_cash
        self.rng = np.random.RandomState(seed)
        
        # Pivot to wide format (date x security)
        self.price_df = df.pivot(index="date", columns="security", values="close").sort_index()
        self.dates = list(self.price_df.index)
        self.num_dates = len(self.dates)
        
        # Build date-to-state-index lookup
        self.cached_states = cached_states
        self.date_to_idx = {}
        if cached_states:
            for idx, d in enumerate(cached_states['dates']):
                self.date_to_idx[str(d)[:10]] = idx
        
        # Valid start range
        self.max_start_idx = max(1, self.num_dates - episode_length)
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible sampling."""
        self.rng = np.random.RandomState(seed)
    
    def _get_observation(self, date) -> np.ndarray:
        """Get observation from cached states for a given date."""
        date_str = str(date)[:10]
        if date_str in self.date_to_idx:
            return self.cached_states['states'][self.date_to_idx[date_str]].astype(np.float32)
        return np.zeros(64, dtype=np.float32)
    
    def _compute_reward(
        self, 
        prev_value: float, 
        curr_value: float,
        risk_free_rate: float = 0.04,
    ) -> float:
        """
        Compute step reward (Sharpe-style reward).
        
        Uses daily excess return normalized by volatility estimate.
        """
        if prev_value <= 0:
            return 0.0
        
        daily_return = (curr_value - prev_value) / prev_value
        daily_rf = risk_free_rate / 252
        excess_return = daily_return - daily_rf
        
        # Simple volatility estimate (can be improved)
        vol_estimate = 0.02  # ~32% annualized vol assumption
        
        return excess_return / vol_estimate
    
    def collect_single(self, start_idx: Optional[int] = None) -> Trajectory:
        """
        Collect a single trajectory with correct price handling.
        
        Args:
            start_idx: Specific start index (or None for random)
            
        Returns:
            Completed Trajectory object
        """
        # Sample start index if not provided
        if start_idx is None:
            start_idx = self.rng.randint(0, self.max_start_idx)
        
        # Create trajectory
        trajectory = Trajectory(start_idx=start_idx)
        
        # Get episode dates
        end_idx = min(start_idx + self.episode_length, self.num_dates)
        episode_dates = self.dates[start_idx:end_idx]
        
        # Initialize portfolio
        portfolio_value = self.initial_cash
        cash = self.initial_cash
        holdings = {asset: 0.0 for asset in ASSETS}
        
        trajectory.portfolio_values.append(portfolio_value)
        
        # Run episode
        for date in episode_dates:
            # Get observation from cache
            obs = self._get_observation(date)
            trajectory.states.append(obs.copy())
            
            # Get action from policy (stochastic for exploration)
            action, _ = self.policy.predict(obs, deterministic=False)
            trajectory.actions.append(action.copy())
            
            # Normalize weights
            weights = np.clip(action, 0, 1)
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(len(ASSETS)) / len(ASSETS)
            
            # Get prices for this date
            try:
                prices = {asset: self.price_df.loc[date, asset] for asset in ASSETS}
            except KeyError:
                # Skip if prices not available
                trajectory.rewards.append(0.0)
                continue
            
            # Calculate current portfolio value
            holdings_value = sum(holdings.get(a, 0) * prices.get(a, 0) for a in ASSETS)
            prev_value = cash + holdings_value
            
            # Rebalance to target weights
            target_values = {a: weights[j] * prev_value for j, a in enumerate(ASSETS)}
            
            # Execute trades
            for j, asset in enumerate(ASSETS):
                current_value = holdings.get(asset, 0) * prices.get(asset, 0)
                target_value = target_values[asset]
                diff = target_value - current_value
                
                if abs(diff) > 1 and prices.get(asset, 0) > 0:
                    shares_to_trade = diff / prices[asset]
                    holdings[asset] = holdings.get(asset, 0) + shares_to_trade
                    cash -= diff
            
            # Calculate new portfolio value after trades
            holdings_value = sum(holdings.get(a, 0) * prices.get(a, 0) for a in ASSETS)
            portfolio_value = cash + holdings_value
            
            # Compute reward
            reward = self._compute_reward(prev_value, portfolio_value)
            trajectory.rewards.append(reward)
            trajectory.portfolio_values.append(portfolio_value)
        
        trajectory.episode_length = len(trajectory.states)
        return trajectory
    
    def collect(
        self,
        num_rollouts: int,
        lambda_mdd: float = 5.0,
        risk_free_rate: float = 0.04,
        verbose: bool = True,
    ) -> List[Trajectory]:
        """
        Collect multiple trajectories.
        
        Args:
            num_rollouts: Number of trajectories to collect (M)
            lambda_mdd: Drawdown penalty for scoring
            risk_free_rate: Risk-free rate for Sharpe
            verbose: Print progress
            
        Returns:
            List of completed Trajectory objects with computed scores
        """
        trajectories = []
        
        for i in range(num_rollouts):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Collecting trajectory {i+1}/{num_rollouts}...")
            
            # Collect trajectory
            traj = self.collect_single()
            
            # Compute score
            traj.compute_score(lambda_mdd, risk_free_rate)
            
            trajectories.append(traj)
        
        return trajectories


# =======================
# ELITE FILTER
# =======================

class EliteFilter:
    """
    Filters trajectories to select the elite set.
    
    Progressive percentile:
    - Iterations 1-2: top 25%
    - Iterations 3+: top 10%
    
    Enforces minimum elite count of 30.
    """
    
    def __init__(
        self,
        percentile: float = 0.10,
        min_elites: int = 30,
    ):
        """
        Initialize the filter.
        
        Args:
            percentile: Fraction to keep (e.g., 0.10 = top 10%)
            min_elites: Minimum number of elites to keep
        """
        self.percentile = percentile
        self.min_elites = min_elites
    
    def get_progressive_percentile(self, iteration: int) -> float:
        """
        Get percentile based on iteration (progressive filtering).
        
        Args:
            iteration: Current ReST iteration (1-indexed)
            
        Returns:
            Percentile to use for this iteration
        """
        if iteration <= 2:
            return 0.25  # Top 25% for iterations 1-2
        else:
            return 0.10  # Top 10% for iterations 3+
    
    def filter(
        self,
        trajectories: List[Trajectory],
        iteration: int = 3,  # Default to stricter filtering
    ) -> Tuple[List[Trajectory], float]:
        """
        Filter trajectories to elite set.
        
        Args:
            trajectories: List of trajectories with computed scores
            iteration: Current iteration (for progressive percentile)
            
        Returns:
            Tuple of (elite_trajectories, percentile_used)
        """
        if not trajectories:
            return [], 0.0
        
        # Get percentile for this iteration
        percentile = self.get_progressive_percentile(iteration)
        
        # Sort by score (descending)
        sorted_trajs = sorted(trajectories, key=lambda t: t.score, reverse=True)
        
        # Calculate number to keep
        num_to_keep = max(
            self.min_elites,
            int(len(trajectories) * percentile)
        )
        num_to_keep = min(num_to_keep, len(trajectories))
        
        # Select elites
        elites = sorted_trajs[:num_to_keep]
        
        return elites, percentile
    
    def get_filter_stats(
        self,
        all_trajectories: List[Trajectory],
        elite_trajectories: List[Trajectory],
    ) -> Dict[str, Any]:
        """
        Get statistics about the filtering.
        
        Args:
            all_trajectories: All trajectories before filtering
            elite_trajectories: Elite trajectories after filtering
            
        Returns:
            Dictionary of filter statistics
        """
        all_scores = [t.score for t in all_trajectories]
        elite_scores = [t.score for t in elite_trajectories]
        
        return {
            "num_total": len(all_trajectories),
            "num_elite": len(elite_trajectories),
            "score_threshold": min(elite_scores) if elite_scores else 0.0,
            "mean_score_all": np.mean(all_scores) if all_scores else 0.0,
            "mean_score_elite": np.mean(elite_scores) if elite_scores else 0.0,
            "score_gap": (np.mean(elite_scores) - np.mean(all_scores)) if all_scores else 0.0,
        }


# =======================
# HELPER FUNCTIONS
# =======================

def extract_elite_data(
    elite_trajectories: List[Trajectory],
) -> Dict[str, List]:
    """
    Extract all data from elite trajectories for PPO training.
    
    Args:
        elite_trajectories: List of elite Trajectory objects
        
    Returns:
        Dictionary with:
        - 'states': All states from elite trajectories
        - 'actions': All actions from elite trajectories  
        - 'rewards': All rewards from elite trajectories
        - 'trajectory_indices': Which trajectory each tuple came from
    """
    all_states = []
    all_actions = []
    all_rewards = []
    traj_indices = []
    
    for traj_idx, traj in enumerate(elite_trajectories):
        for s, a, r in traj.get_training_tuples():
            all_states.append(s)
            all_actions.append(a)
            all_rewards.append(r)
            traj_indices.append(traj_idx)
    
    return {
        'states': all_states,
        'actions': all_actions,
        'rewards': all_rewards,
        'trajectory_indices': traj_indices,
    }


def get_elite_statistics(elite_trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Get summary statistics from elite trajectories.
    
    Args:
        elite_trajectories: List of elite Trajectory objects
        
    Returns:
        Dictionary of statistics for logging
    """
    if not elite_trajectories:
        return {}
    
    scores = [t.score for t in elite_trajectories]
    sharpes = [t.sharpe for t in elite_trajectories]
    mdds = [t.max_drawdown for t in elite_trajectories]
    start_indices = [t.start_idx for t in elite_trajectories]
    
    # Collect all actions for entropy/variance computation
    all_actions = []
    for traj in elite_trajectories:
        all_actions.extend(traj.actions)
    
    return {
        'scores': scores,
        'sharpes': sharpes,
        'mdds': mdds,
        'start_indices': start_indices,
        'all_actions': all_actions,
        'mean_score': np.mean(scores),
        'mean_sharpe': np.mean(sharpes),
        'mean_mdd': np.mean(mdds),
        'mean_episode_length': np.mean([len(t) for t in elite_trajectories]),
    }
