# =======================
# File: ppo/rest_utils.py
# =======================
# Utility functions for ReST (Reinforcement learning from Self-Training)
# 
# Contains:
#   - Trajectory scoring (Sharpe, Max Drawdown)
#   - Logging utilities for ReST diagnostics
#   - Helper functions for metrics computation
#
# Note: All computations in this file are CPU-based (numpy).
# Dependencies: numpy, json, datetime, pathlib, typing, dataclasses (all in pyproject.toml or stdlib)
#
# =======================

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


# =======================
# SCORING FUNCTIONS
# =======================

def compute_trajectory_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Compute Sharpe ratio over a full trajectory's returns.
    
    This is the trajectory-level Sharpe (not per-step sum).
    Used for elite selection in ReST.
    
    Args:
        returns: Array of daily portfolio returns [r_1, ..., r_T]
        risk_free_rate: Annual risk-free rate (default 4%)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Per-step risk-free rate (daily)
    per_step_rf = risk_free_rate / 252
    
    # Excess returns
    excess_returns = returns - per_step_rf
    
    # Compute Sharpe
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns)
    
    if std_returns < 1e-8:
        # No volatility - return scaled mean
        return float(mean_excess * np.sqrt(252))
    
    # Annualized Sharpe
    sharpe = (mean_excess / std_returns) * np.sqrt(252)
    return float(sharpe)


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Compute maximum drawdown from portfolio value path.
    
    MDD = max((peak - current) / peak)
    
    Args:
        portfolio_values: Array of portfolio values over time
        
    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.15 = 15% drawdown)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # Running maximum (peak)
    peak = np.maximum.accumulate(portfolio_values)
    
    # Drawdown at each point
    drawdown = (peak - portfolio_values) / (peak + 1e-8)
    
    # Maximum drawdown
    max_dd = np.max(drawdown)
    return float(max(0.0, max_dd))


def compute_trajectory_score(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    lambda_mdd: float = 5.0,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Compute trajectory score for elite selection.
    
    R(tau) = Sharpe(tau) - lambda * MDD(tau)
    
    Args:
        returns: Array of daily portfolio returns
        portfolio_values: Array of portfolio values
        lambda_mdd: Drawdown penalty weight (default 5.0)
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Trajectory score R(tau)
    """
    sharpe = compute_trajectory_sharpe(returns, risk_free_rate)
    mdd = compute_max_drawdown(portfolio_values)
    
    score = sharpe - lambda_mdd * mdd
    return float(score)


def compute_returns_from_values(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Compute simple returns from portfolio values.
    
    Args:
        portfolio_values: Array of portfolio values [V_0, V_1, ..., V_T]
        
    Returns:
        Array of returns [r_1, ..., r_T] where r_t = (V_t - V_{t-1}) / V_{t-1}
    """
    if len(portfolio_values) < 2:
        return np.array([])
    
    values = np.array(portfolio_values)
    returns = (values[1:] - values[:-1]) / (values[:-1] + 1e-8)
    return returns


# =======================
# ENTROPY & DIAGNOSTICS
# =======================

def compute_action_entropy(actions: np.ndarray) -> float:
    """
    Compute entropy of action distribution.
    
    Higher entropy = more exploration / diverse actions.
    Low entropy = policy is becoming deterministic.
    
    Args:
        actions: Array of actions (portfolio weights) [T, n_assets]
        
    Returns:
        Average entropy across timesteps
    """
    if len(actions) == 0:
        return 0.0
    
    actions = np.array(actions)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    
    # Ensure weights are valid probabilities (sum to 1, positive)
    actions = np.clip(actions, 1e-8, 1.0)
    actions = actions / actions.sum(axis=1, keepdims=True)
    
    # Compute entropy per timestep: -sum(p * log(p))
    log_actions = np.log(actions + 1e-8)
    entropy_per_step = -np.sum(actions * log_actions, axis=1)
    
    # Average entropy
    return float(np.mean(entropy_per_step))


def compute_weight_variance(actions: np.ndarray) -> float:
    """
    Compute variance of portfolio weights across time.
    
    Low variance = static allocation (potential failure indicator).
    
    Args:
        actions: Array of actions (portfolio weights) [T, n_assets]
        
    Returns:
        Average variance of weights across time
    """
    if len(actions) == 0:
        return 0.0
    
    actions = np.array(actions)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    
    # Variance per asset across time
    variance_per_asset = np.var(actions, axis=0)
    
    # Average variance
    return float(np.mean(variance_per_asset))


# =======================
# LOGGING
# =======================

@dataclass
class RESTIterationMetrics:
    """Metrics for a single ReST iteration."""
    iteration: int
    timestamp: str
    
    # Trajectory statistics
    num_rollouts: int
    num_elites: int
    elite_percentile: float
    
    # Scores
    mean_score_all: float
    std_score_all: float
    mean_score_elite: float
    std_score_elite: float
    score_gap: float  # Delta = E[R(tau)|elite] - E[R(tau)]
    
    # Component scores
    mean_sharpe_elite: float
    mean_mdd_elite: float
    
    # Diagnostics
    action_entropy: float
    weight_variance: float
    
    # Elite start date distribution (for regime collapse detection)
    elite_start_dates: List[int]
    
    # Evaluation metrics (held-out)
    eval_sharpe: Optional[float] = None
    eval_total_return: Optional[float] = None
    eval_max_drawdown: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RESTLogger:
    """
    Logger for ReST training diagnostics.
    
    Tracks:
    - Mean R(tau) over all rollouts vs elite
    - Action entropy / weight variance
    - Histogram of elite start dates
    - Held-out evaluation performance
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.iteration_metrics: List[RESTIterationMetrics] = []
        self.best_eval_score = float('-inf')
        
    def log_iteration(
        self,
        iteration: int,
        all_scores: List[float],
        elite_scores: List[float],
        elite_sharpes: List[float],
        elite_mdds: List[float],
        elite_actions: List[np.ndarray],
        elite_start_indices: List[int],
        elite_percentile: float,
        eval_metrics: Optional[Dict[str, float]] = None,
    ) -> RESTIterationMetrics:
        """
        Log metrics for a ReST iteration.
        
        Args:
            iteration: Current iteration number
            all_scores: R(tau) scores for all trajectories
            elite_scores: R(tau) scores for elite trajectories
            elite_sharpes: Sharpe ratios for elite trajectories
            elite_mdds: Max drawdowns for elite trajectories
            elite_actions: Actions from elite trajectories (for entropy)
            elite_start_indices: Start indices of elite trajectories
            elite_percentile: Percentile used for filtering
            eval_metrics: Optional held-out evaluation metrics
            
        Returns:
            RESTIterationMetrics object
        """
        # Compute statistics
        all_scores = np.array(all_scores)
        elite_scores = np.array(elite_scores)
        
        mean_all = float(np.mean(all_scores))
        std_all = float(np.std(all_scores))
        mean_elite = float(np.mean(elite_scores))
        std_elite = float(np.std(elite_scores))
        score_gap = mean_elite - mean_all
        
        # Compute entropy and variance from elite actions
        if elite_actions:
            all_actions = np.vstack([np.array(a) for a in elite_actions])
            entropy = compute_action_entropy(all_actions)
            weight_var = compute_weight_variance(all_actions)
        else:
            entropy = 0.0
            weight_var = 0.0
        
        # Create metrics object
        metrics = RESTIterationMetrics(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            num_rollouts=len(all_scores),
            num_elites=len(elite_scores),
            elite_percentile=elite_percentile,
            mean_score_all=mean_all,
            std_score_all=std_all,
            mean_score_elite=mean_elite,
            std_score_elite=std_elite,
            score_gap=score_gap,
            mean_sharpe_elite=float(np.mean(elite_sharpes)) if elite_sharpes else 0.0,
            mean_mdd_elite=float(np.mean(elite_mdds)) if elite_mdds else 0.0,
            action_entropy=entropy,
            weight_variance=weight_var,
            elite_start_dates=elite_start_indices,
            eval_sharpe=eval_metrics.get('sharpe_ratio') if eval_metrics else None,
            eval_total_return=eval_metrics.get('total_return') if eval_metrics else None,
            eval_max_drawdown=eval_metrics.get('max_drawdown') if eval_metrics else None,
        )
        
        self.iteration_metrics.append(metrics)
        
        # Update best eval score
        if eval_metrics and eval_metrics.get('sharpe_ratio') is not None:
            eval_sharpe = eval_metrics['sharpe_ratio']
            if eval_sharpe > self.best_eval_score:
                self.best_eval_score = eval_sharpe
        
        # Save iteration log
        self._save_iteration_log(metrics)
        
        # Print summary
        self._print_iteration_summary(metrics)
        
        return metrics
    
    def _save_iteration_log(self, metrics: RESTIterationMetrics):
        """Save iteration metrics to JSON file."""
        filepath = self.log_dir / f"rest_iter_{metrics.iteration:02d}_metrics.json"
        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def _print_iteration_summary(self, metrics: RESTIterationMetrics):
        """Print iteration summary to console."""
        print(f"\n{'='*60}")
        print(f"ReST Iteration {metrics.iteration} Summary")
        print(f"{'='*60}")
        print(f"  Rollouts: {metrics.num_rollouts} | Elites: {metrics.num_elites} ({metrics.elite_percentile*100:.0f}%)")
        print(f"  Score (all):   {metrics.mean_score_all:+.3f} +/- {metrics.std_score_all:.3f}")
        print(f"  Score (elite): {metrics.mean_score_elite:+.3f} +/- {metrics.std_score_elite:.3f}")
        print(f"  Gap (Delta):   {metrics.score_gap:+.3f}")
        print(f"  Elite Sharpe:  {metrics.mean_sharpe_elite:.3f}")
        print(f"  Elite MDD:     {metrics.mean_mdd_elite:.3f}")
        print(f"  Action Entropy: {metrics.action_entropy:.3f}")
        print(f"  Weight Variance: {metrics.weight_variance:.4f}")
        
        if metrics.eval_sharpe is not None:
            print(f"\n  Held-out Evaluation:")
            print(f"    Sharpe:      {metrics.eval_sharpe:.3f}")
            print(f"    Return:      {metrics.eval_total_return*100:.2f}%")
            print(f"    Max DD:      {metrics.eval_max_drawdown*100:.2f}%")
        
        # Failure indicators
        warnings = []
        if metrics.score_gap < 0.1:
            warnings.append("Low score gap - elites not differentiating well")
        if metrics.action_entropy < 0.5:
            warnings.append("Low entropy - policy may be collapsing")
        if metrics.weight_variance < 0.001:
            warnings.append("Low weight variance - static allocations")
        
        if warnings:
            print(f"\n  [WARNING]:")
            for w in warnings:
                print(f"    - {w}")
        
        print(f"{'='*60}\n")
    
    def save_full_log(self):
        """Save complete training log."""
        filepath = self.log_dir / "rest_training_log.json"
        log_data = {
            "total_iterations": len(self.iteration_metrics),
            "best_eval_score": self.best_eval_score,
            "iterations": [m.to_dict() for m in self.iteration_metrics],
        }
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Full training log saved to: {filepath}")
    
    def check_failure_indicators(self, metrics: RESTIterationMetrics) -> List[str]:
        """
        Check for ReST failure indicators.
        
        Returns:
            List of failure indicator messages (empty if no failures)
        """
        failures = []
        
        # Score gap too small
        if metrics.score_gap < 0.05:
            failures.append("CRITICAL: Score gap near zero - rollouts not differentiating")
        
        # Entropy collapse
        if metrics.action_entropy < 0.3:
            failures.append("CRITICAL: Entropy collapse - policy too deterministic")
        
        # Static allocations
        if metrics.weight_variance < 0.0005:
            failures.append("CRITICAL: Static allocations - agent not adapting")
        
        # Elite concentration in single period (check if >50% from same region)
        if metrics.elite_start_dates:
            dates = np.array(metrics.elite_start_dates)
            # Check if elites are clustered (std < 10% of range)
            if len(dates) > 5:
                date_std = np.std(dates)
                date_range = np.max(dates) - np.min(dates) + 1
                if date_std < date_range * 0.1:
                    failures.append("WARNING: Elites concentrated in single period")
        
        return failures
