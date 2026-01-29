# =======================
# File: ppo/elite_replay_env.py
# =======================
# Elite Replay Environment for ReST + SB3 PPO Integration
#
# This environment streams pre-collected elite trajectories to SB3 PPO.
# It does NOT simulate markets - it just replays stored (s, a, r) tuples.
#
# Key design:
#   - PPO calls env.step() and receives stored rewards
#   - SB3 computes advantages/values "normally" on elite-only experience
#   - Episodes end exactly where stored trajectory ends
#
# Note: CPU-based environment (no GPU operations).
# Dependencies: gymnasium, numpy (all in pyproject.toml via stable-baselines3)
#
# =======================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from .rest_trajectory import Trajectory


class EliteReplayEnv(gym.Env):
    """
    Replay environment that streams elite trajectories to SB3 PPO.
    
    Behavior:
    - Does NOT simulate markets
    - Streams elite tuples: (s_t, a_t, r_t)
    - Reward returned is the same step reward used during rollout collection
    - Episodes end exactly where the stored trajectory ends
    - Actions from PPO are IGNORED - we return stored rewards regardless
    
    This lets SB3 PPO compute advantages/values "normally" on elite-only
    experience, maintaining compatibility with the existing PPO implementation.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        elite_trajectories: List[Trajectory],
        observation_dim: int = 64,
        action_dim: int = 9,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize the replay environment.
        
        Args:
            elite_trajectories: List of elite Trajectory objects
            observation_dim: Dimension of observation space (default 64)
            action_dim: Dimension of action space (default 9)
            shuffle: Whether to shuffle trajectory order between epochs
            seed: Random seed for shuffling
        """
        super().__init__()
        
        if not elite_trajectories:
            raise ValueError("elite_trajectories cannot be empty")
        
        # Store trajectories
        self.original_trajectories = elite_trajectories
        self.trajectories = list(elite_trajectories)  # Working copy
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        
        # Define spaces (same as PortfolioEnv)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(observation_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Current position in replay
        self.current_traj_idx = 0
        self.current_step = 0
        self.epoch_count = 0
        
        # Statistics
        self.total_steps_replayed = 0
        self.trajectories_completed = 0
    
    def _current_trajectory(self) -> Trajectory:
        """Get current trajectory being replayed."""
        return self.trajectories[self.current_traj_idx]
    
    def _current_state(self) -> np.ndarray:
        """Get current state from trajectory."""
        traj = self._current_trajectory()
        if self.current_step < len(traj.states):
            return np.array(traj.states[self.current_step], dtype=np.float32)
        else:
            # Fallback - shouldn't normally happen
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset to start of next trajectory.
        
        This cycles through elite trajectories round-robin style.
        Optionally shuffles order each epoch.
        
        Args:
            seed: Random seed
            options: Optional configuration
            
        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Move to next trajectory
        self.current_traj_idx += 1
        
        # Check if we've completed an epoch (gone through all trajectories)
        if self.current_traj_idx >= len(self.trajectories):
            self.current_traj_idx = 0
            self.epoch_count += 1
            
            # Shuffle trajectories for next epoch
            if self.shuffle:
                self.rng.shuffle(self.trajectories)
        
        # Reset step counter
        self.current_step = 0
        
        # Get initial observation
        obs = self._current_state()
        
        info = {
            "trajectory_idx": self.current_traj_idx,
            "trajectory_start_idx": self._current_trajectory().start_idx,
            "trajectory_length": len(self._current_trajectory()),
            "epoch": self.epoch_count,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step by returning stored trajectory data.
        
        IMPORTANT: The action is IGNORED. We return the stored reward
        from the elite trajectory regardless of what action PPO produces.
        This is intentional for the ReST approach.
        
        Args:
            action: Action from PPO (IGNORED - we use stored actions/rewards)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        traj = self._current_trajectory()
        
        # Get stored reward for current step
        if self.current_step < len(traj.rewards):
            reward = traj.rewards[self.current_step]
        else:
            reward = 0.0
        
        # Advance step
        self.current_step += 1
        self.total_steps_replayed += 1
        
        # Check if trajectory is complete
        terminated = self.current_step >= len(traj.states)
        truncated = False
        
        if terminated:
            self.trajectories_completed += 1
        
        # Get next observation (or zeros if terminated)
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._current_state()
        
        # Build info dict
        info = {
            "trajectory_idx": self.current_traj_idx,
            "step_in_trajectory": self.current_step,
            "trajectory_score": traj.score,
            "trajectory_sharpe": traj.sharpe,
            "stored_action": traj.actions[self.current_step - 1] if self.current_step > 0 and self.current_step - 1 < len(traj.actions) else None,
        }
        
        # Add portfolio value if available
        if self.current_step < len(traj.portfolio_values):
            info["portfolio_value"] = traj.portfolio_values[self.current_step]
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = "human"):
        """Render current state (for debugging)."""
        traj = self._current_trajectory()
        print(f"Replaying trajectory {self.current_traj_idx}/{len(self.trajectories)}")
        print(f"  Step: {self.current_step}/{len(traj.states)}")
        print(f"  Score: {traj.score:.3f}")
        print(f"  Epoch: {self.epoch_count}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay statistics."""
        return {
            "num_trajectories": len(self.trajectories),
            "current_trajectory_idx": self.current_traj_idx,
            "current_step": self.current_step,
            "epoch_count": self.epoch_count,
            "total_steps_replayed": self.total_steps_replayed,
            "trajectories_completed": self.trajectories_completed,
        }
    
    def get_total_timesteps(self) -> int:
        """Get total timesteps across all trajectories (for PPO.learn())."""
        return sum(len(t.states) for t in self.trajectories)
    
    @property
    def num_trajectories(self) -> int:
        """Number of elite trajectories."""
        return len(self.trajectories)


def create_elite_replay_env(
    elite_trajectories: List[Trajectory],
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> EliteReplayEnv:
    """
    Factory function to create an elite replay environment.
    
    Args:
        elite_trajectories: List of elite Trajectory objects
        shuffle: Whether to shuffle trajectory order
        seed: Random seed
        
    Returns:
        EliteReplayEnv instance
    """
    return EliteReplayEnv(
        elite_trajectories=elite_trajectories,
        shuffle=shuffle,
        seed=seed,
    )
