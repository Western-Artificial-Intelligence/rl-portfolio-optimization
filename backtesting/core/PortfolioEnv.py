# =======================
# File: backtesting/core/PortfolioEnv.py
# =======================
# Gymnasium-compliant Portfolio Trading Environment
# Updated to support Super-State observations from DeepAR + FRED
# =======================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path

from .data_handler import DataHandler
from .execution import ExecutionSimulator

# Add project root to path for ppo imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import SuperStateBuilder (lazy import to avoid circular dependencies)
SUPER_STATE_AVAILABLE = False
try:
    from ppo.super_state import SuperStateBuilder, TOTAL_STATE_DIM
    SUPER_STATE_AVAILABLE = True
except ImportError:
    TOTAL_STATE_DIM = 64  # Fallback


class PortfolioEnv(gym.Env):
    """
    Gymnasium-compliant Portfolio Trading Environment.
    
    The agent observes market state and outputs portfolio weights.
    Supports two observation modes:
    
    1. Legacy mode (use_super_state=False): 
       Simple [price, sentiment] per asset
       
    2. Super-State mode (use_super_state=True):
       64-dimensional vector with DeepAR forecasts + FRED macro data
    
    Action Space:
        Box(0, 1, shape=(n_assets,)) - Portfolio weights that sum to 1
    
    Observation Space:
        Legacy: Box(-inf, inf, shape=(n_assets * 2,))
        Super-State: Box(-1, 1, shape=(64,))
    
    Reward Modes:
        - "log_return": Log of portfolio value ratio
        - "simple_return": Percentage return
        - "risk_adjusted": Excess return / volatility
        - "sharpe": Annualized Sharpe Ratio
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Default assets matching the DeepAR model
    DEFAULT_ASSETS = [
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

    def __init__(
        self,
        df,
        assets=None,
        initial_cash=1_000_000,
        window=1,
        reward_mode="sharpe",
        reward_window=30,
        risk_free_rate=0.04,  # ~4% annual risk-free rate
        drawdown_penalty=0.0,
        use_super_state=True,
        context_length=60,
        cached_states=None,  # Pre-computed super-states from precompute_states.py
    ):
        """
        Initialize the Portfolio Environment.
        
        Args:
            df: DataFrame with columns ['security', 'date', 'close', ...]
            assets: List of asset names to trade. If None, uses DEFAULT_ASSETS.
            initial_cash: Starting portfolio value
            window: Lookback window (legacy mode only)
            reward_mode: One of 'log_return', 'simple_return', 'risk_adjusted', 'sharpe'
            reward_window: Window for computing rolling statistics
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            drawdown_penalty: Penalty coefficient for drawdowns
            use_super_state: If True, use 64-dim Super-State observations
            context_length: Days of history for DeepAR context (Super-State mode)
        """
        super().__init__()

        # Validate reward mode
        valid_reward_modes = {"log_return", "simple_return", "risk_adjusted", "sharpe"}
        if reward_mode not in valid_reward_modes:
            raise ValueError(
                f"Unknown reward mode '{reward_mode}'. Expected one of {valid_reward_modes}"
            )

        # Use default assets if not specified
        if assets is None:
            assets = self.DEFAULT_ASSETS.copy()
        
        self.df = df
        self.assets = assets
        self.n_assets = len(assets)
        self.initial_cash = initial_cash
        self.window = window
        self.reward_mode = reward_mode
        self.reward_window = reward_window
        self.risk_free_rate = risk_free_rate
        self.drawdown_penalty = drawdown_penalty
        self.use_super_state = use_super_state
        self.context_length = context_length
        
        # Cached super-states for fast training
        self.cached_states = cached_states  # Dict with 'states' and 'dates'
        self.date_to_state_idx = {}
        if cached_states is not None:
            # Build date-to-index lookup for O(1) access
            for idx, date_str in enumerate(cached_states['dates']):
                self.date_to_state_idx[date_str] = idx
            print(f"✓ Using cached super-states ({len(self.date_to_state_idx)} dates)")

        # Initialize data handler and execution simulator
        self.data_handler = DataHandler(self.df)
        self.simulator = ExecutionSimulator(self.initial_cash)

        # Initialize SuperStateBuilder ONLY if no cache and super-state enabled
        self.state_builder = None
        if self.use_super_state and cached_states is None:
            if not SUPER_STATE_AVAILABLE:
                raise ImportError(
                    "SuperStateBuilder not available. "
                    "Make sure ppo/super_state.py exists and is importable."
                )
            self.state_builder = SuperStateBuilder(project_root=str(PROJECT_ROOT))
            obs_dim = TOTAL_STATE_DIM  # 64
            obs_low, obs_high = -1.0, 1.0
        elif self.use_super_state:
            # Using cache, dimensions same as super-state
            obs_dim = TOTAL_STATE_DIM  # 64
            obs_low, obs_high = -1.0, 1.0
        else:
            obs_dim = self.n_assets * 2  # Price and sentiment for each asset
            obs_low, obs_high = -np.inf, np.inf

        # Define action and observation spaces
        # Action: portfolio weights for each asset (will be normalized to sum to 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation: either Super-State or legacy price+sentiment
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.current_date = None
        self.done = False
        self.prev_value = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.portfolio_history = []
        self.returns_history = []
        self.high_watermark = self.initial_cash

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.data_handler.reset()
        self.simulator.reset()
        self.current_step = 0
        self.done = False
        self.portfolio_value = self.initial_cash
        self.prev_value = self.initial_cash
        self.portfolio_history = []
        self.returns_history = []
        self.high_watermark = self.initial_cash

        # Skip initial rows to ensure enough context for DeepAR
        if self.use_super_state:
            for _ in range(self.context_length):
                ts, row = self.data_handler.next()
                if row is None:
                    break
                self.current_date = self._extract_date(row, ts)

        # Get first observation
        ts, row = self.data_handler.next()
        if row is not None:
            self.current_date = self._extract_date(row, ts)
        obs = self._build_observation(row)
        
        return obs, {}

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Array of portfolio weights (will be normalized)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to valid portfolio weights
        weights = np.clip(action, 0, 1)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # If all zeros, equal weight
            weights = np.ones(self.n_assets) / self.n_assets

        # Get next market data
        ts, row = self.data_handler.next()
        if row is None:
            self.done = True
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,
                False,
                {"portfolio_value": self.portfolio_value},
            )

        self.current_date = self._extract_date(row, ts)

        # Get prices for execution
        prices = self._get_prices(row)
        
        # Execute trades
        weights_dict = {asset: weight for asset, weight in zip(self.assets, weights)}
        self.simulator.execute(weights_dict, prices, ts)

        # Update portfolio tracking
        self.portfolio_value = self.simulator.portfolio_value
        self.portfolio_history.append(self.portfolio_value)
        self.high_watermark = max(self.high_watermark, self.portfolio_value)

        # Compute reward
        reward = self._compute_reward()

        # Build observation
        obs = self._build_observation(row)
        self.current_step += 1

        truncated = False

        return (
            obs,
            reward,
            self.done,
            truncated,
            {
                "portfolio_value": self.portfolio_value,
                "step": self.current_step,
                "date": self.current_date,
            },
        )

    def _extract_date(self, row, ts):
        """Extract date from row or timestamp."""
        if hasattr(row, 'get'):
            return row.get('date', str(ts))
        elif hasattr(row, 'name'):
            return str(row.name)
        return str(ts)

    def _get_prices(self, row):
        """Extract prices for each asset from the row."""
        prices = {}
        for asset in self.assets:
            # Try different column name patterns
            if hasattr(row, 'get'):
                price = row.get(asset) or row.get('close') or row.get('PX_LAST')
            else:
                price = getattr(row, asset, None) or getattr(row, 'close', None)
            
            if price is None:
                # Try to get from filtered DataFrame
                asset_data = self.df[self.df['security'] == asset]
                if not asset_data.empty and self.current_date:
                    date_data = asset_data[asset_data['date'] == self.current_date]
                    if not date_data.empty:
                        price = date_data['close'].iloc[0]
            
            prices[asset] = price if price is not None else 100.0  # Default price
        
        return prices

    def _compute_reward(self):
        """Compute reward based on selected reward mode."""
        # Calculate simple return
        if self.prev_value <= 0:
            simple_return = 0.0
        else:
            simple_return = (self.portfolio_value - self.prev_value) / self.prev_value

        # Track returns history
        self.returns_history.append(simple_return)
        if len(self.returns_history) > self.reward_window:
            self.returns_history.pop(0)

        # Compute reward based on mode
        if self.reward_mode == "log_return":
            if self.portfolio_value > 0 and self.prev_value > 0:
                reward = np.log(self.portfolio_value / self.prev_value)
            else:
                reward = 0.0
                
        elif self.reward_mode == "simple_return":
            reward = simple_return
            
        elif self.reward_mode == "risk_adjusted":
            window_returns = np.array(self.returns_history[-self.reward_window:])
            volatility = np.std(window_returns) if len(window_returns) > 1 else 1e-8
            per_step_rf = self.risk_free_rate / 252
            excess = simple_return - per_step_rf
            reward = excess / (volatility + 1e-8)
            
        elif self.reward_mode == "sharpe":
            # Annualized Sharpe Ratio
            window_returns = np.array(self.returns_history[-self.reward_window:])
            if len(window_returns) > 1:
                per_step_rf = self.risk_free_rate / 252
                excess_returns = window_returns - per_step_rf
                mean_excess = np.mean(excess_returns)
                std_returns = np.std(window_returns)
                if std_returns > 1e-8:
                    # Annualized Sharpe
                    reward = (mean_excess / std_returns) * np.sqrt(252)
                else:
                    reward = mean_excess * 252  # No volatility, just scale mean
            else:
                reward = simple_return * 252
        else:
            reward = simple_return

        # Apply drawdown penalty
        if self.drawdown_penalty > 0 and self.high_watermark > 0:
            drawdown = (self.high_watermark - self.portfolio_value) / self.high_watermark
            reward -= self.drawdown_penalty * max(drawdown, 0)

        self.prev_value = self.portfolio_value
        return float(reward)

    def _build_observation(self, row):
        """
        Build the observation vector.
        
        Priority:
        1. Use cached super-states if available (fastest)
        2. Use SuperStateBuilder if no cache (slow, computes DeepAR)
        3. Legacy mode: price + sentiment per asset
        """
        if row is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # FAST PATH: Use cached super-states
        if self.use_super_state and self.cached_states is not None:
            # Normalize date format for lookup
            date_str = str(self.current_date)[:10]  # "YYYY-MM-DD"
            
            if date_str in self.date_to_state_idx:
                idx = self.date_to_state_idx[date_str]
                return self.cached_states['states'][idx].astype(np.float32)
            else:
                # Date not in cache, return zeros
                return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # SLOW PATH: Use SuperStateBuilder (computes DeepAR inference)
        if self.use_super_state and self.state_builder is not None:
            try:
                obs = self.state_builder.build(
                    market_data=self.df,
                    date=self.current_date or "2024-01-01",
                    context_length=self.context_length,
                )
                return obs.astype(np.float32)
            except Exception as e:
                print(f"Warning: SuperStateBuilder failed: {e}")
                return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # LEGACY MODE: price + sentiment per asset
        obs = []
        for a in self.assets:
            if hasattr(row, 'get'):
                price = row.get(a, 0)
                sentiment = row.get(f"sentiment_{a}", 0)
            else:
                price = getattr(row, a, 0)
                sentiment = getattr(row, f"sentiment_{a}", 0)
            obs.extend([price, sentiment])

        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        """Render the current state."""
        print(
            f"Step {self.current_step} ({self.current_date}): "
            f"Portfolio Value = ${self.portfolio_value:,.2f}"
        )

    def get_portfolio_metrics(self):
        """Get summary metrics for the episode."""
        if len(self.portfolio_history) < 2:
            return {}
        
        returns = np.diff(self.portfolio_history) / np.array(self.portfolio_history[:-1])
        
        total_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        
        if len(returns) > 1:
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        max_drawdown = 0.0
        peak = self.initial_cash
        for value in self.portfolio_history:
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "final_value": self.portfolio_value,
            "num_steps": len(self.portfolio_history),
        }
