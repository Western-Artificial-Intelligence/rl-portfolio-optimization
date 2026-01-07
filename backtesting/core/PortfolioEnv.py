import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .data_handler import DataHandler
from .execution import ExecutionSimulator


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        assets,
        initial_cash=1_000_000,
        window=1,
        reward_mode="log_return",
        reward_window=30,
        risk_free_rate=0.0,
        drawdown_penalty=0.0,
    ):
        super().__init__()

        valid_reward_modes = {"log_return", "simple_return", "risk_adjusted"}
        if reward_mode not in valid_reward_modes:
            raise ValueError(
                f"Unknown reward mode '{reward_mode}'. Expected one of {valid_reward_modes}"
            )

        self.df = df
        self.assets = assets
        self.n_assets = len(assets)
        self.initial_cash = initial_cash
        self.window = window
        self.reward_mode = reward_mode
        self.reward_window = reward_window
        self.risk_free_rate = risk_free_rate
        self.drawdown_penalty = drawdown_penalty

        self.data_handler = DataHandler(self.df)
        self.simulator = ExecutionSimulator(self.initial_cash)

        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        obs_dim = self.n_assets * 2  # Price and sentiment for each asset
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.done = False
        self.prev_value = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.portfolio_history = []
        self.returns_history = []
        self.high_watermark = self.initial_cash

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data_handler.reset()
        self.simulator.reset()
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_value = self.initial_cash
        self.portfolio_history = []
        self.returns_history = []
        self.high_watermark = self.initial_cash

        _, row = self.data_handler.next()
        obs = self._build_observation(row)
        return obs, {}

    def step(self, action):
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights) if np.sum(weights) > 0 else 1

        ts, row = self.data_handler.next()
        if row is None:
            self.done = True
            return (
                np.zeros(self.observation_space.shape),
                0.0,
                True,
                False,
                {"portfolio_value": self.portfolio_value},
            )

        prices = {a: row[a] for a in self.assets}
        sentiments = {a: row.get(f"sentiment_{a}", 0) for a in self.assets}

        weights_dict = {asset: weight for asset, weight in zip(self.assets, weights)}
        self.simulator.execute(weights_dict, prices, ts)

        self.portfolio_value = self.simulator.portfolio_value
        self.portfolio_history.append(self.portfolio_value)
        self.high_watermark = max(self.high_watermark, self.portfolio_value)

        reward = self._compute_reward()

        obs = self._build_observation(row)
        self.current_step += 1

        truncated = False

        return (
            obs,
            reward,
            self.done,
            truncated,
            {"portfolio_value": self.portfolio_value},
        )

    def _compute_reward(self):
        if self.prev_value <= 0:
            simple_return = 0.0
        else:
            simple_return = (self.portfolio_value - self.prev_value) / self.prev_value

        self.returns_history.append(simple_return)
        if len(self.returns_history) > self.reward_window:
            self.returns_history.pop(0)

        if self.reward_mode == "log_return":
            reward = np.log(self.portfolio_value / self.prev_value)
        elif self.reward_mode == "simple_return":
            reward = simple_return
        else:  # risk_adjusted
            window_returns = np.array(self.returns_history[-self.reward_window :])
            volatility = np.std(window_returns) if len(window_returns) > 1 else 0.0
            per_step_rf = self.risk_free_rate / 252
            excess = simple_return - per_step_rf
            reward = excess / (volatility + 1e-8)

        if self.drawdown_penalty > 0 and self.high_watermark > 0:
            drawdown = (self.high_watermark - self.portfolio_value) / self.high_watermark
            reward -= self.drawdown_penalty * max(drawdown, 0)

        self.prev_value = self.portfolio_value
        return reward

    def _build_observation(self, row):
        obs = []
        for a in self.assets:
            price = row[a]
            sentiment = row.get(f"sentiment_{a}", 0)
            obs.extend([price, sentiment])

        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        print(
            f"Step {self.current_step}: Portfolio Value = {self.portfolio_value:,.2f}"
        )

