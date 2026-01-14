# =======================
# File: tests/test_portfolio_env.py
# =======================
# Tests for the PortfolioEnv with Super-State integration
# =======================

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPortfolioEnvLegacy:
    """Tests for legacy mode (use_super_state=False)."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        assets = ["AAPL US Equity", "MSFT US Equity"]
        
        data = []
        for date in dates:
            for asset in assets:
                data.append({
                    "security": asset,
                    "date": date.strftime("%Y-%m-%d"),
                    "close": np.random.uniform(100, 200),
                    "log_return": np.random.normal(0, 0.02),
                })
        
        return pd.DataFrame(data)
    
    def test_legacy_mode_initialization(self, sample_df):
        """Test that legacy mode initializes correctly."""
        from backtesting.core.PortfolioEnv import PortfolioEnv
        
        assets = ["AAPL US Equity", "MSFT US Equity"]
        env = PortfolioEnv(
            df=sample_df,
            assets=assets,
            use_super_state=False,
        )
        
        assert env.observation_space.shape == (len(assets) * 2,)
        assert env.action_space.shape == (len(assets),)
    
    def test_legacy_mode_reset_and_step(self, sample_df):
        """Test reset and step in legacy mode."""
        from backtesting.core.PortfolioEnv import PortfolioEnv
        
        assets = ["AAPL US Equity", "MSFT US Equity"]
        env = PortfolioEnv(
            df=sample_df,
            assets=assets,
            use_super_state=False,
        )
        
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)


class TestPortfolioEnvSuperState:
    """Tests for Super-State mode (use_super_state=True)."""
    
    @pytest.fixture
    def project_root(self):
        return PROJECT_ROOT
    
    @pytest.fixture
    def has_model(self, project_root):
        """Check if DeepAR model exists."""
        checkpoint = project_root / "checkpoints" / "deepar" / "deepar_best.pt"
        return checkpoint.exists()
    
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "checkpoints" / "deepar" / "deepar_best.pt").exists(),
        reason="DeepAR checkpoint not found"
    )
    def test_super_state_initialization(self, project_root):
        """Test that Super-State mode initializes correctly."""
        from backtesting.core.PortfolioEnv import PortfolioEnv
        
        # Load real data
        data_path = project_root / "data" / "deepar_dataset.csv"
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        env = PortfolioEnv(
            df=df,
            assets=None,  # Use default 9 assets
            use_super_state=True,
        )
        
        assert env.observation_space.shape == (64,)
        assert env.action_space.shape == (9,)  # 9 default assets
    
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "checkpoints" / "deepar" / "deepar_best.pt").exists(),
        reason="DeepAR checkpoint not found"
    )
    def test_super_state_reset(self, project_root):
        """Test reset in Super-State mode."""
        from backtesting.core.PortfolioEnv import PortfolioEnv
        
        data_path = project_root / "data" / "deepar_dataset.csv"
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        env = PortfolioEnv(
            df=df,
            use_super_state=True,
        )
        
        obs, info = env.reset()
        
        assert obs.shape == (64,)
        assert obs.min() >= -1.0
        assert obs.max() <= 1.0
    
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "checkpoints" / "deepar" / "deepar_best.pt").exists(),
        reason="DeepAR checkpoint not found"
    )
    def test_super_state_step(self, project_root):
        """Test step in Super-State mode."""
        from backtesting.core.PortfolioEnv import PortfolioEnv
        
        data_path = project_root / "data" / "deepar_dataset.csv"
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        env = PortfolioEnv(
            df=df,
            use_super_state=True,
            reward_mode="sharpe",
        )
        
        obs, info = env.reset()
        action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(action)
        
        assert obs.shape == (64,)
        assert isinstance(reward, float)
        assert "portfolio_value" in info


class TestRewardModes:
    """Tests for different reward modes."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        assets = ["AAPL US Equity"]
        
        data = []
        for date in dates:
            for asset in assets:
                data.append({
                    "security": asset,
                    "date": date.strftime("%Y-%m-%d"),
                    "close": np.random.uniform(100, 200),
                })
        
        return pd.DataFrame(data)
    
    @pytest.mark.parametrize("reward_mode", ["log_return", "simple_return", "risk_adjusted", "sharpe"])
    def test_reward_modes(self, sample_df, reward_mode):
        """Test all reward modes work."""
        from backtesting.core.PortfolioEnv import PortfolioEnv
        
        env = PortfolioEnv(
            df=sample_df,
            assets=["AAPL US Equity"],
            use_super_state=False,
            reward_mode=reward_mode,
        )
        
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        assert not np.isnan(reward)


# ============================================
# Run tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
