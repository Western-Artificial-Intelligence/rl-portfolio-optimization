# =======================
# File: tests/test_super_state.py
# =======================
# Unit tests for the SuperStateBuilder class
# =======================

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSuperStateBuilderUnit:
    """Unit tests using mock data (fast, test logic)."""
    
    def test_normalize_values(self):
        """Test that normalization produces values in [-1, 1] range."""
        from ppo.super_state import SuperStateBuilder
        
        # Mock the heavy initialization
        with patch.object(SuperStateBuilder, '__init__', lambda x: None):
            builder = SuperStateBuilder()
            builder._normalize = SuperStateBuilder._normalize.__get__(builder)
        
        # Test normalization
        assert builder._normalize(0.0, 0.0, 10.0) == -1.0  # min -> -1
        assert builder._normalize(10.0, 0.0, 10.0) == 1.0  # max -> 1
        assert builder._normalize(5.0, 0.0, 10.0) == 0.0   # mid -> 0
        
        # Test clipping
        assert builder._normalize(15.0, 0.0, 10.0) == 1.0  # Above max -> 1
        assert builder._normalize(-5.0, 0.0, 10.0) == -1.0 # Below min -> -1

    def test_sentiment_features_are_zeros(self):
        """Test that sentiment features are placeholder zeros."""
        from ppo.super_state import SuperStateBuilder, SENTIMENT_FEATURES
        
        with patch.object(SuperStateBuilder, '__init__', lambda x: None):
            builder = SuperStateBuilder()
            builder._get_sentiment_features = SuperStateBuilder._get_sentiment_features.__get__(builder)
        
        sentiment = builder._get_sentiment_features()
        
        assert sentiment.shape == (SENTIMENT_FEATURES,)
        assert np.all(sentiment == 0.0)

    def test_state_feature_names_count(self):
        """Test that feature names match expected dimension."""
        from ppo.super_state import get_state_feature_names, TOTAL_STATE_DIM
        
        names = get_state_feature_names()
        assert len(names) == TOTAL_STATE_DIM
        assert len(names) == 64

    def test_get_nearest_value(self):
        """Test the forward-fill value lookup."""
        from ppo.super_state import SuperStateBuilder
        
        with patch.object(SuperStateBuilder, '__init__', lambda x: None):
            builder = SuperStateBuilder()
            builder._get_nearest_value = SuperStateBuilder._get_nearest_value.__get__(builder)
        
        # Create test DataFrame
        df = pd.DataFrame({
            "value": [10.0, 20.0, 30.0]
        }, index=pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]))
        
        # Test exact match
        result = builder._get_nearest_value(df, pd.Timestamp("2024-01-05"), "value")
        assert result == 20.0
        
        # Test forward fill (date between values)
        result = builder._get_nearest_value(df, pd.Timestamp("2024-01-07"), "value")
        assert result == 20.0  # Uses Jan 5 value
        
        # Test date before any data
        result = builder._get_nearest_value(df, pd.Timestamp("2023-12-01"), "value")
        assert result is None


class TestSuperStateBuilderIntegration:
    """Integration tests using real data (slower, test real behavior)."""
    
    @pytest.fixture
    def project_root(self):
        return PROJECT_ROOT
    
    @pytest.fixture
    def has_required_files(self, project_root):
        """Check if required files exist for integration tests."""
        required = [
            project_root / "checkpoints" / "deepar" / "deepar_best.pt",
            project_root / "checkpoints" / "deepar" / "training_summary.json",
            project_root / "checkpoints" / "deepar" / "series_to_idx.json",
            project_root / "data" / "FRED" / "VIXCLS.csv",
        ]
        return all(f.exists() for f in required)
    
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "checkpoints" / "deepar" / "deepar_best.pt").exists(),
        reason="DeepAR checkpoint not found"
    )
    def test_initialization(self, project_root):
        """Test that SuperStateBuilder initializes with real files."""
        from ppo.super_state import SuperStateBuilder
        
        builder = SuperStateBuilder(project_root=str(project_root))
        
        assert builder.forecaster is not None
        assert len(builder.series_to_idx) == 9
        assert builder.state_dim == 64

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "checkpoints" / "deepar" / "deepar_best.pt").exists(),
        reason="DeepAR checkpoint not found"
    )
    def test_build_returns_correct_shape(self, project_root):
        """Test that build() returns array of correct shape."""
        from ppo.super_state import SuperStateBuilder, TOTAL_STATE_DIM
        
        builder = SuperStateBuilder(project_root=str(project_root))
        
        # Load real market data
        data_path = project_root / "data" / "deepar_dataset.csv"
        if data_path.exists():
            market_data = pd.read_csv(data_path)
            market_data["date"] = pd.to_datetime(market_data["date"])
            
            state = builder.build(market_data, date="2024-06-15")
            
            assert isinstance(state, np.ndarray)
            assert state.shape == (TOTAL_STATE_DIM,)
            assert state.shape == (64,)

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "checkpoints" / "deepar" / "deepar_best.pt").exists(),
        reason="DeepAR checkpoint not found"
    )
    def test_build_values_normalized(self, project_root):
        """Test that all output values are in [-1, 1] range."""
        from ppo.super_state import SuperStateBuilder
        
        builder = SuperStateBuilder(project_root=str(project_root))
        
        # Load real market data
        data_path = project_root / "data" / "deepar_dataset.csv"
        if data_path.exists():
            market_data = pd.read_csv(data_path)
            market_data["date"] = pd.to_datetime(market_data["date"])
            
            state = builder.build(market_data, date="2024-06-15")
            
            assert state.min() >= -1.0, f"Min value {state.min()} is less than -1"
            assert state.max() <= 1.0, f"Max value {state.max()} is greater than 1"

    def test_fred_features_with_missing_date(self, project_root):
        """Test that FRED features handle missing dates gracefully."""
        from ppo.super_state import SuperStateBuilder, FRED_FEATURES
        
        # Only run if files exist
        if not (project_root / "checkpoints" / "deepar" / "deepar_best.pt").exists():
            pytest.skip("DeepAR checkpoint not found")
        
        builder = SuperStateBuilder(project_root=str(project_root))
        
        # Use a date that definitely doesn't exist (way in the past)
        fred_features = builder._get_fred_features("1900-01-01")
        
        # Should return defaults, not crash
        assert fred_features.shape == (FRED_FEATURES,)
        assert not np.any(np.isnan(fred_features))


# ============================================
# Run tests directly
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
