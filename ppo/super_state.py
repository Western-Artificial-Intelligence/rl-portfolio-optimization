# =======================
# File: ppo/super_state.py
# =======================
# Super-State Builder for PPO Agent
# Combines DeepAR forecasts, FRED macro data, and sentiment placeholders
# into a unified observation vector for the PPO agent.
# =======================

import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Import DeepAR model (using relative import from project root)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from deepAR.model import DeepARModel, DeepARForecaster


# ============================================
# CONFIGURATION
# ============================================

# Default paths (relative to project root)
DEFAULT_DEEPAR_CHECKPOINT = "checkpoints/deepar/deepar_best.pt"
DEFAULT_DEEPAR_CONFIG = "checkpoints/deepar/training_summary.json"
DEFAULT_SERIES_MAPPING = "checkpoints/deepar/series_to_idx.json"
DEFAULT_FRED_DIR = "data/FRED"

# Super-State dimensions
# - 9 stocks × 6 features each = 54 forecast features
# - 6 FRED macro features
# - 4 sentiment placeholders
NUM_STOCKS = 9
FORECAST_FEATURES_PER_STOCK = 6
FRED_FEATURES = 6
SENTIMENT_FEATURES = 4  # Placeholder for future NLP integration

TOTAL_STATE_DIM = (NUM_STOCKS * FORECAST_FEATURES_PER_STOCK) + FRED_FEATURES + SENTIMENT_FEATURES
# = 54 + 6 + 4 = 64


class SuperStateBuilder:
    """
    Builds unified observation vectors for the PPO agent.
    
    The "Super-State" combines:
    1. DeepAR forecasts: Per-stock predictions (mean, std, skew, confidence, q10, q90)
    2. FRED macro data: Economic regime indicators (VIX, yield curve, fed rate)
    3. Sentiment placeholders: Reserved for future NLP integration
    
    The PPO agent receives this as its observation at each time step.
    
    Example usage:
        builder = SuperStateBuilder()
        state = builder.build(market_data, date="2024-06-15")
        # state is a np.ndarray of shape (64,) with normalized values
    """

    def __init__(
        self,
        deepar_checkpoint: Optional[str] = None,
        deepar_config: Optional[str] = None,
        series_mapping: Optional[str] = None,
        fred_dir: Optional[str] = None,
        project_root: Optional[str] = None,
    ):
        """
        Initialize the Super-State builder.
        
        Args:
            deepar_checkpoint: Path to trained DeepAR model weights (.pt file)
            deepar_config: Path to training config JSON (for model architecture)
            series_mapping: Path to series-to-index mapping JSON
            fred_dir: Path to directory containing FRED CSV files
            project_root: Root directory of the project (auto-detected if None)
        """
        # Auto-detect project root
        if project_root is None:
            project_root = str(Path(__file__).parent.parent)
        self.project_root = Path(project_root)
        
        # Set paths
        self.deepar_checkpoint = self.project_root / (deepar_checkpoint or DEFAULT_DEEPAR_CHECKPOINT)
        self.deepar_config_path = self.project_root / (deepar_config or DEFAULT_DEEPAR_CONFIG)
        self.series_mapping_path = self.project_root / (series_mapping or DEFAULT_SERIES_MAPPING)
        self.fred_dir = self.project_root / (fred_dir or DEFAULT_FRED_DIR)
        
        # Load components
        self.forecaster = self._load_deepar_model()
        self.series_to_idx = self._load_series_mapping()
        self.fred_data = self._load_fred_data()
        
        # For normalization
        self._setup_normalization_params()
        
        print(f"✓ SuperStateBuilder initialized")
        print(f"  - DeepAR model loaded from: {self.deepar_checkpoint}")
        print(f"  - Tracking {len(self.series_to_idx)} securities")
        print(f"  - FRED data loaded: {list(self.fred_data.keys())}")
        print(f"  - Total state dimension: {TOTAL_STATE_DIM}")

    def _load_deepar_model(self) -> DeepARForecaster:
        """Load the trained DeepAR model from checkpoint."""
        # Load config
        with open(self.deepar_config_path, "r") as f:
            config_data = json.load(f)
        
        config = config_data.get("config", config_data)
        
        # Build model architecture
        model = DeepARModel(
            input_size=1,  # Univariate (log returns)
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            embedding_dim=config.get("embedding_dim", 16),
            num_series=config.get("num_series", NUM_STOCKS),
            num_dynamic_features=len(config.get("dynamic_features", [])),
        )
        
        # Load weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(self.deepar_checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        return DeepARForecaster(model=model, device=device)

    def _load_series_mapping(self) -> Dict[str, int]:
        """Load the series-to-index mapping."""
        with open(self.series_mapping_path, "r") as f:
            return json.load(f)

    def _load_fred_data(self) -> Dict[str, pd.DataFrame]:
        """Load all FRED data files into memory."""
        fred_data = {}
        
        # Load each FRED series
        for series in ["T10Y2Y", "VIXCLS", "FEDFUNDS"]:
            filepath = self.fred_dir / f"{series}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath)
                # Standardize date column name
                date_col = "Date" if "Date" in df.columns else "date"
                df["date"] = pd.to_datetime(df[date_col])
                df = df.set_index("date").sort_index()
                fred_data[series] = df
            else:
                print(f"⚠️ Warning: FRED file not found: {filepath}")
        
        return fred_data

    def _setup_normalization_params(self):
        """
        Set up normalization parameters for each feature type.
        
        These are approximate ranges based on historical data.
        The goal is to normalize all features to roughly [-1, 1] range.
        """
        # DeepAR forecast features (log returns are typically small)
        self.forecast_norm = {
            "forecast_mean": {"min_val": -0.1, "max_val": 0.1},    # Expected daily return
            "forecast_std": {"min_val": 0.0, "max_val": 0.1},      # Volatility
            "forecast_skew": {"min_val": -2.0, "max_val": 2.0},    # Skewness
            "confidence": {"min_val": 0.0, "max_val": 1.0},        # Already normalized
            "quantile_10": {"min_val": -0.15, "max_val": 0.05},    # Downside quantile
            "quantile_90": {"min_val": -0.05, "max_val": 0.15},    # Upside quantile
        }
        
        # FRED macro features
        self.fred_norm = {
            "VIXCLS": {"min_val": 10.0, "max_val": 80.0},           # VIX range
            "vix_regime": {"min_val": 0, "max_val": 3},             # 4 regimes (0-3)
            "T10Y2Y": {"min_val": -1.5, "max_val": 2.5},            # Yield spread
            "yield_inverted": {"min_val": 0, "max_val": 1},         # Binary flag
            "FEDFUNDS": {"min_val": 0.0, "max_val": 6.0},           # Fed Funds rate
            "fed_direction": {"min_val": -1, "max_val": 1},         # Hiking/cutting
        }

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to [-1, 1] range.
        
        Formula: 2 * (value - min) / (max - min) - 1
        Values outside the range are clipped.
        """
        if max_val == min_val:
            return 0.0
        normalized = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
        return float(np.clip(normalized, -1.0, 1.0))

    def _get_forecast_features(
        self,
        market_data: pd.DataFrame,
        context_length: int = 60,
    ) -> np.ndarray:
        """
        Get DeepAR forecast features for all tracked securities.
        
        Args:
            market_data: DataFrame with columns ['security', 'date', 'close'] 
                         containing at least `context_length` days of history
            context_length: Number of historical days to use for forecasting
        
        Returns:
            Flattened array of shape (NUM_STOCKS * FORECAST_FEATURES_PER_STOCK,)
        """
        all_features = []
        
        # Get ordered list of securities
        securities = sorted(self.series_to_idx.keys(), key=lambda x: self.series_to_idx[x])
        
        for security in securities:
            series_idx = self.series_to_idx[security]
            
            # Get price history for this security
            security_data = market_data[market_data["security"] == security]
            
            if len(security_data) >= context_length:
                # Use log returns if available, otherwise compute from close
                if "log_return" in security_data.columns:
                    prices = security_data["log_return"].values[-context_length:]
                else:
                    closes = security_data["close"].values[-context_length:]
                    prices = np.log(closes[1:] / closes[:-1])
                    prices = np.pad(prices, (1, 0), mode="edge")
                
                # Get forecast from DeepAR
                forecast = self.forecaster.get_forecast_vector(
                    prices=prices,
                    series_idx=series_idx,
                    context_length=context_length,
                )
                
                # Normalize each feature
                features = [
                    self._normalize(forecast["forecast_mean"], **self.forecast_norm["forecast_mean"]),
                    self._normalize(forecast["forecast_std"], **self.forecast_norm["forecast_std"]),
                    self._normalize(forecast["forecast_skew"], **self.forecast_norm["forecast_skew"]),
                    self._normalize(forecast["confidence"], **self.forecast_norm["confidence"]),
                    self._normalize(forecast["quantile_10"], **self.forecast_norm["quantile_10"]),
                    self._normalize(forecast["quantile_90"], **self.forecast_norm["quantile_90"]),
                ]
            else:
                # Not enough data - use defaults (zeros)
                features = [0.0] * FORECAST_FEATURES_PER_STOCK
            
            all_features.extend(features)
        
        return np.array(all_features, dtype=np.float32)

    def _get_fred_features(self, date: str) -> np.ndarray:
        """
        Get FRED macro features for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
        
        Returns:
            Array of shape (FRED_FEATURES,) with normalized macro indicators
        """
        target_date = pd.Timestamp(date)
        
        # Initialize with defaults
        features = {
            "VIXCLS": 20.0,          # Historical average VIX
            "vix_regime": 1,          # Normal regime
            "T10Y2Y": 0.5,            # Normal positive spread
            "yield_inverted": 0,      # Not inverted
            "FEDFUNDS": 3.0,          # Moderate rate
            "fed_direction": 0,       # Neutral
        }
        
        # Get VIX
        if "VIXCLS" in self.fred_data:
            vix_df = self.fred_data["VIXCLS"]
            vix_value = self._get_nearest_value(vix_df, target_date, "VIXCLS")
            if vix_value is not None:
                features["VIXCLS"] = vix_value
                # Compute VIX regime: 0=Low(<15), 1=Normal(15-25), 2=High(25-35), 3=Extreme(>35)
                if vix_value < 15:
                    features["vix_regime"] = 0
                elif vix_value < 25:
                    features["vix_regime"] = 1
                elif vix_value < 35:
                    features["vix_regime"] = 2
                else:
                    features["vix_regime"] = 3
        
        # Get Yield Curve
        if "T10Y2Y" in self.fred_data:
            yield_df = self.fred_data["T10Y2Y"]
            yield_value = self._get_nearest_value(yield_df, target_date, "T10Y2Y")
            if yield_value is not None:
                features["T10Y2Y"] = yield_value
                features["yield_inverted"] = 1 if yield_value < 0 else 0
        
        # Get Fed Funds Rate
        if "FEDFUNDS" in self.fred_data:
            fed_df = self.fred_data["FEDFUNDS"]
            fed_value = self._get_nearest_value(fed_df, target_date, "FEDFUNDS")
            if fed_value is not None:
                features["FEDFUNDS"] = fed_value
                # Compute direction from recent change
                prev_value = self._get_nearest_value(
                    fed_df, target_date - pd.Timedelta(days=30), "FEDFUNDS"
                )
                if prev_value is not None:
                    if fed_value > prev_value + 0.1:
                        features["fed_direction"] = 1   # Hiking
                    elif fed_value < prev_value - 0.1:
                        features["fed_direction"] = -1  # Cutting
                    else:
                        features["fed_direction"] = 0   # Stable
        
        # Normalize all features
        normalized = [
            self._normalize(features["VIXCLS"], **self.fred_norm["VIXCLS"]),
            self._normalize(features["vix_regime"], **self.fred_norm["vix_regime"]),
            self._normalize(features["T10Y2Y"], **self.fred_norm["T10Y2Y"]),
            self._normalize(features["yield_inverted"], **self.fred_norm["yield_inverted"]),
            self._normalize(features["FEDFUNDS"], **self.fred_norm["FEDFUNDS"]),
            self._normalize(features["fed_direction"], **self.fred_norm["fed_direction"]),
        ]
        
        return np.array(normalized, dtype=np.float32)

    def _get_nearest_value(
        self, 
        df: pd.DataFrame, 
        target_date: pd.Timestamp,
        column: str,
    ) -> Optional[float]:
        """
        Get the value nearest to target_date (forward-fill logic).
        
        Args:
            df: DataFrame with DatetimeIndex
            target_date: Target date to look up
            column: Column name to retrieve
        
        Returns:
            The value at or before target_date, or None if not found
        """
        if df.empty:
            return None
        
        # Filter to dates on or before target
        valid = df[df.index <= target_date]
        if valid.empty:
            return None
        
        # Get most recent value
        value = valid[column].iloc[-1]
        
        # Handle "." placeholder values from FRED
        if isinstance(value, str) and value == ".":
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _get_sentiment_features(self) -> np.ndarray:
        """
        Get sentiment features (placeholder zeros for future NLP integration).
        
        When FinBERT/NLP module is implemented, this method will be updated
        to return actual sentiment scores.
        
        Returns:
            Array of shape (SENTIMENT_FEATURES,) filled with zeros
        """
        # Placeholder zeros - to be replaced with FinBERT output later
        return np.zeros(SENTIMENT_FEATURES, dtype=np.float32)

    def build(
        self,
        market_data: pd.DataFrame,
        date: str,
        context_length: int = 60,
    ) -> np.ndarray:
        """
        Build the complete Super-State observation vector.
        
        This is the main interface that the PPO agent uses. It combines:
        1. DeepAR forecast features for all stocks
        2. FRED macro economic indicators
        3. Sentiment placeholders (zeros for now)
        
        Args:
            market_data: DataFrame with columns ['security', 'date', 'close']
                         or ['security', 'date', 'log_return']
                         Must contain at least `context_length` days of history
                         for each security.
            date: Current date in YYYY-MM-DD format for FRED lookup
            context_length: Number of historical days for DeepAR context
        
        Returns:
            np.ndarray of shape (64,) with all values normalized to [-1, 1]
            
            Structure:
            - Indices 0-53:  DeepAR forecasts (9 stocks × 6 features)
            - Indices 54-59: FRED macro indicators (6 features)
            - Indices 60-63: Sentiment placeholders (4 zeros)
        
        Example:
            >>> builder = SuperStateBuilder()
            >>> state = builder.build(market_data, date="2024-06-15")
            >>> print(state.shape)  # (64,)
            >>> print(state.min(), state.max())  # Should be in [-1, 1]
        """
        # Get all feature components
        forecast_features = self._get_forecast_features(market_data, context_length)
        fred_features = self._get_fred_features(date)
        sentiment_features = self._get_sentiment_features()
        
        # Concatenate into super-state
        super_state = np.concatenate([
            forecast_features,   # Shape: (54,)
            fred_features,       # Shape: (6,)
            sentiment_features,  # Shape: (4,)
        ])
        
        # CRITICAL: Sanitize NaN and Inf values to prevent PPO training corruption
        # Replace NaN with 0.0, Inf with clipped values
        if np.any(~np.isfinite(super_state)):
            super_state = np.nan_to_num(super_state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure values are clipped to [-1, 1] range
        super_state = np.clip(super_state, -1.0, 1.0)
        
        assert super_state.shape == (TOTAL_STATE_DIM,), \
            f"Expected shape ({TOTAL_STATE_DIM},), got {super_state.shape}"
        
        return super_state

    @property
    def state_dim(self) -> int:
        """Return the dimension of the Super-State vector."""
        return TOTAL_STATE_DIM

    @property
    def securities(self) -> List[str]:
        """Return ordered list of tracked securities."""
        return sorted(self.series_to_idx.keys(), key=lambda x: self.series_to_idx[x])


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_state_feature_names() -> List[str]:
    """
    Get human-readable names for each index in the Super-State vector.
    
    Useful for debugging and visualization.
    
    Returns:
        List of 64 feature names
    """
    names = []
    
    # Forecast features (9 stocks × 6 features)
    securities = [
        "AAPL", "AMZN", "META", "MSFT", "NDX", 
        "NVDA", "PSQ", "SPX", "TSLA"
    ]
    forecast_features = [
        "mean", "std", "skew", "confidence", "q10", "q90"
    ]
    
    for security in securities:
        for feat in forecast_features:
            names.append(f"{security}_{feat}")
    
    # FRED features
    names.extend([
        "VIX", "VIX_regime", "YieldCurve", "YieldInverted", 
        "FedFunds", "FedDirection"
    ])
    
    # Sentiment placeholders
    names.extend([
        "sentiment_1", "sentiment_2", "sentiment_3", "sentiment_4"
    ])
    
    return names


# ============================================
# MAIN: Test the SuperStateBuilder
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SuperStateBuilder")
    print("=" * 60)
    
    # Initialize builder
    builder = SuperStateBuilder()
    
    # Load sample market data
    data_path = builder.project_root / "data" / "deepar_dataset.csv"
    market_data = pd.read_csv(data_path)
    market_data["date"] = pd.to_datetime(market_data["date"])
    
    # Rename columns to match expected format
    if "close" not in market_data.columns and "PX_LAST" in market_data.columns:
        market_data["close"] = market_data["PX_LAST"]
    
    # Use a date that exists in the data
    test_date = "2024-06-15"
    
    # Build super-state
    print(f"\nBuilding super-state for date: {test_date}")
    state = builder.build(market_data, date=test_date)
    
    print(f"\n✓ Super-state built successfully!")
    print(f"  Shape: {state.shape}")
    print(f"  Range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"  Mean: {state.mean():.3f}")
    print(f"  Std: {state.std():.3f}")
    
    # Show feature breakdown
    print(f"\nFeature breakdown:")
    print(f"  Forecast features (0-53): mean={state[:54].mean():.3f}")
    print(f"  FRED features (54-59): {state[54:60]}")
    print(f"  Sentiment placeholders (60-63): {state[60:64]}")
