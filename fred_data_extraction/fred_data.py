# =======================
# File: fred_data_extraction/fred_data.py
# =======================
# FRED (Federal Reserve Economic Data) API Integration
# Fetches macro-economic indicators for regime-aware portfolio optimization
# =======================

import os
import pandas as pd
import requests
from datetime import datetime
from typing import Optional, Dict, List

# ============================================
# CONFIGURATION
# ============================================
# Set your FRED API key here or via environment variable
# Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # Leave blank, set via .env

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Key macro indicators for regime detection
FRED_SERIES = {
    "T10Y2Y": "10-Year Treasury Constant Maturity Minus 2-Year Treasury (Yield Curve)",
    "VIXCLS": "CBOE Volatility Index (VIX) - Market Fear Gauge",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "CPIAUCSL": "Consumer Price Index (Inflation)",
    "UNRATE": "Unemployment Rate",
    "GDP": "Gross Domestic Product",
    "UMCSENT": "University of Michigan Consumer Sentiment",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread (Credit Spread)",
}


class FREDDataFetcher:
    """
    Fetches and processes Federal Reserve Economic Data (FRED) for
    macro-economic regime awareness in portfolio optimization.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FRED data fetcher.
        
        Args:
            api_key: FRED API key. If None, uses environment variable or config.
        """
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            raise ValueError(
                "FRED API key is required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter. Get a free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def fetch_series(
        self,
        series_id: str,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        frequency: str = "d",  # d=daily, w=weekly, m=monthly
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series.
        
        Args:
            series_id: FRED series identifier (e.g., "T10Y2Y", "VIXCLS")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            frequency: Data frequency - 'd' (daily), 'w' (weekly), 'm' (monthly)
            
        Returns:
            DataFrame with 'date' and series value columns
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "frequency": frequency,
        }

        try:
            response = requests.get(FRED_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data:
                print(f"Warning: No observations found for {series_id}")
                return pd.DataFrame(columns=["date", series_id])

            observations = data["observations"]
            df = pd.DataFrame(observations)
            
            # Clean and format data
            df = df[["date", "value"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.rename(columns={"value": series_id})
            df = df.dropna()

            print(f"✓ Fetched {len(df)} observations for {series_id}")
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {series_id}: {e}")
            return pd.DataFrame(columns=["date", series_id])

    def fetch_macro_vector(
        self,
        series_ids: Optional[List[str]] = None,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        frequency: str = "d",
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series and merge them into a single DataFrame.
        
        Args:
            series_ids: List of FRED series IDs. Defaults to core macro indicators.
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency
            
        Returns:
            DataFrame with date index and all macro indicators as columns
        """
        if series_ids is None:
            # Default: Core indicators for regime detection
            series_ids = ["T10Y2Y", "VIXCLS", "FEDFUNDS"]

        print(f"\n{'='*50}")
        print("Fetching FRED Macro Data")
        print(f"Series: {series_ids}")
        print(f"Period: {start_date} to {end_date or 'today'}")
        print(f"{'='*50}\n")

        dfs = []
        for series_id in series_ids:
            df = self.fetch_series(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
            )
            if not df.empty:
                dfs.append(df)

        if not dfs:
            print("Warning: No data fetched from FRED")
            return pd.DataFrame()

        # Merge all series on date
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on="date", how="outer")

        result = result.sort_values("date").reset_index(drop=True)
        
        # Forward fill missing values (common for different reporting frequencies)
        result = result.ffill()
        
        print(f"\n✓ Combined macro dataset: {len(result)} observations")
        print(f"  Columns: {list(result.columns)}")
        print(f"  Date range: {result['date'].min()} to {result['date'].max()}")

        return result

    def get_regime_features(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch and engineer regime-detection features from FRED data.
        
        Returns a DataFrame with:
        - Raw macro indicators
        - Derived features (yield curve inversion flag, VIX regime, etc.)
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with regime-aware features
        """
        # Fetch core macro data
        df = self.fetch_macro_vector(
            series_ids=["T10Y2Y", "VIXCLS", "FEDFUNDS"],
            start_date=start_date,
            end_date=end_date,
            frequency="d",
        )

        if df.empty:
            return df

        # ============================================
        # FEATURE ENGINEERING: Regime Detection
        # ============================================
        
        # 1. Yield Curve Inversion (Recession Indicator)
        # T10Y2Y < 0 historically predicts recessions
        if "T10Y2Y" in df.columns:
            df["yield_curve_inverted"] = (df["T10Y2Y"] < 0).astype(int)
            df["yield_curve_slope"] = df["T10Y2Y"].rolling(window=20).mean()

        # 2. VIX Regime Classification
        # Low: < 15, Normal: 15-25, High: 25-35, Extreme: > 35
        if "VIXCLS" in df.columns:
            df["vix_regime"] = pd.cut(
                df["VIXCLS"],
                bins=[0, 15, 25, 35, float("inf")],
                labels=[0, 1, 2, 3],  # Low, Normal, High, Extreme
            ).astype(float)
            df["vix_ma20"] = df["VIXCLS"].rolling(window=20).mean()
            df["vix_zscore"] = (
                (df["VIXCLS"] - df["VIXCLS"].rolling(252).mean()) /
                df["VIXCLS"].rolling(252).std()
            )

        # 3. Fed Funds Rate Change Detection
        if "FEDFUNDS" in df.columns:
            df["fed_rate_change"] = df["FEDFUNDS"].diff()
            df["fed_hiking"] = (df["fed_rate_change"] > 0).astype(int)
            df["fed_cutting"] = (df["fed_rate_change"] < 0).astype(int)

        # Drop NaN rows from rolling calculations
        df = df.dropna().reset_index(drop=True)

        print(f"\n✓ Regime features engineered: {len(df)} observations")
        print(f"  Features: {list(df.columns)}")

        return df

    def save_to_csv(
        self,
        df: pd.DataFrame,
        output_path: str = "data/fred_macro_data.csv",
    ) -> str:
        """
        Save the fetched FRED data to CSV.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved FRED data to: {output_path}")
        return output_path


def download_fred_data(
    output_path: str = "data/fred_macro_data.csv",
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to download FRED macro data.
    
    Args:
        output_path: Where to save the CSV
        start_date: Start date for data fetch
        end_date: End date (defaults to today)
        api_key: FRED API key (optional, uses env var if not provided)
        
    Returns:
        DataFrame with macro regime features
    """
    fetcher = FREDDataFetcher(api_key=api_key)
    df = fetcher.get_regime_features(start_date=start_date, end_date=end_date)
    
    if not df.empty:
        fetcher.save_to_csv(df, output_path)
    
    return df


# ============================================
# MAIN: Run as standalone script
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("FRED Data Extraction for Portfolio Optimization")
    print("=" * 60)
    
    # Check for API key
    if not FRED_API_KEY:
        print("\n⚠️  FRED_API_KEY not set!")
        print("Please set your API key:")
        print("  1. Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  2. Set environment variable: export FRED_API_KEY='your_key_here'")
        print("  3. Or update FRED_API_KEY in this file")
        exit(1)
    
    # Download macro data aligned with price data period
    df = download_fred_data(
        output_path="data/fred_macro_data.csv",
        start_date="2018-01-01",
        end_date=None,  # Today
    )
    
    if not df.empty:
        print("\n" + "=" * 60)
        print("Sample of downloaded data:")
        print("=" * 60)
        print(df.head(10))
        print(f"\nShape: {df.shape}")
