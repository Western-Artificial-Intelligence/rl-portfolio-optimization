# =======================
# File: build_deepar_dataset.py
# =======================
# Combines Bloomberg and NASDAQ-100 price data and engineers features for DeepAR
# Run this before training: python deepAR/bloomberg-data-extraction/build_deepar_dataset.py
# =======================

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
INPUT_FILES = [
    "data/bloomberg_prices.csv",
    "data/nasdaq100_prices.csv",
]
OUTPUT_PATH = "data/deepar_dataset.csv"

# Column mapping for different data sources
# Bloomberg uses: PX_LAST, PX_VOLUME, PX_OPEN, PX_HIGH, PX_LOW
# If your data has different column names, add mappings here
COLUMN_MAPPINGS = {
    # Standard Bloomberg columns (no change needed)
    "PX_LAST": "close",
    "PX_VOLUME": "volume",
    "PX_OPEN": "open",
    "PX_HIGH": "high",
    "PX_LOW": "low",
    # Alternative column names (if present)
    "Close": "close",
    "Volume": "volume",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Adj Close": "adj_close",
}


def load_and_combine_datasets(file_paths: list) -> pd.DataFrame:
    """
    Load multiple CSV files and combine them into a single DataFrame.
    
    Args:
        file_paths: List of paths to price CSV files
        
    Returns:
        Combined DataFrame with all price data
    """
    print("\n" + "=" * 60)
    print("📥 STEP 1: Loading Datasets")
    print("=" * 60)
    
    all_dfs = []
    
    for file_path in tqdm(file_paths, desc="Loading files", unit="file"):
        if not os.path.exists(file_path):
            print(f"  ⚠️  Skipping (not found): {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  ✓ Loaded: {file_path}")
            print(f"    → {len(df):,} rows, {len(df.columns)} columns, {file_size:.2f} MB")
            
            # Add source column for tracking
            df["source"] = os.path.basename(file_path).replace(".csv", "")
            all_dfs.append(df)
            
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {e}")
    
    if not all_dfs:
        raise ValueError("No data files could be loaded!")
    
    # Combine all dataframes
    print(f"\n  Combining {len(all_dfs)} datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"  ✓ Combined dataset: {len(combined):,} total rows")
    
    return combined


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across different data sources.
    
    Args:
        df: Raw DataFrame with possibly inconsistent column names
        
    Returns:
        DataFrame with standardized column names
    """
    print("\n" + "=" * 60)
    print("🔧 STEP 2: Standardizing Columns")
    print("=" * 60)
    
    # Lowercase all column names first
    df.columns = df.columns.str.strip()
    
    # Apply column mappings
    rename_map = {}
    for old_name, new_name in COLUMN_MAPPINGS.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  ✓ Renamed columns: {list(rename_map.keys())}")
    
    # Ensure required columns exist
    required_cols = ["date", "security"]
    
    # Try to find date column
    date_candidates = ["date", "Date", "DATE", "timestamp", "Timestamp"]
    for col in date_candidates:
        if col in df.columns and "date" not in df.columns:
            df = df.rename(columns={col: "date"})
            break
    
    # Try to find security/ticker column
    security_candidates = ["security", "Security", "ticker", "Ticker", "symbol", "Symbol"]
    for col in security_candidates:
        if col in df.columns and "security" not in df.columns:
            df = df.rename(columns={col: "security"})
            break
    
    # Check for price column (prefer PX_LAST or close)
    price_col = None
    for col in ["close", "PX_LAST", "Close", "price", "Price"]:
        if col in df.columns:
            price_col = col
            break
    
    if price_col and price_col != "close":
        df = df.rename(columns={price_col: "close"})
    
    # Check for volume column
    volume_col = None
    for col in ["volume", "PX_VOLUME", "Volume"]:
        if col in df.columns:
            volume_col = col
            break
    
    if volume_col and volume_col != "volume":
        df = df.rename(columns={volume_col: "volume"})
    
    print(f"  ✓ Final columns: {list(df.columns)}")
    
    # Validate required columns
    for col in ["date", "security", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for DeepAR training with progress indicators.
    
    Features created:
    - log_return: Log returns (stationarity)
    - roll_vol_10: 10-day rolling volatility
    - ret_5d: 5-day momentum
    - ret_20d: 20-day momentum
    - zvol: Volume z-score (anomaly detection)
    
    Args:
        df: DataFrame with standardized columns
        
    Returns:
        DataFrame with engineered features
    """
    print("\n" + "=" * 60)
    print("⚙️  STEP 3: Feature Engineering")
    print("=" * 60)
    
    # Parse dates and sort
    print("  Parsing dates...")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["security", "date"]).reset_index(drop=True)
    
    # Get unique securities for progress tracking
    securities = df["security"].unique()
    n_securities = len(securities)
    print(f"  Processing {n_securities} unique securities...\n")
    
    # Initialize feature columns
    df["log_return"] = np.nan
    df["roll_vol_10"] = np.nan
    df["ret_5d"] = np.nan
    df["ret_20d"] = np.nan
    
    if "volume" in df.columns:
        df["zvol"] = np.nan
    
    # Process each security with progress bar
    for security in tqdm(securities, desc="Engineering features", unit="stock", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]'):
        mask = df["security"] == security
        idx = df.index[mask]
        
        prices = df.loc[mask, "close"].values
        
        # 1. Log returns
        log_returns = np.log(prices[1:] / prices[:-1])
        df.loc[idx[1:], "log_return"] = log_returns
        
        # 2. Rolling volatility (10-day)
        for i in range(10, len(idx)):
            window = df.loc[idx[i-9:i+1], "log_return"].values
            df.loc[idx[i], "roll_vol_10"] = np.nanstd(window)
        
        # 3. Momentum returns
        for i in range(5, len(idx)):
            df.loc[idx[i], "ret_5d"] = (prices[i] - prices[i-5]) / prices[i-5]
        
        for i in range(20, len(idx)):
            df.loc[idx[i], "ret_20d"] = (prices[i] - prices[i-20]) / prices[i-20]
        
        # 4. Volume z-score (if volume available)
        if "volume" in df.columns:
            vol = df.loc[mask, "volume"].values
            if len(vol) > 0 and np.std(vol) > 0:
                df.loc[mask, "zvol"] = (vol - np.mean(vol)) / np.std(vol)
    
    print("\n  ✓ Features engineered successfully!")
    
    return df


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing NaN rows and validating data quality.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Cleaned DataFrame ready for training
    """
    print("\n" + "=" * 60)
    print("🧹 STEP 4: Cleaning & Validation")
    print("=" * 60)
    
    initial_rows = len(df)
    
    # Drop rows with NaN in critical columns
    critical_cols = ["date", "security", "close", "log_return", "roll_vol_10"]
    df = df.dropna(subset=critical_cols)
    
    dropped_rows = initial_rows - len(df)
    print(f"  ✓ Dropped {dropped_rows:,} incomplete rows ({100*dropped_rows/initial_rows:.1f}%)")
    
    # Remove any infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Print summary statistics
    print(f"\n  📊 Final Dataset Statistics:")
    print(f"     Total rows: {len(df):,}")
    print(f"     Unique securities: {df['security'].nunique()}")
    print(f"     Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"     Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Feature statistics
    print(f"\n  📈 Feature Statistics:")
    for col in ["log_return", "roll_vol_10", "ret_5d", "ret_20d"]:
        if col in df.columns:
            print(f"     {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
    
    return df


def main():
    """Main function to build the DeepAR dataset."""
    print("\n" + "=" * 60)
    print("🚀 DeepAR Dataset Builder")
    print("=" * 60)
    print(f"   Input files: {INPUT_FILES}")
    print(f"   Output file: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Step 1: Load and combine datasets
    df = load_and_combine_datasets(INPUT_FILES)
    
    # Step 2: Standardize columns
    df = standardize_columns(df)
    
    # Step 3: Engineer features
    df = engineer_features(df)
    
    # Step 4: Clean and validate
    df = clean_and_validate(df)
    
    # Step 5: Save to CSV
    print("\n" + "=" * 60)
    print("💾 STEP 5: Saving Dataset")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else ".", exist_ok=True)
    
    # Save with progress indication
    print(f"  Saving to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"  ✓ Saved successfully! ({file_size:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"\n   Next step: Train DeepAR model with:")
    print(f"   python deepAR/train_deepar.py --data {OUTPUT_PATH}")
    print("\n" + "=" * 60 + "\n")
    
    return df


if __name__ == "__main__":
    main()