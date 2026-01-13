# =======================
# Data Verification Script
# =======================
# Verifies the correctness and quality of downloaded data
# Run this script to check your FRED and other data files
# =======================

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, List

# Data directory - adjust path to look at data folder from scripts location
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_csv_basic(filepath: Path, name: str) -> Optional[pd.DataFrame]:
    """
    Basic checks for a CSV file.
    Returns the DataFrame if successful, None otherwise.
    """
    print(f"\n📁 Checking: {name}")
    print(f"   Path: {filepath}")
    
    # Check if file exists
    if not filepath.exists():
        print(f"   ❌ File not found!")
        return None
    
    # Check file size
    size_bytes = filepath.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"   📊 Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    
    if size_bytes == 0:
        print(f"   ❌ File is empty!")
        return None
    
    # Try to load the CSV
    try:
        df = pd.read_csv(filepath)
        print(f"   ✓ Successfully loaded CSV")
        print(f"   📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   📋 Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"   ❌ Failed to load CSV: {e}")
        return None

def check_date_column(df: pd.DataFrame, date_col: str = "date") -> bool:
    """Check if date column is valid."""
    if date_col not in df.columns:
        # Try common date column names
        for col in ["Date", "DATE", "date", "timestamp", "Timestamp"]:
            if col in df.columns:
                date_col = col
                break
        else:
            print(f"   ⚠️  No date column found (tried: date, Date, DATE, timestamp)")
            return False
    
    try:
        dates = pd.to_datetime(df[date_col])
        min_date = dates.min()
        max_date = dates.max()
        date_range = (max_date - min_date).days
        
        print(f"   📅 Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"   📆 Span: {date_range:,} days (~{date_range/365:.1f} years)")
        
        # Check for future dates
        today = pd.Timestamp.now()
        if max_date > today:
            print(f"   ⚠️  Warning: Contains future dates (max: {max_date})")
        
        # Check for very old dates (potentially erroneous)
        if min_date < pd.Timestamp("1990-01-01"):
            print(f"   ⚠️  Warning: Contains very old dates (min: {min_date})")
        
        return True
    except Exception as e:
        print(f"   ❌ Date parsing failed: {e}")
        return False

def check_numeric_columns(df: pd.DataFrame, exclude_cols: list = None) -> dict:
    """Check numeric columns for issues."""
    exclude_cols = exclude_cols or ["date", "Date", "DATE", "timestamp", "symbol", "ticker"]
    
    results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        if col not in numeric_cols:
            # Try to convert to numeric
            try:
                numeric_data = pd.to_numeric(df[col], errors="coerce")
                non_numeric = df[col].isna().sum() != numeric_data.isna().sum()
                if non_numeric:
                    print(f"   ⚠️  Column '{col}' has non-numeric values")
            except:
                pass
            continue
        
        values = df[col].dropna()
        if len(values) == 0:
            print(f"   ❌ Column '{col}' is all NaN!")
            results[col] = {"status": "empty"}
            continue
        
        stats = {
            "min": values.min(),
            "max": values.max(),
            "mean": values.mean(),
            "std": values.std(),
            "nulls": df[col].isna().sum(),
            "null_pct": df[col].isna().sum() / len(df) * 100,
            "zeros": (values == 0).sum(),
            "negatives": (values < 0).sum(),
        }
        results[col] = stats
    
    return results

def check_missing_data(df: pd.DataFrame) -> None:
    """Check for missing data patterns."""
    null_counts = df.isnull().sum()
    null_pct = df.isnull().sum() / len(df) * 100
    
    has_nulls = null_counts[null_counts > 0]
    
    if len(has_nulls) > 0:
        print(f"\n   Missing Data Summary:")
        for col, count in has_nulls.items():
            pct = null_pct[col]
            status = "⚠️" if pct < 10 else "❌"
            print(f"   {status} {col}: {count:,} missing ({pct:.1f}%)")
    else:
        print(f"   ✓ No missing data!")

def check_duplicates(df: pd.DataFrame, date_col: str = None) -> None:
    """Check for duplicate rows."""
    total_dupes = df.duplicated().sum()
    
    if total_dupes > 0:
        print(f"   ⚠️  Found {total_dupes:,} fully duplicate rows")
    else:
        print(f"   ✓ No duplicate rows")
    
    # Check for duplicate dates if date column exists
    if date_col and date_col in df.columns:
        date_dupes = df.duplicated(subset=[date_col]).sum()
        if date_dupes > 0:
            print(f"   ⚠️  Found {date_dupes:,} duplicate dates")

def verify_fred_data():
    """Verify all FRED data files."""
    print_section("FRED Data Verification")
    
    fred_dir = DATA_DIR / "FRED"
    
    if not fred_dir.exists():
        print("❌ FRED directory not found!")
        return
    
    # Expected FRED series
    expected_series = ["T10Y2Y", "VIXCLS", "FEDFUNDS"]
    
    for series in expected_series:
        filepath = fred_dir / f"{series}.csv"
        df = check_csv_basic(filepath, f"FRED/{series}")
        
        if df is not None:
            check_date_column(df)
            check_missing_data(df)
            check_duplicates(df)
            
            # Series-specific validation
            if series == "VIXCLS" and series in df.columns:
                vix_values = pd.to_numeric(df[series], errors="coerce").dropna()
                if len(vix_values) > 0:
                    print(f"\n   VIX Statistics:")
                    print(f"      Min: {vix_values.min():.2f} (Expected: > 0)")
                    print(f"      Max: {vix_values.max():.2f} (Typical max during crisis: 80+)")
                    print(f"      Mean: {vix_values.mean():.2f} (Historical avg: ~20)")
                    if vix_values.min() < 0:
                        print(f"      ❌ VIX should never be negative!")
                    if vix_values.max() > 100:
                        print(f"      ⚠️  VIX > 100 is unusual (check data)")
            
            elif series == "T10Y2Y" and series in df.columns:
                spread_values = pd.to_numeric(df[series], errors="coerce").dropna()
                if len(spread_values) > 0:
                    print(f"\n   Yield Curve (10Y-2Y) Statistics:")
                    print(f"      Min: {spread_values.min():.2f} (Negative = inverted)")
                    print(f"      Max: {spread_values.max():.2f}")
                    print(f"      Mean: {spread_values.mean():.2f}")
                    inverted_pct = (spread_values < 0).sum() / len(spread_values) * 100
                    print(f"      Inverted periods: {inverted_pct:.1f}% of data")
            
            elif series == "FEDFUNDS" and series in df.columns:
                rate_values = pd.to_numeric(df[series], errors="coerce").dropna()
                if len(rate_values) > 0:
                    print(f"\n   Fed Funds Rate Statistics:")
                    print(f"      Min: {rate_values.min():.2f}% (Can be near 0)")
                    print(f"      Max: {rate_values.max():.2f}%")
                    print(f"      Current (last): {rate_values.iloc[-1]:.2f}%")
                    if rate_values.min() < 0:
                        print(f"      ⚠️  Negative rates unusual for Fed Funds")
            
            # Show sample data
            print(f"\n   Sample Data (first 5 rows):")
            print(df.head().to_string(index=False))

def verify_price_data():
    """Verify price data files."""
    print_section("Price Data Verification")
    
    price_files = [
        ("nasdaq100_prices.csv", ["Open", "High", "Low", "Close", "Volume"]),
        ("sp500_prices.csv", ["Open", "High", "Low", "Close", "Volume"]),
        ("bloomberg_prices.csv", None),  # Unknown structure
    ]
    
    for filename, expected_cols in price_files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"\n⏭️  Skipping {filename} (not found)")
            continue
        
        df = check_csv_basic(filepath, filename)
        
        if df is not None:
            check_date_column(df)
            check_missing_data(df)
            check_duplicates(df)
            
            # Check for price columns
            numeric_stats = check_numeric_columns(df)
            
            # OHLC sanity checks
            for col_name in ["Close", "close", "CLOSE"]:
                if col_name in df.columns:
                    close_vals = pd.to_numeric(df[col_name], errors="coerce").dropna()
                    if len(close_vals) > 0:
                        if close_vals.min() <= 0:
                            print(f"   ⚠️  Warning: {col_name} has non-positive values!")
                        if close_vals.max() > 100000:
                            print(f"   ⚠️  Warning: {col_name} has very large values (check scale)")
                    break
            
            # Check OHLC relationship (High >= Low, etc.)
            if all(col in df.columns for col in ["High", "Low", "Open", "Close"]):
                violations = (
                    (df["High"] < df["Low"]).sum() +
                    (df["High"] < df["Open"]).sum() +
                    (df["High"] < df["Close"]).sum() +
                    (df["Low"] > df["Open"]).sum() +
                    (df["Low"] > df["Close"]).sum()
                )
                if violations > 0:
                    print(f"   ⚠️  OHLC relationship violations: {violations}")
                else:
                    print(f"   ✓ OHLC relationships are valid")

def verify_deepar_dataset():
    """Verify the DeepAR dataset."""
    print_section("DeepAR Dataset Verification")
    
    filepath = DATA_DIR / "deepar_dataset.csv"
    df = check_csv_basic(filepath, "deepar_dataset.csv")
    
    if df is not None:
        check_date_column(df)
        check_missing_data(df)
        check_duplicates(df)
        
        # Check for expected DeepAR columns
        expected = ["target", "start", "dynamic_feat", "static_feat"]
        found = [col for col in expected if any(col in c for c in df.columns)]
        print(f"\n   DeepAR structure check:")
        print(f"      Expected columns: {expected}")
        print(f"      Found related: {found if found else 'None'}")

def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("   DATA VERIFICATION REPORT")
    print("   " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    # Run all verifications
    verify_fred_data()
    verify_price_data()
    verify_deepar_dataset()
    
    # Summary
    print_section("VERIFICATION COMPLETE")
    print("Review the output above for any ❌ errors or ⚠️ warnings.")
    print("If all checks pass with ✓, your data is ready for use!")

if __name__ == "__main__":
    main()
