# =======================
# File: ppo/precompute_states.py
# =======================
# Pre-compute Super-States for Fast PPO Training
#
# This script generates all observations upfront and saves them
# to disk, making training 100x faster by avoiding repeated
# DeepAR inference calls.
#
# Usage:
#   python -m ppo.precompute_states
#   python -m ppo.precompute_states --output data/super_states.npz
# =======================

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo.super_state import SuperStateBuilder, TOTAL_STATE_DIM


def precompute_super_states(
    data_path: str,
    output_path: str,
    context_length: int = 60,
):
    """
    Pre-compute super-states for all dates in the dataset.
    
    This generates observations once and saves them to disk,
    enabling fast training without repeated DeepAR inference.
    
    Args:
        data_path: Path to the market data CSV
        output_path: Path to save the .npz file
        context_length: Days of history for DeepAR context
        
    Output file contains:
        - 'states': (n_dates, 64) array of super-states
        - 'dates': (n_dates,) array of date strings
        - 'metadata': dict with creation info
    """
    print("\n" + "=" * 60)
    print("Pre-computing Super-States for PPO Training")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load market data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["security", "date"]).reset_index(drop=True)
    
    # Get unique dates
    all_dates = sorted(df["date"].unique())
    
    # Skip first context_length dates (need history for DeepAR)
    valid_dates = all_dates[context_length:]
    
    print(f"  Total dates: {len(all_dates)}")
    print(f"  Valid dates (after context): {len(valid_dates)}")
    print(f"  Date range: {valid_dates[0]} to {valid_dates[-1]}")
    print()
    
    # Initialize SuperStateBuilder
    print("Initializing SuperStateBuilder...")
    builder = SuperStateBuilder(project_root=str(PROJECT_ROOT))
    print()
    
    # Pre-compute all states
    print(f"Computing {len(valid_dates)} super-states...")
    print("(This may take 10-30 minutes depending on your CPU)")
    print()
    
    states = []
    date_strs = []
    
    for date in tqdm(valid_dates, desc="Progress"):
        # Convert to string format for FRED lookup
        date_str = date.strftime("%Y-%m-%d")
        
        # Filter data up to this date (for context)
        historical_data = df[df["date"] <= date]
        
        # Build super-state
        try:
            state = builder.build(
                market_data=historical_data,
                date=date_str,
                context_length=context_length,
            )
            states.append(state)
            date_strs.append(date_str)
        except Exception as e:
            print(f"\n⚠️ Warning: Failed for {date_str}: {e}")
            # Use zeros as fallback
            states.append(np.zeros(TOTAL_STATE_DIM, dtype=np.float32))
            date_strs.append(date_str)
    
    # Convert to arrays
    states_array = np.array(states, dtype=np.float32)
    dates_array = np.array(date_strs)
    
    # Verify no NaN values
    nan_count = np.sum(~np.isfinite(states_array))
    if nan_count > 0:
        print(f"\n⚠️ Warning: Found {nan_count} NaN/Inf values, replacing with 0")
        states_array = np.nan_to_num(states_array, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to compressed numpy file
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(
        output_path,
        states=states_array,
        dates=dates_array,
        context_length=context_length,
        total_dim=TOTAL_STATE_DIM,
        created_at=datetime.now().isoformat(),
    )
    
    # Print summary
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Shape: {states_array.shape}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Value range: [{states_array.min():.3f}, {states_array.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("✓ Pre-computation complete!")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNow run training with cached states:")
    print(f"  python -m ppo.train_ppo --use-cache")
    

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute super-states for PPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "deepar_dataset.csv"),
        help="Path to market data CSV",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "super_states.npz"),
        help="Path to save pre-computed states",
    )
    
    parser.add_argument(
        "--context-length",
        type=int,
        default=60,
        help="Days of history for DeepAR context",
    )
    
    args = parser.parse_args()
    
    precompute_super_states(
        data_path=args.data,
        output_path=args.output,
        context_length=args.context_length,
    )


if __name__ == "__main__":
    main()
