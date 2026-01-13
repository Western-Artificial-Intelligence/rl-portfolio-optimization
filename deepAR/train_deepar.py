# =======================
# File: deepAR/train_deepar.py
# =======================
# Training script for DeepAR probabilistic forecasting model
# Trains on historical stock price data to predict future distributions
# =======================

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepAR.model import DeepARModel, create_deepar_model


# ============================================
# CONFIGURATION
# ============================================
DEFAULT_CONFIG = {
    # Model architecture
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "embedding_dim": 16,
    
    # Training
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 50,
    "patience": 10,  # Early stopping patience
    
    # Data
    "context_length": 60,       # Days of history to use
    "prediction_length": 5,     # Days to forecast
    "train_split": 0.8,
    "val_split": 0.1,
    # test_split = 1 - train - val = 0.1
    
    # Features
    "target_column": "log_return",
    "dynamic_features": ["roll_vol_10", "ret_5d", "ret_20d", "zvol"],
    "static_features": [],  # Will be populated if static data available
}


# ============================================
# DATASET
# ============================================
class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.
    Creates sliding windows of (context, target) pairs.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        series_col: str = "security",
        target_col: str = "log_return",
        dynamic_features: List[str] = None,
        context_length: int = 60,
        prediction_length: int = 5,
        series_to_idx: Dict[str, int] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with columns [date, security, target, features...]
            series_col: Column name for series identifier
            target_col: Column name for target variable
            dynamic_features: List of dynamic feature column names
            context_length: Number of historical steps
            prediction_length: Number of future steps to predict
            series_to_idx: Mapping from series name to integer index
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_col = target_col
        self.dynamic_features = dynamic_features or []
        
        # Create series index mapping
        if series_to_idx is None:
            unique_series = data[series_col].unique()
            self.series_to_idx = {s: i for i, s in enumerate(unique_series)}
        else:
            self.series_to_idx = series_to_idx
        
        self.num_series = len(self.series_to_idx)
        
        # Build sample index: (series_name, start_idx)
        self.samples = []
        self._build_samples(data, series_col)

        # Store data grouped by series for efficient access
        self.data_by_series = {}
        for series_name in self.series_to_idx.keys():
            series_data = data[data[series_col] == series_name].sort_values("date")
            self.data_by_series[series_name] = series_data.reset_index(drop=True)

    def _build_samples(self, data: pd.DataFrame, series_col: str):
        """Build list of valid (series, start_idx) sample pairs."""
        window_size = self.context_length + self.prediction_length
        
        for series_name in self.series_to_idx.keys():
            series_data = data[data[series_col] == series_name]
            num_samples = len(series_data) - window_size + 1
            
            for start_idx in range(max(0, num_samples)):
                self.samples.append((series_name, start_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        series_name, start_idx = self.samples[idx]
        series_data = self.data_by_series[series_name]
        
        end_context = start_idx + self.context_length
        end_prediction = end_context + self.prediction_length
        
        # Extract context and target
        context = series_data[self.target_col].iloc[start_idx:end_context].values
        target = series_data[self.target_col].iloc[end_context:end_prediction].values
        
        # Extract dynamic features for context window
        if self.dynamic_features:
            dynamic = series_data[self.dynamic_features].iloc[start_idx:end_context].values
        else:
            dynamic = np.zeros((self.context_length, 1))
        
        # Get series index
        series_idx = self.series_to_idx[series_name]
        
        return {
            "context": torch.tensor(context, dtype=torch.float32).unsqueeze(-1),
            "target": torch.tensor(target, dtype=torch.float32).unsqueeze(-1),
            "dynamic_features": torch.tensor(dynamic, dtype=torch.float32),
            "series_idx": torch.tensor(series_idx, dtype=torch.long),
        }


# ============================================
# DATA LOADING
# ============================================
def load_training_data(
    price_csv: str,
    static_csv: Optional[str] = None,
    config: Dict = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and preprocess training data from CSV files.
    
    Args:
        price_csv: Path to price/feature CSV (from build_deepar_dataset.py)
        static_csv: Path to static features CSV (optional)
        config: Configuration dictionary
        
    Returns:
        Tuple of (processed DataFrame, updated config)
    """
    config = config or DEFAULT_CONFIG.copy()
    
    print(f"\nLoading data from: {price_csv}")
    df = pd.read_csv(price_csv)
    
    # Parse date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    # Ensure required columns exist
    required_cols = ["date", "security", config["target_column"]]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by series and date
    df = df.sort_values(["security", "date"]).reset_index(drop=True)
    
    # Validate dynamic features
    available_dynamic = [f for f in config["dynamic_features"] if f in df.columns]
    if len(available_dynamic) < len(config["dynamic_features"]):
        missing = set(config["dynamic_features"]) - set(available_dynamic)
        print(f"Warning: Missing dynamic features: {missing}")
    config["dynamic_features"] = available_dynamic
    
    # Load static features if available
    if static_csv and os.path.exists(static_csv):
        print(f"Loading static features from: {static_csv}")
        static_df = pd.read_csv(static_csv)
        # Merge static features (would need proper column mapping)
        # For now, just note it's available
        config["static_features"] = list(static_df.columns)
    
    # Get number of unique series
    num_series = df["security"].nunique()
    config["num_series"] = num_series
    
    print(f"\n{'='*50}")
    print("Data Summary:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Unique series: {num_series}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Dynamic features: {config['dynamic_features']}")
    print(f"  Target column: {config['target_column']}")
    print(f"{'='*50}\n")
    
    return df, config


def create_dataloaders(
    df: pd.DataFrame,
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train/val/test DataLoaders with proper time-based splits.
    
    Args:
        df: Preprocessed DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, series_to_idx)
    """
    # Time-based split (no data leakage)
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    
    train_end_idx = int(n_dates * config["train_split"])
    val_end_idx = int(n_dates * (config["train_split"] + config["val_split"]))
    
    train_end_date = dates[train_end_idx]
    val_end_date = dates[val_end_idx]
    
    train_df = df[df["date"] < train_end_date]
    val_df = df[(df["date"] >= train_end_date) & (df["date"] < val_end_date)]
    test_df = df[df["date"] >= val_end_date]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_df):,} samples (< {train_end_date.date()})")
    print(f"  Val:   {len(val_df):,} samples ({train_end_date.date()} - {val_end_date.date()})")
    print(f"  Test:  {len(test_df):,} samples (>= {val_end_date.date()})")
    
    # Create series mapping from training data
    series_to_idx = {s: i for i, s in enumerate(train_df["security"].unique())}
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_df,
        target_col=config["target_column"],
        dynamic_features=config["dynamic_features"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        series_to_idx=series_to_idx,
    )
    
    val_dataset = TimeSeriesDataset(
        val_df,
        target_col=config["target_column"],
        dynamic_features=config["dynamic_features"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        series_to_idx=series_to_idx,
    )
    
    test_dataset = TimeSeriesDataset(
        test_df,
        target_col=config["target_column"],
        dynamic_features=config["dynamic_features"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        series_to_idx=series_to_idx,
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,} windows")
    print(f"  Val:   {len(val_dataset):,} windows")
    print(f"  Test:  {len(test_dataset):,} windows")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader, series_to_idx


# ============================================
# TRAINING
# ============================================
class DeepARTrainer:
    """Training loop for DeepAR model."""

    def __init__(
        self,
        model: DeepARModel,
        config: Dict,
        device: str = None,
    ):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Move to device
            context = batch["context"].to(self.device)
            target = batch["target"].to(self.device)
            series_idx = batch["series_idx"].to(self.device)
            dynamic = batch["dynamic_features"].to(self.device)
            
            # Combine context and target for teacher forcing
            full_sequence = torch.cat([context, target], dim=1)
            
            # Forward pass
            self.optimizer.zero_grad()
            mu, sigma, _ = self.model(
                target=full_sequence[:, :-1, :],  # Input: all but last
                series_idx=series_idx,
                dynamic_features=dynamic,
            )
            
            # Compute loss only on prediction horizon
            pred_start = self.config["context_length"] - 1
            mu_pred = mu[:, pred_start:, :]
            sigma_pred = sigma[:, pred_start:, :]
            target_pred = full_sequence[:, self.config["context_length"]:, :]
            
            loss = self.model.loss(target_pred, mu_pred, sigma_pred)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / max(num_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                context = batch["context"].to(self.device)
                target = batch["target"].to(self.device)
                series_idx = batch["series_idx"].to(self.device)
                dynamic = batch["dynamic_features"].to(self.device)
                
                full_sequence = torch.cat([context, target], dim=1)
                
                mu, sigma, _ = self.model(
                    target=full_sequence[:, :-1, :],
                    series_idx=series_idx,
                    dynamic_features=dynamic,
                )
                
                pred_start = self.config["context_length"] - 1
                mu_pred = mu[:, pred_start:, :]
                sigma_pred = sigma[:, pred_start:, :]
                target_pred = full_sequence[:, self.config["context_length"]:, :]
                
                loss = self.model.loss(target_pred, mu_pred, sigma_pred)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str = "checkpoints",
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config["epochs"]):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            # Print progress
            print(
                f"Epoch {epoch+1:3d}/{self.config['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(save_dir, "deepar_best.pt")
                self.save_checkpoint(checkpoint_path)
                print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config["patience"]:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        # Save final model
        final_path = os.path.join(save_dir, "deepar_final.pt")
        self.save_checkpoint(final_path)
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
        print(f"{'='*60}\n")
        
        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))


# ============================================
# EVALUATION
# ============================================
def evaluate_model(
    model: DeepARModel,
    test_loader: DataLoader,
    device: str,
    config: Dict,
) -> Dict[str, float]:
    """
    Evaluate trained model on test set.
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_targets = []
    all_means = []
    all_stds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            series_idx = batch["series_idx"].to(device)
            
            # Get predictions
            predictions = model.predict(
                context=context,
                series_idx=series_idx,
                prediction_length=config["prediction_length"],
                num_samples=100,
            )
            
            all_targets.append(target.cpu().numpy())
            all_means.append(predictions["mean"].cpu().numpy())
            all_stds.append(predictions["std"].cpu().numpy())
    
    # Concatenate all predictions
    targets = np.concatenate(all_targets, axis=0).squeeze()
    means = np.concatenate(all_means, axis=0)
    stds = np.concatenate(all_stds, axis=0)
    
    # Compute metrics
    # MAE
    mae = np.abs(targets - means).mean()
    
    # RMSE
    rmse = np.sqrt(((targets - means) ** 2).mean())
    
    # Continuous Ranked Probability Score (CRPS) - simplified
    # Measures how well the predicted distribution matches the actual value
    z_scores = np.abs(targets - means) / (stds + 1e-6)
    crps_approx = z_scores.mean()
    
    # Coverage: % of true values within prediction intervals
    lower = means - 1.96 * stds  # 95% CI
    upper = means + 1.96 * stds
    coverage_95 = ((targets >= lower) & (targets <= upper)).mean()
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "crps_approx": crps_approx,
        "coverage_95": coverage_95,
        "mean_std": stds.mean(),  # Average predicted uncertainty
    }
    
    print(f"\n{'='*50}")
    print("Test Set Evaluation Metrics:")
    print(f"{'='*50}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"{'='*50}\n")
    
    return metrics


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Train DeepAR forecasting model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/deepar_dataset.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--static",
        type=str,
        default="data/nasdaq100_static.csv",
        help="Path to static features CSV (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/deepar",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--context", type=int, default=60, help="Context length (days)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon (days)")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "hidden_size": args.hidden,
        "num_layers": args.layers,
        "context_length": args.context,
        "prediction_length": args.horizon,
    })
    
    print("\n" + "="*60)
    print("DeepAR Training Pipeline")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print("="*60 + "\n")
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        print("\nPlease run the data preparation script first:")
        print("  python deepAR/bloomberg-data-extraction/build_deepar_dataset.py")
        return
    
    # Load data
    df, config = load_training_data(args.data, args.static, config)
    
    # Create data loaders
    train_loader, val_loader, test_loader, series_to_idx = create_dataloaders(df, config)
    
    # Save series mapping
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "series_to_idx.json"), "w") as f:
        json.dump(series_to_idx, f, indent=2)
    
    # Create model
    num_dynamic_features = len(config["dynamic_features"])
    model = DeepARModel(
        input_size=1,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        num_dynamic_features=num_dynamic_features,
        num_series=len(series_to_idx),
        embedding_dim=config["embedding_dim"],
    )
    
    print(f"\nModel Architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Dynamic features: {num_dynamic_features}")
    print(f"  Unique series: {len(series_to_idx)}")
    
    # Train
    trainer = DeepARTrainer(model, config)
    history = trainer.fit(train_loader, val_loader, save_dir=args.output)
    
    # Evaluate on test set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate_model(model, test_loader, device, config)
    
    # Save final config and metrics
    final_output = {
        "config": config,
        "metrics": metrics,
        "series_mapping": series_to_idx,
    }
    with open(os.path.join(args.output, "training_summary.json"), "w") as f:
        json.dump(final_output, f, indent=2, default=str)
    
    print("\n✓ Training complete! Model saved to:", args.output)


if __name__ == "__main__":
    main()
