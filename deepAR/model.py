# =======================
# File: deepAR/model.py
# =======================
# DeepAR-style Probabilistic Forecasting Model
# Outputs distribution parameters (mean, variance) for uncertainty-aware RL
# =======================

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class GaussianOutput(nn.Module):
    """
    Output layer that produces Gaussian distribution parameters.
    Maps hidden state to (mu, sigma) for probabilistic forecasting.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: Hidden state tensor [batch, seq_len, hidden_size]
            
        Returns:
            mu: Mean of the predicted distribution [batch, seq_len, 1]
            sigma: Standard deviation (positive) [batch, seq_len, 1]
        """
        mu = self.mu_layer(hidden)
        # Ensure sigma is positive using softplus
        sigma = nn.functional.softplus(self.sigma_layer(hidden)) + 1e-6
        return mu, sigma


class DeepARModel(nn.Module):
    """
    DeepAR-style Probabilistic Forecasting Model.
    
    This model predicts the parameters of a probability distribution
    for future values, rather than point estimates. This allows the
    PPO agent to be uncertainty-aware.
    
    Architecture:
        Input -> Embedding -> LSTM -> Gaussian Output -> (mu, sigma)
    
    Key Features:
        - Autoregressive: Uses previous predictions as input
        - Probabilistic: Outputs distribution parameters
        - Multi-variate: Can handle multiple time series with shared parameters
        - Covariate-aware: Incorporates external features
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_static_features: int = 0,
        num_dynamic_features: int = 0,
        embedding_dim: int = 16,
        num_series: int = 100,  # Number of unique time series (stocks)
    ):
        """
        Initialize the DeepAR model.
        
        Args:
            input_size: Size of the target variable (usually 1 for univariate)
            hidden_size: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_static_features: Number of static covariates (e.g., sector)
            num_dynamic_features: Number of time-varying covariates (e.g., volume)
            embedding_dim: Dimension for series embedding
            num_series: Number of unique time series (for embedding)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_series = num_series
        self.num_dynamic_features = num_dynamic_features
        self.embedding_dim = embedding_dim

        # Series embedding (learns unique representation for each stock)
        self.series_embedding = nn.Embedding(num_series, embedding_dim)

        # Calculate total input size to LSTM
        # = target + dynamic features + series embedding + static features
        lstm_input_size = (
            input_size +
            num_dynamic_features +
            embedding_dim +
            num_static_features
        )

        # Core LSTM network
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output distribution layer
        self.output_layer = GaussianOutput(hidden_size)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        target: torch.Tensor,
        series_idx: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
        dynamic_features: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            target: Target time series [batch, seq_len, 1]
            series_idx: Index of each series for embedding [batch]
            static_features: Static covariates [batch, num_static] (optional)
            dynamic_features: Time-varying covariates [batch, seq_len, num_dynamic] (optional)
            hidden: Initial hidden state (optional)
            
        Returns:
            mu: Predicted means [batch, seq_len, 1]
            sigma: Predicted standard deviations [batch, seq_len, 1]
            hidden: Final hidden state tuple
        """
        batch_size, seq_len, _ = target.shape

        # Get series embeddings and expand to sequence length
        series_emb = self.series_embedding(series_idx)  # [batch, embedding_dim]
        series_emb = series_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, emb]

        # Build input tensor
        inputs = [target, series_emb]

        if dynamic_features is not None:
            # Handle case where dynamic features are shorter than target sequence
            # This happens during training when we only have features for context
            # but target includes prediction horizon
            dyn_seq_len = dynamic_features.shape[1]
            if dyn_seq_len < seq_len:
                # Pad dynamic features with zeros for the prediction horizon
                padding = torch.zeros(
                    batch_size, seq_len - dyn_seq_len, dynamic_features.shape[2],
                    device=dynamic_features.device, dtype=dynamic_features.dtype
                )
                dynamic_features = torch.cat([dynamic_features, padding], dim=1)
            elif dyn_seq_len > seq_len:
                # Truncate if dynamic features are longer
                dynamic_features = dynamic_features[:, :seq_len, :]
            inputs.append(dynamic_features)

        if static_features is not None:
            static_expanded = static_features.unsqueeze(1).expand(-1, seq_len, -1)
            inputs.append(static_expanded)

        # Concatenate all inputs
        lstm_input = torch.cat(inputs, dim=-1)

        # LSTM forward pass
        if hidden is None:
            lstm_out, hidden = self.lstm(lstm_input)
        else:
            lstm_out, hidden = self.lstm(lstm_input, hidden)

        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Get distribution parameters
        mu, sigma = self.output_layer(lstm_out)

        return mu, sigma, hidden

    def predict(
        self,
        context: torch.Tensor,
        series_idx: torch.Tensor,
        prediction_length: int,
        static_features: Optional[torch.Tensor] = None,
        future_dynamic_features: Optional[torch.Tensor] = None,
        num_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate probabilistic forecasts autoregressively.
        
        Args:
            context: Historical context [batch, context_len, 1]
            series_idx: Series indices [batch]
            prediction_length: Number of steps to forecast
            static_features: Static covariates [batch, num_static]
            future_dynamic_features: Future covariates [batch, pred_len, num_dynamic]
            num_samples: Number of Monte Carlo samples for uncertainty
            
        Returns:
            Dictionary containing:
                - 'mean': Point forecast (mean of samples)
                - 'std': Standard deviation of forecasts
                - 'samples': Raw Monte Carlo samples
                - 'quantiles': [0.1, 0.5, 0.9] quantiles
        """
        self.eval()
        batch_size = context.shape[0]
        device = context.device

        with torch.no_grad():
            # Create zero dynamic features if model was trained with them
            context_dynamic = None
            if self.num_dynamic_features > 0:
                context_dynamic = torch.zeros(
                    batch_size, context.shape[1], self.num_dynamic_features,
                    device=device, dtype=context.dtype
                )
            
            # Encode context to get initial hidden state
            _, _, hidden = self.forward(
                target=context,
                series_idx=series_idx,
                static_features=static_features,
                dynamic_features=context_dynamic,
            )

            # Generate samples autoregressively
            all_samples = []
            
            for _ in range(num_samples):
                sample_trajectory = []
                current_hidden = hidden
                last_value = context[:, -1:, :]  # [batch, 1, 1]

                for t in range(prediction_length):
                    # Get dynamic features for this timestep if available
                    dynamic_t = None
                    if future_dynamic_features is not None:
                        dynamic_t = future_dynamic_features[:, t:t+1, :]
                    elif self.num_dynamic_features > 0:
                        # Create zero dynamic features for this step
                        dynamic_t = torch.zeros(
                            batch_size, 1, self.num_dynamic_features,
                            device=device, dtype=context.dtype
                        )

                    # Forward pass for single step
                    mu, sigma, current_hidden = self.forward(
                        target=last_value,
                        series_idx=series_idx,
                        static_features=static_features,
                        dynamic_features=dynamic_t,
                        hidden=current_hidden,
                    )

                    # Sample from predicted distribution
                    dist = torch.distributions.Normal(mu[:, -1, :], sigma[:, -1, :])
                    sample = dist.sample()  # [batch, 1]
                    sample_trajectory.append(sample)

                    # Use sample as next input (autoregressive)
                    last_value = sample.unsqueeze(1)  # [batch, 1, 1]

                # Stack trajectory [batch, pred_len]
                trajectory = torch.cat(sample_trajectory, dim=-1)
                all_samples.append(trajectory)

            # Stack all samples [num_samples, batch, pred_len]
            samples = torch.stack(all_samples, dim=0)

            # Compute statistics
            mean = samples.mean(dim=0)  # [batch, pred_len]
            std = samples.std(dim=0)    # [batch, pred_len]
            
            # Compute quantiles
            q10 = torch.quantile(samples, 0.1, dim=0)
            q50 = torch.quantile(samples, 0.5, dim=0)
            q90 = torch.quantile(samples, 0.9, dim=0)

        return {
            "mean": mean,
            "std": std,
            "samples": samples,
            "quantiles": {
                "q10": q10,
                "q50": q50,
                "q90": q90,
            },
        }

    def loss(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for Gaussian distribution.
        
        Args:
            target: Ground truth values [batch, seq_len, 1]
            mu: Predicted means [batch, seq_len, 1]
            sigma: Predicted std devs [batch, seq_len, 1]
            
        Returns:
            Scalar loss value
        """
        dist = torch.distributions.Normal(mu, sigma)
        nll = -dist.log_prob(target)
        return nll.mean()


class DeepARForecaster:
    """
    High-level wrapper for DeepAR model with training and inference utilities.
    This class provides the interface that the PPO agent will use.
    """

    def __init__(
        self,
        model: DeepARModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the forecaster wrapper.
        
        Args:
            model: Trained DeepARModel instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str, config: Dict) -> "DeepARForecaster":
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to saved model weights
            config: Model configuration dictionary
            
        Returns:
            DeepARForecaster instance
        """
        model = DeepARModel(**config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model)

    def get_forecast_vector(
        self,
        prices: np.ndarray,
        series_idx: int,
        context_length: int = 60,
        prediction_length: int = 5,
        num_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecast vector for PPO agent's state.
        
        This is the main interface for the RL agent. It takes recent price
        history and returns the forecast distribution parameters.
        
        Args:
            prices: Recent price history [context_length]
            series_idx: Index of the stock/series
            context_length: Number of historical points to use
            prediction_length: Number of future steps to forecast
            num_samples: Monte Carlo samples for uncertainty
            
        Returns:
            Dictionary with forecast features for PPO state:
                - 'forecast_mean': Expected return over horizon
                - 'forecast_std': Uncertainty (volatility) of forecast
                - 'forecast_skew': Asymmetry of forecast distribution
                - 'confidence': Inverse of normalized uncertainty
        """
        self.model.eval()

        # Prepare inputs
        prices_tensor = torch.tensor(
            prices[-context_length:], dtype=torch.float32
        ).reshape(1, -1, 1).to(self.device)
        
        series_tensor = torch.tensor([series_idx], dtype=torch.long).to(self.device)

        # Get predictions
        predictions = self.model.predict(
            context=prices_tensor,
            series_idx=series_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

        # Extract features for PPO state
        mean = predictions["mean"].cpu().numpy().flatten()
        std = predictions["std"].cpu().numpy().flatten()
        samples = predictions["samples"].cpu().numpy()

        # Compute forecast features
        forecast_return = mean.mean()  # Average expected return
        forecast_volatility = std.mean()  # Average uncertainty
        
        # Compute skewness from samples
        from scipy import stats
        all_samples = samples.flatten()
        forecast_skew = stats.skew(all_samples) if len(all_samples) > 2 else 0.0
        
        # Confidence: inverse of normalized uncertainty
        # Higher confidence = lower uncertainty relative to mean
        confidence = 1.0 / (1.0 + forecast_volatility / (abs(forecast_return) + 1e-6))

        return {
            "forecast_mean": forecast_return,
            "forecast_std": forecast_volatility,
            "forecast_skew": forecast_skew,
            "confidence": confidence,
            "quantile_10": predictions["quantiles"]["q10"].cpu().numpy().mean(),
            "quantile_90": predictions["quantiles"]["q90"].cpu().numpy().mean(),
        }


# ============================================
# UTILITY FUNCTIONS
# ============================================

def create_deepar_model(
    num_series: int = 500,
    input_size: int = 1,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    num_static_features: int = 3,  # sector, beta, market_cap
    num_dynamic_features: int = 4,  # volume, momentum, volatility, etc.
) -> DeepARModel:
    """
    Factory function to create a DeepAR model with default parameters.
    
    Args:
        num_series: Number of unique stocks/time series
        input_size: Dimension of target variable
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        num_static_features: Number of static covariates
        num_dynamic_features: Number of dynamic covariates
        
    Returns:
        Configured DeepARModel instance
    """
    return DeepARModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_static_features=num_static_features,
        num_dynamic_features=num_dynamic_features,
        num_series=num_series,
    )


if __name__ == "__main__":
    # Quick test of model architecture
    print("Testing DeepAR Model Architecture...")
    
    model = create_deepar_model(num_series=100)
    print(f"\nModel created: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 60
    
    target = torch.randn(batch_size, seq_len, 1)
    series_idx = torch.randint(0, 100, (batch_size,))
    
    mu, sigma, hidden = model(target, series_idx)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {target.shape}")
    print(f"  Output mu shape: {mu.shape}")
    print(f"  Output sigma shape: {sigma.shape}")
    print(f"  Hidden state shapes: {hidden[0].shape}, {hidden[1].shape}")
    
    # Test prediction
    print(f"\nPrediction test:")
    predictions = model.predict(
        context=target,
        series_idx=series_idx,
        prediction_length=5,
        num_samples=50,
    )
    print(f"  Mean forecast shape: {predictions['mean'].shape}")
    print(f"  Std forecast shape: {predictions['std'].shape}")
    print(f"  Samples shape: {predictions['samples'].shape}")
    
    print("\n✓ All tests passed!")
