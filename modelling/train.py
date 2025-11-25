"""
Weather Forecasting Model Training Script

Multi-station time series forecasting using tsai's PatchTST architecture.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tsai.all import (
    Nan2Value,
    ShowGraph,
    SlidingWindowPanel,
    TSForecaster,
    TSStandardize,
    get_splits,
    mae,
    rmse,
)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    window_len: int = 60  # Input window length (timesteps)
    horizon: int = 6  # Prediction horizon (timesteps)
    valid_size: float = 0.2  # Validation set size
    random_state: int = 23  # Random seed for reproducibility
    n_epochs: int = 20  # Number of training epochs
    lr_max: float = 1e-3  # Maximum learning rate
    batch_size: int = 128  # Batch size
    arch: str = "PatchTST"  # Model architecture


@dataclass
class TrainingResult:
    """Results from model training."""

    model_path: str
    total_samples: int
    train_samples: int
    valid_samples: int
    num_stations: int
    final_train_loss: float
    final_valid_loss: float
    mae_per_variable: Dict[str, float]
    training_history: Dict[str, list]


def prepare_data(df: pd.DataFrame, features: list[str] = None) -> pd.DataFrame:
    """
    Prepare multi-station time series data for training.

    Args:
        df: Input dataframe with columns ['id', 'timestamp', ...features]
        features: List of feature column names to use

    Returns:
        Cleaned and sorted dataframe ready for windowing
    """
    if features is None:
        features = ["temperature", "wind_average", "wind_gust", "wind_bearing"]

    # Sort by station ID and timestamp
    df_sorted = df.sort_values(["id", "timestamp"]).reset_index(drop=True)

    # Handle missing values per station group
    # Forward fill then backward fill within each station
    df_sorted[features] = df_sorted.groupby("id")[features].transform(
        lambda x: x.ffill().bfill()
    )

    # Drop any remaining rows with missing values
    df_sorted = df_sorted.dropna(subset=features)

    print("Data preparation complete:")
    print(f"  - Total stations: {df_sorted['id'].nunique()}")
    print(f"  - Total rows: {len(df_sorted)}")
    print(f"  - Missing values: {df_sorted[features].isnull().sum().sum()}")

    return df_sorted


def create_windows(
    df: pd.DataFrame, window_len: int, horizon: int, features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for multi-station time series.

    Args:
        df: Prepared dataframe
        window_len: Length of input window
        horizon: Prediction horizon
        features: Feature columns

    Returns:
        Tuple of (X, y) arrays
    """
    print("\nCreating sliding windows:")
    print(f"  - Window length: {window_len} timesteps")
    print(f"  - Horizon: {horizon} timesteps")

    X, y = SlidingWindowPanel(
        window_len=window_len,
        horizon=horizon,
        unique_id_cols=["id"],
        get_x=features,
        get_y=features,
        sort_by="timestamp",
    )(df)

    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")

    # Verify no NaN values
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN values detected in training data after windowing!")

    print("  - Data quality: OK (no NaN values)")

    return X, y


def train_model(
    df: pd.DataFrame,
    config: Optional[TrainingConfig] = None,
    model_dir: str = "models",
    model_name: str = "weather_forecast",
    features: Optional[list[str]] = None,
    show_plot: bool = True,
) -> TrainingResult:
    """
    Train a multi-station weather forecasting model.

    Args:
        df: Input dataframe with columns ['id', 'timestamp'] + features
        config: Training configuration (uses defaults if None)
        model_dir: Directory to save model
        model_name: Base name for saved model file
        features: Feature columns to use (defaults to weather variables)
        show_plot: Whether to show training plot

    Returns:
        TrainingResult object with model path and performance metrics
    """
    # Use default config if not provided
    if config is None:
        config = TrainingConfig()

    # Use default features if not provided
    if features is None:
        features = ["temperature", "wind_average", "wind_gust", "wind_bearing"]

    print("=" * 60)
    print("MULTI-STATION WEATHER FORECASTING MODEL TRAINING")
    print("=" * 60)

    # Step 1: Prepare data
    print("\n[1/5] Preparing data...")
    df_clean = prepare_data(df, features)

    # Step 2: Create sliding windows
    print("\n[2/5] Creating sliding windows...")
    X, y = create_windows(df_clean, config.window_len, config.horizon, features)

    # Step 3: Create train/validation split
    print("\n[3/5] Creating train/validation split...")
    splits = get_splits(
        range(len(y)),
        valid_size=config.valid_size,
        shuffle=True,
        random_state=config.random_state,
    )
    print(f"  - Train samples: {len(splits[0])}")
    print(f"  - Valid samples: {len(splits[1])}")

    # Step 4: Build and train model
    print(f"\n[4/5] Building {config.arch} model...")

    # Create callbacks
    callbacks = []
    if show_plot:
        callbacks.append(ShowGraph())

    fcst = TSForecaster(
        X,
        y,
        splits=splits,
        path=model_dir,
        batch_tfms=[TSStandardize(), Nan2Value()],
        bs=config.batch_size,
        arch=config.arch,
        metrics=[mae, rmse],
        cbs=callbacks,
    )

    print(f"\n  Training for {config.n_epochs} epochs...")
    fcst.fit_one_cycle(n_epoch=config.n_epochs, lr_max=config.lr_max)

    # Step 5: Evaluate and save
    print("\n[5/5] Evaluating model...")

    # Get validation predictions
    preds, targets = fcst.get_X_preds(X[splits[1]], y[splits[1]])

    # Calculate MAE per variable
    mae_per_var = np.abs(preds - targets).mean(axis=(0, 1))
    mae_dict = {
        feature: float(mae_val) for feature, mae_val in zip(features, mae_per_var)
    }

    print("\n  Validation MAE per variable:")
    for feature, mae_val in mae_dict.items():
        print(f"    - {feature}: {mae_val:.4f}")

    # Get training history
    train_loss = fcst.recorder.values[-1][0]
    valid_loss = fcst.recorder.values[-1][1]

    print(f"\n  Final training loss: {train_loss:.4f}")
    print(f"  Final validation loss: {valid_loss:.4f}")

    # Save model
    model_path = Path(model_dir) / f"{model_name}.pkl"
    fcst.export(str(model_path))
    print(f"\n  Model saved to: {model_path}")

    # Extract training history
    history = {
        "train_loss": [v[0] for v in fcst.recorder.values],
        "valid_loss": [v[1] for v in fcst.recorder.values],
        "mae": [v[2] for v in fcst.recorder.values],
        "rmse": [v[3] for v in fcst.recorder.values],
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Return results
    return TrainingResult(
        model_path=str(model_path),
        total_samples=len(X),
        train_samples=len(splits[0]),
        valid_samples=len(splits[1]),
        num_stations=df_clean["id"].nunique(),
        final_train_loss=float(train_loss),
        final_valid_loss=float(valid_loss),
        mae_per_variable=mae_dict,
        training_history=history,
    )


def main():
    """Example usage."""
    import sys

    # Load data
    data_path = "data.csv"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Configure training
    config = TrainingConfig(window_len=60, horizon=6, n_epochs=20, batch_size=128)

    # Train model
    result = train_model(
        df=df,
        config=config,
        model_dir="models",
        model_name="weather_forecast_multi_station",
    )

    # Print summary
    print("\nTraining Summary:")
    print(f"  Model: {result.model_path}")
    print(f"  Stations: {result.num_stations}")
    print(
        f"  Samples: {result.total_samples} total, "
        f"{result.train_samples} train, {result.valid_samples} valid"
    )
    print(f"  Final validation loss: {result.final_valid_loss:.4f}")


if __name__ == "__main__":
    main()
