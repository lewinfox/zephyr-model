"""
TimesFM Zero-Shot Weather Forecasting

Uses Google's pretrained TimesFM 2.5 foundation model to generate weather
forecasts without any training. Each weather variable is forecast independently
(univariate) per station.

Usage:
    python timesfm_forecast.py               # runs backtest + current forecast
    python timesfm_forecast.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup so we can reuse train.py data loading ──────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train import get_training_data  # noqa: E402  (needs path setup above)

# ── defaults ──────────────────────────────────────────────────────────────────
FEATURES = ["temperature", "wind_average", "wind_gust", "wind_bearing"]
CONTEXT_LEN = 60   # 10 hours of history at 10-min intervals
HORIZON = 6        # 1 hour ahead


# ── model loading ─────────────────────────────────────────────────────────────

def load_model():
    """
    Load the TimesFM 2.5 (200M) pretrained model.

    Downloads ~800 MB of weights from HuggingFace on first run; cached
    locally afterwards.
    """
    import timesfm  # imported here so the module is usable without timesfm installed
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    return model


# ── backtesting ───────────────────────────────────────────────────────────────

def _rolling_eval_windows(
    series: np.ndarray,
    context_len: int,
    horizon: int,
    stride: int,
    test_start: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (inputs, actuals) lists for rolling-window evaluation."""
    n = len(series)
    inputs, actuals = [], []
    for i in range(test_start, n - horizon, stride):
        ctx = series[max(0, i - context_len):i]
        if len(ctx) < 2:          # need at least 2 points for TimesFM
            continue
        inputs.append(ctx.astype(float))
        actuals.append(series[i:i + horizon].astype(float))
    return inputs, actuals


def backtest(
    df: pd.DataFrame,
    model,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    stride: int | None = None,
    features: list[str] | None = None,
    test_fraction: float = 0.2,
) -> dict:
    """
    Rolling-window backtest: slide a window through the held-out test portion
    of each station's history and compare TimesFM forecasts to actuals.

    Args:
        df:            DataFrame with columns ['station_id', 'timestamp'] + features.
        model:         Loaded TimesFM model (from load_model()).
        context_len:   Past timesteps fed to the model as context.
        horizon:       Timesteps to forecast at each evaluation point.
        stride:        Gap between evaluation points (default: horizon → non-overlapping).
        features:      Variables to evaluate (default: all four weather vars).
        test_fraction: Fraction of each station's data held out for evaluation.

    Returns:
        dict with keys
            'mae'     – {variable: float}
            'rmse'    – {variable: float}
            'records' – DataFrame of every (station, variable, horizon_step) prediction
    """
    if features is None:
        features = FEATURES
    if stride is None:
        stride = horizon

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    all_records: list[dict] = []

    stations = df["station_id"].unique()
    print(f"  Backtesting {len(stations)} stations × {len(features)} variables "
          f"(test={test_fraction:.0%}, stride={stride})...")

    for station_id, station_df in df.groupby("station_id"):
        station_df = station_df.reset_index(drop=True)
        n = len(station_df)
        test_start = int(n * (1 - test_fraction))

        if test_start < context_len:
            continue  # not enough history for this station

        for var in features:
            series = station_df[var].values
            inputs, actuals = _rolling_eval_windows(
                series, context_len, horizon, stride, test_start
            )
            if not inputs:
                continue

            # Batch all windows for this (station, variable) in one call
            point_preds, _ = model.forecast(horizon=horizon, inputs=inputs)

            eval_indices = list(range(test_start, n - horizon, stride))[:len(inputs)]
            for idx, (pred, actual, eval_i) in enumerate(
                zip(point_preds, actuals, eval_indices)
            ):
                ts = station_df.iloc[eval_i]["timestamp"]
                for h in range(min(horizon, len(actual))):
                    all_records.append({
                        "station_id": station_id,
                        "variable": var,
                        "timestamp": ts,
                        "horizon_step": h + 1,
                        "predicted": float(pred[h]),
                        "actual": float(actual[h]),
                    })

    records = pd.DataFrame(all_records)
    if records.empty:
        return {"mae": {}, "rmse": {}, "records": records}

    records["error"] = records["predicted"] - records["actual"]
    records["abs_error"] = records["error"].abs()

    mae_dict, rmse_dict = {}, {}
    for var in features:
        vdf = records[(records["variable"] == var) & records["actual"].notna()]
        if len(vdf):
            mae_dict[var] = float(vdf["abs_error"].mean())
            rmse_dict[var] = float(np.sqrt((vdf["error"] ** 2).mean()))

    return {"mae": mae_dict, "rmse": rmse_dict, "records": records}


def persistence_backtest(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    stride: int | None = None,
    features: list[str] | None = None,
    test_fraction: float = 0.2,
) -> dict:
    """
    Naive persistence baseline: forecast = last observed value for every step.

    This is the simplest possible benchmark — TimesFM should beat it on all
    but the shortest horizons.
    """
    if features is None:
        features = FEATURES
    if stride is None:
        stride = horizon

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    all_records: list[dict] = []

    for station_id, station_df in df.groupby("station_id"):
        station_df = station_df.reset_index(drop=True)
        n = len(station_df)
        test_start = int(n * (1 - test_fraction))

        for var in features:
            series = station_df[var].values.astype(float)

            for i in range(test_start, n - horizon, stride):
                last_val = series[i - 1]
                actual = series[i:i + horizon]
                ts = station_df.iloc[i]["timestamp"]

                for h in range(min(horizon, len(actual))):
                    all_records.append({
                        "station_id": station_id,
                        "variable": var,
                        "timestamp": ts,
                        "horizon_step": h + 1,
                        "predicted": last_val,
                        "actual": float(actual[h]),
                    })

    records = pd.DataFrame(all_records)
    if records.empty:
        return {"mae": {}, "rmse": {}, "records": records}

    records["error"] = records["predicted"] - records["actual"]
    records["abs_error"] = records["error"].abs()

    mae_dict, rmse_dict = {}, {}
    for var in features:
        vdf = records[(records["variable"] == var) & records["actual"].notna()]
        if len(vdf):
            mae_dict[var] = float(vdf["abs_error"].mean())
            rmse_dict[var] = float(np.sqrt((vdf["error"] ** 2).mean()))

    return {"mae": mae_dict, "rmse": rmse_dict, "records": records}


# ── live forecasting ──────────────────────────────────────────────────────────

def forecast_current(
    df: pd.DataFrame,
    model,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate forecasts from the most recent context_len observations per station.

    Returns a DataFrame with columns:
        station_id, variable, horizon_step, forecast_from,
        predicted, q10, q50, q90
    """
    if features is None:
        features = FEATURES

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    rows: list[dict] = []

    for station_id, station_df in df.groupby("station_id"):
        context = station_df.tail(context_len)
        last_ts = context.iloc[-1]["timestamp"]

        for var in features:
            series = context[var].values.astype(float)
            point_pred, quantile_pred = model.forecast(horizon=horizon, inputs=[series])

            n_q = quantile_pred.shape[-1]
            for h in range(horizon):
                rows.append({
                    "station_id": station_id,
                    "variable": var,
                    "horizon_step": h + 1,
                    "forecast_from": last_ts,
                    "predicted": float(point_pred[0][h]),
                    # Use first, middle, last quantile as p10 / p50 / p90
                    "q10": float(quantile_pred[0][h][0]),
                    "q50": float(quantile_pred[0][h][n_q // 2]),
                    "q90": float(quantile_pred[0][h][-1]),
                })

    return pd.DataFrame(rows)


# ── reporting ─────────────────────────────────────────────────────────────────

def print_backtest_comparison(
    timesfm_results: dict,
    persistence_results: dict,
    features: list[str] | None = None,
) -> None:
    """Print a formatted side-by-side MAE comparison."""
    if features is None:
        features = FEATURES

    print()
    print("=" * 68)
    print("BACKTEST  –  TimesFM vs Persistence Baseline (MAE, lower is better)")
    print("=" * 68)
    print(f"{'Variable':<24} {'TimesFM':>10} {'Persistence':>13} {'Δ':>10}")
    print("-" * 68)

    for var in features:
        tfm = timesfm_results["mae"].get(var, float("nan"))
        per = persistence_results["mae"].get(var, float("nan"))
        if not (np.isnan(tfm) or np.isnan(per)) and per > 0:
            delta = (per - tfm) / per * 100
            arrow = "▲" if delta > 0 else "▼"
            print(f"  {var:<22} {tfm:>10.3f} {per:>13.3f} {arrow}{abs(delta):>8.1f}%")
        else:
            print(f"  {var:<22} {tfm:>10.3f} {per:>13.3f} {'—':>10}")

    print("=" * 68)
    print("  ▲ = TimesFM better than persistence   ▼ = persistence better")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    STATION_NAME = "Coronet Tandems"
    MAX_DISTANCE = 20

    print(f"Loading data: stations within {MAX_DISTANCE} km of '{STATION_NAME}'...")
    df = get_training_data(STATION_NAME, MAX_DISTANCE)
    print(f"  {len(df):,} rows, {df['station_id'].nunique()} stations")

    print("\nLoading TimesFM model (downloads ~800 MB on first run)...")
    model = load_model()
    print("  Model ready.\n")

    print("Running persistence baseline...")
    persistence_results = persistence_backtest(df)

    print("Running TimesFM backtest...")
    timesfm_results = backtest(df, model)

    print_backtest_comparison(timesfm_results, persistence_results)

    print("\nGenerating current 1-hour forecast per station...")
    forecasts = forecast_current(df, model)
    print()
    for station_id in forecasts["station_id"].unique():
        sdf = forecasts[forecasts["station_id"] == station_id]
        print(f"Station {station_id}:")
        print(sdf[["variable", "horizon_step", "predicted", "q10", "q90"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
