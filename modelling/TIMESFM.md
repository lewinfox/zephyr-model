# TimesFM Zero-Shot Weather Forecasting

[TimesFM](https://github.com/google-research/timesfm) is Google's pretrained time-series foundation model. This integration lets you generate weather forecasts **without any training** — just load the model and predict.

## How it differs from PatchTST

| | PatchTST (current) | TimesFM |
|---|---|---|
| Training required | Yes (~20 min on GPU) | **No** |
| Variables | Multivariate (all 4 at once) | Univariate (one at a time) |
| Cross-variable relationships | Learned | Not captured |
| Uncertainty estimates | No | Yes (quantile bands) |
| New data adaptation | Retrain | Just run |

## Installation

```bash
uv add timesfm
# or: pip install timesfm
```

The model weights (~800 MB) are downloaded from HuggingFace automatically on first use and cached locally.

## Quick start

```python
from timesfm_forecast import load_model, forecast_current
from train import get_training_data

df = get_training_data("Coronet Tandems", 20)
model = load_model()
forecasts = forecast_current(df, model)
print(forecasts)
```

## Backtesting

The backtest rolls a sliding window through the last 20% of each station's history, comparing TimesFM forecasts to actual observations. A naive persistence baseline (predict = last known value) is included for comparison.

```python
from timesfm_forecast import load_model, backtest, persistence_backtest, print_backtest_comparison
from train import get_training_data

df = get_training_data("Coronet Tandems", 20)
model = load_model()

pers = persistence_backtest(df)
tfm  = backtest(df, model)
print_backtest_comparison(tfm, pers)
```

Example output:
```
====================================================================
BACKTEST  –  TimesFM vs Persistence Baseline (MAE, lower is better)
====================================================================
Variable                   TimesFM   Persistence          Δ
--------------------------------------------------------------------
  temperature                0.412         0.631    ▲  34.7%
  wind_average               3.821         4.102    ▲   6.8%
  wind_gust                  5.103         5.891    ▲  13.4%
  wind_bearing              18.204        22.310    ▲  18.4%
====================================================================
```
*(Numbers are illustrative — run the backtest to see real results.)*

## Interactive notebook

Open `timesfm_forecast.ipynb` for a step-by-step walkthrough with charts:

```bash
jupyter notebook modelling/timesfm_forecast.ipynb
```

## Run from command line

```bash
cd modelling
python timesfm_forecast.py
```

This runs the full pipeline: persistence baseline → TimesFM backtest → comparison table → current forecast for all stations.

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `context_len` | 60 | Past timesteps fed to the model (10 hours) |
| `horizon` | 6 | Steps to forecast (1 hour) |
| `stride` | 6 | Gap between backtest evaluation points |
| `test_fraction` | 0.2 | Fraction of history held out for backtesting |

Increasing `context_len` (TimesFM supports up to 16 384 steps) may improve accuracy for variables with strong daily cycles like temperature.
