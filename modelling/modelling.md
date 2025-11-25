# Weather Forecasting Model - Approach Documentation

## Overview

This document outlines the modeling approach for building a predictive weather model to forecast temperature and wind variables (average, gust, bearing) at 10-minute intervals, with predictions extending 1-2 hours into the future.

## Recommended Library: tsai

Instead of the standard FastAI tabular learner, we're using **[tsai](https://github.com/timeseriesAI/tsai)** - a library built on top of FastAI/PyTorch specifically designed for time series tasks. It provides high-level APIs similar to FastAI's approach while being optimized for time series forecasting.

### Why tsai?

- Built on FastAI/PyTorch with familiar API patterns
- Native support for multivariate multi-step forecasting
- State-of-the-art models specifically designed for time series
- Handles missing data in time series
- Easy-to-use high-level abstractions

## Proposed Architecture

### Primary Model: PatchTST

**[PatchTST](https://timeseriesai.github.io/tsai/models.patchtst.html)** - A state-of-the-art transformer model (ICLR 2023) designed for long-term time series forecasting.

### Alternative Models

- **TSTPlus** - Transformer-based with auto-configuring output heads
- **InceptionTimePlus** - CNN-based alternative
- **TSiTPlus** - Another transformer variant

All "Plus" models auto-configure output heads to match target dimensions, making them ideal for multi-step forecasting.

## Data Structure

### Input Data Format

Our data (`data.csv`) contains:
- `id`: Station/sensor identifier (19 unique stations)
- `timestamp`: Unix timestamp (10-minute intervals)
- `temperature`: Temperature readings
- `wind_average`: Average wind speed
- `wind_gust`: Wind gust speed
- `wind_bearing`: Wind direction in degrees

Dataset size: ~600k rows with some missing values.

### Modeling Approach

#### Robustness

In the data, as in real life, stations may go offline, miss data, fail to report,
etc. It would be good if the model was trained to be robust to missing data.

#### Single Multivariate Model

Train one model that predicts all 4 variables simultaneously:
- **Input (X)**: Last N timesteps of all 4 variables (e.g., 60 timesteps = 10 hours)
- **Output (y)**: Next 6-12 timesteps of all 4 variables (1-2 hours ahead)
- **Advantage**: Captures inter-variable relationships (e.g., how wind affects temperature)

We want to predict outputs for all stations (the `id` field) in the training
data.

## Time Horizons

Given 10-minute intervals:
- **6 steps ahead** = 1 hour forecast
- **12 steps ahead** = 2 hours forecast

Start with these horizons and experiment with longer predictions to evaluate model performance degradation.

## Implementation Workflow

### 1. Data Preparation

```python
# Load and prepare data
- Load CSV into pandas
- Create temporal features (hour of day, day of week, season)
- Normalize/standardize the data
```

### 2. Create Sliding Windows

```python
# Use tsai's SlidingWindow
from tsai.data.preparation import SlidingWindow

# Example: 60 timesteps input (10 hours), 12 timesteps output (2 hours)
X, y = SlidingWindow(window_len=60, horizon=12)(ts_data)
```

### 3. Split Data

```python
# Use tsai's TimeSplitter for proper time series splits
from tsai.data.preparation import TimeSplitter

splits = TimeSplitter(valid_size=0.2)(y)
```

### 4. Build TSForecaster

```python
from tsai.tslearner import TSForecaster

fcst = TSForecaster(
    X, y,
    splits=splits,
    path='models',
    tfms=tfms,                    # Data transformations
    batch_tfms=TSStandardize(),   # Batch transformations
    bs=512,                       # Batch size
    arch="PatchTST",             # or "TSTPlus"
    metrics=[mae, rmse],         # Evaluation metrics
    cbs=ShowGraph()              # Callbacks
)
```

### 5. Train Model

```python
# Use FastAI's one-cycle training
fcst.fit_one_cycle(n_epoch=50, lr_max=1e-3)

# Export trained model
fcst.export("weather_forecast.pkl")
```

### 6. Evaluate & Iterate

- Evaluate predictions at different horizons (1 hour vs 2 hours)
- Visualize predictions vs actual values
- Calculate metrics (MAE, RMSE) per variable
- Experiment with:
  - Different window sizes (30, 60, 120 timesteps)
  - Different horizons (6, 12, 24 steps)
  - Different architectures (PatchTST vs TSTPlus vs InceptionTimePlus)

## Handling Data Challenges

### Missing Values

- tsai has built-in support for handling missing data
- Some models can handle NaN values directly with masking

### Multiple Variables with Different Scales

- Use `TSStandardize()` batch transformation
- Normalizes each variable independently
- Automatically reversed during prediction

### Irregular Time Intervals

- Ensure timestamps are regularized to 10-minute intervals
- Fill gaps with interpolation or marking as missing

## Expected Results

### Short-term (1 hour / 6 steps)
- **Temperature**: High accuracy expected (weather changes slowly)
- **Wind**: Moderate accuracy (more volatile than temperature)
- Typical MAE: ~0.5-1°C for temperature, ~10-20% error for wind

### Medium-term (2 hours / 12 steps)
- **Temperature**: Good accuracy (gradual degradation)
- **Wind**: Lower accuracy (harder to predict)
- Performance degrades as horizon increases

### Variable Difficulty
- **Temperature**: Easiest to predict (smooth patterns)
- **Wind Bearing**: Moderate difficulty
- **Wind Gust/Average**: Hardest (high volatility)

## Key Advantages of This Approach

✅ **High-level API** - FastAI-style ease of use
✅ **State-of-the-art models** - PatchTST proven for forecasting
✅ **Multi-step native support** - Built for your exact use case
✅ **Multivariate support** - Handles multiple variables naturally
✅ **FastAI ecosystem** - Familiar patterns (learners, callbacks, one-cycle)
✅ **Production ready** - Easy model export and deployment

## Next Steps

1. Install tsai: `pip install tsai` or add to requirements
2. Create exploratory notebook to examine data quality and patterns
3. Implement basic single-station, single-variable model as proof of concept
4. Expand to multivariate model
5. Experiment with different configurations
6. Evaluate and compare results across time horizons

## References

- [tsai GitHub Repository](https://github.com/timeseriesAI/tsai)
- [tsai Documentation](https://timeseriesai.github.io/tsai/)
- [PatchTST Model Documentation](https://timeseriesai.github.io/tsai/models.patchtst.html)
- [PatchTST Tutorial Notebook](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/15_PatchTST_a_new_transformer_for_LTSF.ipynb)
- [tsai Forecasting Discussion](https://github.com/timeseriesAI/tsai/discussions/125)
- [Walk with FastAI - Time Series Lesson](https://walkwithfastai.com/TimeSeries)
- [Multi-step Forecasting Strategies](https://machinelearningmastery.com/multi-step-time-series-forecasting/)
