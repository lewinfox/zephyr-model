# Zephyr Weather Forecasting Model - Project Overview

## Project Mission

This repository implements a machine learning system to predict weather conditions for **paragliding safety** in New Zealand. The system downloads real-time weather data from [zephyrapp.nz](https://zephyrapp.nz), trains predictive models on historical observations, and generates hourly forecasts to help pilots make informed decisions about when to fly.

### Key Goal
Create a **reusable workflow** for predicting weather in specific geographic areas, defined as "all weather stations within an X kilometer radius of station Y". Current focus: **Coronet Peak region, Queenstown**.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZEPHYR API (zephyrapp.nz)                    │
│              Weather Station Data (10-minute intervals)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   get-data.py        │
                │   (Async Downloads)  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  /data/*.json        │
                │  45,617 files, 4.3GB │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  compile-data.py     │
                │  (Batch Processing)  │
                └──────────┬───────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │       zephyr-model.db (SQLite)      │
         ├─────────────────────────────────────┤
         │  observations      (~600k rows)     │
         │  stations          (19 stations)    │
         │  station_distances (all pairs)      │
         └─────────────┬───────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Regional SQL Query  │
            │  (e.g., corodata.sql)│
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  modelling/data.csv  │
            │  (Training Dataset)  │
            └──────────┬───────────┘
                       │
                       ▼
         ┌─────────────────────────────────┐
         │         train.py                │
         ├─────────────────────────────────┤
         │  1. Temporal Feature Extraction │
         │  2. Sliding Window Creation     │
         │  3. Train/Valid Split (80/20)   │
         │  4. PatchTST Model Training     │
         │  5. Evaluation & Export         │
         └─────────────┬───────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   models/*.pkl       │
            │   (Trained Model)    │
            └──────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  DEPLOYMENT          │
            │  (Hourly Predictions)│
            └──────────────────────┘
```

---

## Repository Structure

```
zephyr-model/
├── data/                           # Raw API downloads (45,617 JSON files)
│   └── zephyr-scrape-*.json       # Named by Unix timestamp
│
├── data_preparation/               # ETL Pipeline
│   ├── get-data.py                # Download from Zephyr API
│   ├── compile-data.py            # JSON → SQLite database
│   └── main.py                    # JSON parsing utilities
│
├── modelling/                      # Machine Learning Pipeline
│   ├── train.py                   # Complete training script ⭐
│   ├── train_basic_model.ipynb    # Interactive tutorial
│   ├── data.csv                   # Training dataset (19 stations, 600k rows)
│   ├── modelling.md               # ML architecture decisions
│   ├── models/                    # Exported model artifacts
│   └── sql/
│       └── corodata.sql           # Example: Coronet Peak area query
│
├── zephyr-model.db                # SQLite database (4.4GB)
├── pyproject.toml                 # Python dependencies
├── Makefile                       # Code quality tools (ruff)
├── .env                           # API key: ZEPHYR_DATASTORE_KEY
└── README.md                      # Setup instructions
```

---

## Workflow: From Data to Predictions

### Phase 1: Data Collection

**Tool**: `data_preparation/get-data.py`

```bash
# Download data starting from a specific date
python data_preparation/get-data.py "2024-01-01"
```

**What it does**:
- Fetches weather data from `https://api.zephyrapp.nz/v1/json-output`
- Uses API key from `.env` file: `ZEPHYR_DATASTORE_KEY`
- Async downloads with `aiohttp` for efficiency
- Saves as `zephyr-scrape-<UNIX_TIMESTAMP>.json` files
- Each file contains 10-minute interval readings from all active stations

**Output**: JSON files in `/data/` directory

---

### Phase 2: Database Creation

**Tool**: `data_preparation/compile-data.py`

```bash
# Process all JSON files into SQLite database
python data_preparation/compile-data.py
```

**What it does**:
- Reads all JSON files from `/data/`
- Batch processes in chunks of 1,000 files (memory management)
- Creates three tables in `zephyr-model.db`:

**Table 1: observations**
```sql
id, name, timestamp, temperature, wind_average, wind_gust,
wind_bearing, latitude, longitude, elevation, ...
```
- All weather readings from all stations
- ~600k rows in current dataset

**Table 2: stations**
```sql
id, name, coordinates_lat, coordinates_lon
```
- Unique weather station metadata
- 19 stations in current dataset

**Table 3: station_distances**
```sql
id_from, id_to, km_between
```
- Pairwise distances between all stations (haversine formula)
- Bidirectional entries for flexible querying
- Enables radius-based station selection

**Output**: `zephyr-model.db` SQLite database

---

### Phase 3: Regional Data Extraction

**Tool**: SQL queries (example: `modelling/sql/corodata.sql`)

```sql
-- Example: Get all stations within 30km of Coronet Summit
SELECT DISTINCT
    s.id,
    s.name,
    o.timestamp,
    o.temperature,
    o.wind_average,
    o.wind_gust,
    o.wind_bearing
FROM stations s
JOIN station_distances sd ON s.id = sd.id_from
JOIN observations o ON s.id = o.id
WHERE sd.id_to = '<CORONET_SUMMIT_STATION_ID>'
  AND sd.km_between <= 30
ORDER BY s.id, o.timestamp;
```

**Export as CSV**:
```bash
sqlite3 zephyr-model.db < modelling/sql/corodata.sql > modelling/data.csv
```

**Output**: `modelling/data.csv` - training dataset for specific region

---

### Phase 4: Model Training

**Tool**: `modelling/train.py` (main entry point ⭐)

```bash
python modelling/train.py
```

**Training Pipeline** (5 steps):

#### Step 1: Data Preparation
```python
df_clean = prepare_data(df, features, include_temporal=True)
```
- **Temporal Feature Extraction**: Creates cyclical encodings
  - `hour_sin/cos` - 24-hour cycle (time of day)
  - `day_sin/cos` - 365-day cycle (seasonal patterns)
  - `weekday_sin/cos` - 7-day cycle (weekly patterns)
- **Missing Value Handling**: Forward/backward fill per station
- **Sorting**: By station ID and timestamp
- **Validation**: Removes rows with remaining NaNs

#### Step 2: Sliding Window Creation
```python
X, y = create_windows(df_clean, window_len=60, horizon=6, features)
```
- **Input Window**: 60 timesteps (10 hours at 10-min intervals)
- **Prediction Horizon**: 6 timesteps (1 hour ahead)
- **Multi-station Support**: Uses `SlidingWindowPanel` from tsai
- **Output Shapes**:
  - `X`: (num_samples, 60, num_features)
  - `y`: (num_samples, 6, num_features)

#### Step 3: Train/Validation Split
```python
splits = get_splits(range(len(y)), valid_size=0.2, shuffle=True)
```
- **Train**: 80% of samples
- **Validation**: 20% of samples
- **Shuffle**: Random split with fixed seed for reproducibility

#### Step 4: Model Training
```python
fcst = TSForecaster(
    X, y,
    splits=splits,
    arch="PatchTST",
    batch_tfms=[TSStandardize(), Nan2Value()],
    metrics=[mae, rmse]
)
fcst.fit_one_cycle(n_epoch=20, lr_max=1e-3)
```
- **Architecture**: PatchTST (Patch Time Series Transformer)
  - State-of-the-art from ICLR 2023
  - Designed for long-term multivariate forecasting
  - Native multi-step prediction
- **Batch Transformations**:
  - `TSStandardize()` - Z-score normalization
  - `Nan2Value()` - Safety net for any remaining NaNs
- **Training**:
  - One-cycle learning rate schedule
  - 20 epochs
  - Batch size: 128
  - Max LR: 1e-3

#### Step 5: Evaluation & Export
```python
mae_per_var = calculate_mae_per_variable(preds, targets)
fcst.export("models/weather_forecast_multi_station.pkl")
```
- **Metrics**: MAE and RMSE per variable
- **Variables**:
  - Temperature (°C)
  - Wind Average (km/h)
  - Wind Gust (km/h)
  - Wind Bearing (degrees)
- **Export**: Model saved as `.pkl` file for deployment

**Configuration Options** (TrainingConfig dataclass):
```python
window_len: int = 60      # Input window length
horizon: int = 6          # Prediction horizon
valid_size: float = 0.2   # Validation split
n_epochs: int = 20        # Training epochs
lr_max: float = 1e-3      # Max learning rate
batch_size: int = 128     # Batch size
arch: str = "PatchTST"    # Model architecture
```

---

## Predicted Variables

| Variable | Unit | Characteristics | Prediction Difficulty |
|----------|------|-----------------|----------------------|
| **Temperature** | °C | Slow-changing, smooth patterns | Easy ⭐ |
| **Wind Average** | km/h | Moderate volatility | Moderate ⭐⭐ |
| **Wind Gust** | km/h | High volatility, sudden changes | Hard ⭐⭐⭐ |
| **Wind Bearing** | degrees (0-360°) | Cyclical, moderate changes | Moderate ⭐⭐ |

---

## Expected Performance

Based on `modelling/modelling.md`:

**1-Hour Ahead (6 timesteps)**:
- Temperature MAE: ~0.5-1°C
- Wind Average MAE: ~10-20% of actual value
- Wind Gust MAE: Higher due to volatility

**Confidence by Horizon**:
- **1 hour**: High confidence
- **2 hours**: Moderate confidence
- **3+ hours**: Degrades significantly

---

## Key Design Decisions

### Why PatchTST?
1. **Multivariate Support**: Predicts all 4 variables simultaneously
2. **Captures Relationships**: Wind affects temperature, etc.
3. **Efficient**: Patch-based attention mechanism
4. **Proven**: State-of-the-art results on benchmark datasets

### Why Multi-Station Training?
1. **Spatial Patterns**: Nearby stations share weather patterns
2. **Data Augmentation**: 19 stations provide more training samples
3. **Generalization**: Model learns regional dynamics
4. **Robustness**: Handles missing data from individual stations

### Why Temporal Features?
1. **Seasonality**: Day of year captures seasonal patterns
2. **Daily Cycles**: Hour of day captures temperature/wind cycles
3. **Weekly Patterns**: Day of week captures human/environmental patterns
4. **Cyclical Encoding**: sin/cos prevents discontinuity (23:59 → 00:00)

---

## Dependencies

**Core Libraries** (from `pyproject.toml`):
```toml
pandas      # Data manipulation
tsai>=0.4.1 # Time series AI (FastAI wrapper for PyTorch)
seaborn     # Visualization
ruff>=0.14.6 # Code formatting/linting
```

**Data Collection** (dev-requirements.txt):
```
aiohttp     # Async HTTP for API downloads
requests    # HTTP client
```

**Development**:
```
mypy        # Type checking
ipykernel   # Jupyter notebook support
```

**Python Version**: 3.12 (specified in `.python-version`)

**Package Manager**: UV (modern Python package manager)

---

## Configuration

### Environment Variables (`.env`)
```bash
ZEPHYR_DATASTORE_KEY=e851c75c-fda5-489d-b42c-4db314ce11ab
```
Required for API authentication in `get-data.py`.

### Project Setup
```bash
# Install UV (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Code Quality
```bash
make format    # Format with ruff
make check     # Lint with ruff
make all       # Both format and check
```

---

## Data Characteristics

**Temporal Coverage**: April 2024 onwards (based on JSON filenames)

**Sampling Rate**: 10-minute intervals

**Current Dataset**:
- **Raw JSON**: 45,617 files, 4.3GB
- **Database**: 4.4GB SQLite
- **Training CSV**: 31MB, ~600k rows, 19 stations

**Missing Data**:
- Some stations have gaps in readings
- Handled via forward/backward fill per station
- Remaining NaNs dropped before training

---

## Coronet Peak Use Case

**Location**: Queenstown, New Zealand

**Approach**:
1. Identify Coronet Peak station ID from `stations` table
2. Query `station_distances` for all stations within 30km radius
3. Extract observations for those stations from specific time period
4. Export as CSV: `modelling/data.csv`
5. Train model using `modelling/train.py`
6. Deploy model for hourly predictions

**Example Query**: See `modelling/sql/corodata.sql`

---

## Reusable Workflow Template

To create predictions for a new geographic area:

### 1. Define Geographic Area
```sql
-- In modelling/sql/area_name.sql
SELECT DISTINCT
    s.id,
    o.timestamp,
    o.temperature,
    o.wind_average,
    o.wind_gust,
    o.wind_bearing
FROM stations s
JOIN station_distances sd ON s.id = sd.id_from
JOIN observations o ON s.id = o.id
WHERE sd.id_to = '<CENTER_STATION_ID>'
  AND sd.km_between <= <RADIUS_KM>
  AND o.timestamp >= <START_DATE_UNIX>
ORDER BY s.id, o.timestamp;
```

### 2. Extract Training Data
```bash
sqlite3 zephyr-model.db < modelling/sql/area_name.sql > modelling/area_data.csv
```

### 3. Train Model
```python
# In modelling/train_area.py
from train import train_model, TrainingConfig

df = pd.read_csv("modelling/area_data.csv")
config = TrainingConfig(
    window_len=60,
    horizon=6,
    n_epochs=20
)

result = train_model(
    df=df,
    config=config,
    model_dir="models",
    model_name=f"weather_forecast_{area_name}"
)
```

### 4. Deploy for Predictions
```python
# Load trained model
from tsai.inference import load_learner
fcst = load_learner("models/weather_forecast_area_name.pkl")

# Make predictions on new data
preds, targets = fcst.get_X_preds(X_new, y_new)
```

---

## Future Enhancements

### Short-term
- [ ] Automate regional model training via CLI
- [ ] Add model versioning and tracking
- [ ] Create inference API endpoint
- [ ] Implement continuous retraining pipeline

### Medium-term
- [ ] Experiment with ensemble models
- [ ] Add uncertainty quantification
- [ ] Optimize hyperparameters per region
- [ ] Create visualization dashboard

### Long-term
- [ ] Real-time prediction service integration with zephyrapp.nz
- [ ] Mobile app for paragliding safety alerts
- [ ] Extend to other flying sites across New Zealand
- [ ] Incorporate additional data sources (satellite, radar)

---

## Quick Reference

### Common Tasks

**Download latest data**:
```bash
python data_preparation/get-data.py "$(date -d '30 days ago' '+%Y-%m-%d')"
```

**Update database**:
```bash
python data_preparation/compile-data.py
```

**Train model**:
```bash
python modelling/train.py
```

**Interactive exploration**:
```bash
jupyter notebook modelling/train_basic_model.ipynb
```

**Query stations**:
```bash
sqlite3 zephyr-model.db "SELECT * FROM stations WHERE name LIKE '%Coronet%';"
```

**Check model performance**:
```python
from tsai.inference import load_learner
fcst = load_learner("models/weather_forecast_multi_station.pkl")
# Evaluate on test set
```

---

## Contact & Support

**Project Repository**: Current directory (`/home/lewin/git/zephyr-model`)

**Data Source**: [zephyrapp.nz](https://zephyrapp.nz)

**Primary Use Case**: Paragliding safety in New Zealand

---

## Cloud Deployment with GPU

### Docker Setup

The project includes GPU-enabled Docker configuration for cloud training:

**Files**:
- `Dockerfile` - GPU-enabled PyTorch image with CUDA 12.4
- `docker-compose.yml` - Local testing and development
- `.dockerignore` - Excludes unnecessary files from container
- `requirements-train.txt` - Training-specific dependencies

### Building and Running Locally

**Prerequisites**:
- Docker installed
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support

**Build the image**:
```bash
docker build -t zephyr-model:latest .
```

**Run training**:
```bash
docker-compose up train
```

**Run Jupyter notebook** (for interactive development):
```bash
docker-compose up notebook
# Access at http://localhost:8888
```

**Custom training command**:
```bash
docker run --gpus all \
  -v $(pwd)/modelling/data.csv:/workspace/modelling/data.csv:ro \
  -v $(pwd)/models:/workspace/models \
  zephyr-model:latest \
  python modelling/train.py
```

### Cloud Platform Options

#### 1. **AWS SageMaker** (Recommended for production)

**Pros**:
- Managed infrastructure
- Built-in experiment tracking
- Easy model deployment
- Spot instance support for cost savings

**Setup**:
```bash
# Push Docker image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag zephyr-model:latest <account>.dkr.ecr.us-east-1.amazonaws.com/zephyr-model:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/zephyr-model:latest

# Create training job via SageMaker SDK
```

**Cost estimate**: ~$0.50-2.00/hour for ml.g4dn.xlarge (1 GPU)

#### 2. **Google Cloud Vertex AI**

**Pros**:
- Integrated with Google Cloud Storage
- Good for TPU workloads
- Auto-scaling options

**Setup**:
```bash
# Push to Google Container Registry
gcloud auth configure-docker
docker tag zephyr-model:latest gcr.io/<project-id>/zephyr-model:latest
docker push gcr.io/<project-id>/zephyr-model:latest

# Submit training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=zephyr-training \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/<project-id>/zephyr-model:latest
```

**Cost estimate**: ~$0.35-1.50/hour for n1-standard-4 with T4 GPU

#### 3. **Azure Machine Learning**

**Pros**:
- Good integration with Azure services
- Strong enterprise support
- Hybrid cloud options

**Setup**:
```bash
# Push to Azure Container Registry
az acr login --name <registry-name>
docker tag zephyr-model:latest <registry-name>.azurecr.io/zephyr-model:latest
docker push <registry-name>.azurecr.io/zephyr-model:latest

# Create training job via Azure ML SDK
```

**Cost estimate**: ~$0.50-2.00/hour for NC6 (1 K80 GPU)

#### 4. **Lambda Labs** (Recommended for budget)

**Pros**:
- Very cost-effective (~$0.50/hour for A100)
- Simple pricing
- No commitment required

**Cons**:
- Less managed (raw VM access)
- Limited availability during high demand

**Setup**:
```bash
# SSH into Lambda instance
ssh ubuntu@<instance-ip>

# Clone repo and run docker
git clone <repo-url>
cd zephyr-model
docker-compose up train
```

**Cost estimate**: ~$0.50-1.10/hour for RTX 6000 Ada / A100

#### 5. **Vast.ai** (Cheapest option)

**Pros**:
- Extremely cost-effective (~$0.10-0.30/hour)
- On-demand GPU marketplace
- Docker-native

**Cons**:
- Variable reliability
- Community-hosted GPUs
- May have interruptions

**Setup**: Use the web interface to deploy your Docker image

**Cost estimate**: ~$0.10-0.30/hour for RTX 3090 / RTX 4090

### Recommended Workflow

**Development**:
1. Use `docker-compose` locally for testing (if you have a GPU)
2. Or use Vast.ai for cheap experimentation

**Production Training**:
1. Use AWS SageMaker or Lambda Labs
2. Enable experiment tracking (W&B or MLflow)
3. Save models to cloud storage (S3, GCS)
4. Set up automated retraining pipeline

### Data Transfer Strategy

For cloud training, you'll need to transfer your data:

**Option 1: Include in Docker image** (for small datasets < 100MB)
```dockerfile
COPY modelling/data.csv /workspace/modelling/data.csv
```

**Option 2: Download from cloud storage** (recommended for large datasets)
```bash
# In Docker entrypoint or training script
aws s3 cp s3://zephyr-data/data.csv /workspace/modelling/data.csv
```

**Option 3: Mount cloud storage** (for very large datasets)
- AWS: S3FS or EFS
- GCP: Cloud Storage FUSE
- Azure: Blob FUSE

### GPU Optimization Tips

**Check GPU availability in container**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

**Increase batch size**: With GPU, you can increase from 128 to 512 or 1024

**Mixed precision training** (for faster training):
```python
# Add to train.py
from fastai.callback.fp16 import MixedPrecision
fcst = TSForecaster(..., cbs=[MixedPrecision()])
```

**Multi-GPU training**:
```python
# Modify docker-compose.yml to use all GPUs
devices:
  - driver: nvidia
    count: all  # Use all available GPUs
    capabilities: [gpu]
```

### Estimated Training Times

Based on current dataset (600k rows, 19 stations):

| GPU | Batch Size | Epochs | Est. Time | Cost (Lambda) |
|-----|------------|--------|-----------|---------------|
| CPU (laptop) | 128 | 20 | ~8 hours | - |
| T4 | 512 | 20 | ~30 min | ~$0.25 |
| RTX 6000 Ada | 1024 | 20 | ~15 min | ~$0.13 |
| A100 | 2048 | 20 | ~10 min | ~$0.08 |

### Continuous Training Pipeline

**Automated workflow**:
1. New data arrives from Zephyr API (daily/weekly)
2. Trigger retraining via cloud scheduler
3. Train model on GPU instance
4. Evaluate against held-out test set
5. If metrics improve, deploy new model
6. Archive old model version

**Implementation options**:
- AWS: EventBridge + Lambda + SageMaker
- GCP: Cloud Scheduler + Cloud Functions + Vertex AI
- Airflow / Prefect for orchestration

---

## Recent Updates

- **2025-11-26**:
  - Fixed timestamp parsing in `extract_temporal_features()` to correctly handle Unix timestamps in seconds
  - Added Docker configuration for GPU-enabled cloud training
  - Created comprehensive cloud deployment documentation
- **Recent commits**: Added training script, cleaned up repository, added UV package manager and Makefile

---

This document provides a comprehensive overview of the Zephyr weather forecasting pipeline. For detailed ML architecture decisions, see `modelling/modelling.md`. For setup instructions, see `README.md`.
