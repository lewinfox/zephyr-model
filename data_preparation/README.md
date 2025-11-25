# Zephyr Data Preparation

This directory contains scripts for downloading and processing weather data from the Zephyr API.

## Quick Start

The recommended way to update your data is using the incremental update script:

```bash
# Set your API key
export ZEPHYR_DATASTORE_KEY='e851c75c-fda5-489d-b42c-4db314ce11ab'

# Run incremental update (automatically detects what data you need)
python data_preparation/incremental_update.py
```

This will:
1. Check your existing database for the most recent observation
2. Download only new data from the API since that timestamp
3. Incrementally update all three tables: `observations`, `stations`, and `station_distances`

## Scripts

### `incremental_update.py` (Recommended)

The main script for downloading and processing data incrementally.

**Features:**
- Automatically detects the most recent data in your database
- Downloads only new data to minimize API usage and processing time
- Updates all three core tables incrementally
- Creates database schema automatically on first run
- Efficient batch processing with progress updates

**Usage:**

```bash
# Automatic incremental update (recommended)
python incremental_update.py

# Force update from a specific date
python incremental_update.py --from-date 2024-01-01 --force

# Download data for a specific date range
python incremental_update.py --from-date 2024-01-01 --to-date 2024-01-31 --force
```

**Arguments:**
- `--from-date`: Start date (YYYY-MM-DD, Unix timestamp, or ISO8601 string)
- `--to-date`: End date (same formats as above)
- `--force`: Force using `--from-date` even if database has newer data

### Legacy Scripts

The following scripts are kept for reference but are superseded by `incremental_update.py`:

- `get-data.py`: Downloads raw JSON files from the API
- `compile-data.py`: Processes JSON files and builds database tables
- `main.py`: Parses individual datapoint files

## Database Schema

The script creates a SQLite database (`zephyr-model.db`) with three tables:

### `observations`
Weather observations from all stations at different timestamps.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Station ID |
| name | TEXT | Station name |
| lastUpdate_seconds | INTEGER | Last update timestamp from station |
| isError | INTEGER | Error flag (0 or 1) |
| isOffline | INTEGER | Offline flag (0 or 1) |
| coordinates_lat | REAL | Latitude |
| coordinates_lon | REAL | Longitude |
| elevation | REAL | Elevation in meters |
| currentAverage | REAL | Average wind speed (km/h) |
| currentGust | REAL | Wind gust speed (km/h) |
| currentBearing | REAL | Wind direction |
| currentTemperature | REAL | Temperature (Celsius) |
| timestamp | INTEGER | Observation timestamp (from filename) |

### `stations`
Unique weather stations.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Station ID (Primary Key) |
| name | TEXT | Station name |
| coordinates_lat | REAL | Latitude |
| coordinates_lon | REAL | Longitude |

### `station_distances`
Pairwise distances between all stations using the Haversine formula.

| Column | Type | Description |
|--------|------|-------------|
| id_from | TEXT | Source station ID |
| id_to | TEXT | Destination station ID |
| km_between | REAL | Distance in kilometers |

## API Information

**Endpoint:** `https://api.zephyrapp.nz/v1/json-output`

**Parameters:**
- `key`: API key (required)
- `dateFrom`: Optional, can be a Unix timestamp or UTC ISO8601 string
- `dateTo`: Optional, same format as above

**Response:**
Returns a list of URLs pointing to JSON files containing weather observations. Each file is named `zephyr-scrape-{timestamp}.json` where the timestamp indicates when the observations were collected.

## Performance Tips

1. **Regular updates:** Run the script daily or weekly to keep downloads small
2. **Batch processing:** The script processes observations in batches of 1000 by default
3. **Indexing:** The script creates indexes on frequently queried columns for faster queries
4. **Cleanup:** Temporary files are automatically cleaned up after processing

## Troubleshooting

**No new data message:**
If you see "No new data available," your database is already up to date with the API.

**API key error:**
Make sure the `ZEPHYR_DATASTORE_KEY` environment variable is set:
```bash
export ZEPHYR_DATASTORE_KEY='your-key-here'
```

**Database locked:**
Close any other applications that might have the database open (SQLite browsers, notebooks, etc.).

**Memory issues:**
If processing very large datasets, the batch size can be adjusted in the `process_observations()` function.
