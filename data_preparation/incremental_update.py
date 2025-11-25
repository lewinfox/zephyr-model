"""
Incremental data update script for Zephyr weather data.

This script efficiently updates the local SQLite database by:
1. Checking for existing data and finding the most recent observation timestamp
2. Downloading only new data from the API since that timestamp
3. Incrementally updating the observations, stations, and station_distances tables
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import sqlite3
import tempfile
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import aiohttp
import pandas as pd

# Configuration
SQLITE_DB = "zephyr-model.db"
ZEPHYR_DATASTORE_KEY = os.environ.get("ZEPHYR_DATASTORE_KEY")
ZEPHYR_DATASTORE_URL = "https://api.zephyrapp.nz/v1/json-output"


def get_db_connection():
    """Get a connection to the SQLite database."""
    return sqlite3.connect(SQLITE_DB)


def initialize_database(conn: sqlite3.Connection):
    """
    Initialize the database schema if it doesn't exist.

    Creates the three core tables: observations, stations, and station_distances.
    """
    cursor = conn.cursor()

    # Create observations table with appropriate schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id TEXT,
            name TEXT,
            lastUpdate_seconds INTEGER,
            isError INTEGER,
            isOffline INTEGER,
            coordinates_lat REAL,
            coordinates_lon REAL,
            elevation REAL,
            currentAverage REAL,
            currentGust REAL,
            currentBearing REAL,
            currentTemperature REAL,
            timestamp INTEGER
        )
    """)

    # Create index on timestamp for efficient querying
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_observations_timestamp
        ON observations(timestamp)
    """)

    # Create index on station id for efficient joins
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_observations_id
        ON observations(id)
    """)

    # Create stations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            id TEXT PRIMARY KEY,
            name TEXT,
            coordinates_lat REAL,
            coordinates_lon REAL
        )
    """)

    # Create station_distances table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS station_distances (
            id_from TEXT,
            id_to TEXT,
            km_between REAL,
            PRIMARY KEY (id_from, id_to)
        )
    """)

    conn.commit()
    print("Database schema initialized")


def get_latest_timestamp(conn: sqlite3.Connection) -> int | None:
    """
    Get the timestamp of the most recent observation in the database.

    Returns:
        int | None: Unix timestamp of the most recent observation, or None if no data exists
    """
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM observations")
    result = cursor.fetchone()[0]

    if result is not None:
        print(f"Most recent observation: {datetime.datetime.fromtimestamp(result)} (timestamp: {result})")
    else:
        print("No existing observations found")

    return result


def parse_date(x: int | str | datetime.datetime | None) -> int | None:
    """
    Parse date into Unix timestamp.

    Args:
        x: Can be a datetime object, ISO8601 string, or Unix timestamp

    Returns:
        int | None: Unix timestamp
    """
    allowed_formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]

    if x is None or isinstance(x, int):
        return x
    elif isinstance(x, datetime.datetime):
        return int(x.timestamp())
    elif isinstance(x, str):
        for fmt in allowed_formats:
            try:
                ts = datetime.datetime.strptime(x, fmt)
                return int(ts.timestamp())
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date string: {x}")

    raise ValueError(f"Unable to interpret {x} as a timestamp")


def get_download_urls(
    from_date: int | str | datetime.datetime | None = None,
    to_date: int | str | datetime.datetime | None = None,
) -> list[dict]:
    """
    Get download URLs from the Zephyr API.

    Args:
        from_date: Optional start date
        to_date: Optional end date

    Returns:
        list[dict]: List of dictionaries containing download URLs
    """
    import requests

    from_date = parse_date(from_date)
    to_date = parse_date(to_date)

    params = {"key": ZEPHYR_DATASTORE_KEY}
    if from_date is not None:
        params["dateFrom"] = from_date
    if to_date is not None:
        params["dateTo"] = to_date

    print(f"Requesting data from API (from_date={from_date}, to_date={to_date})")
    res = requests.get(ZEPHYR_DATASTORE_URL, params=params)
    res.raise_for_status()

    urls = res.json()
    print(f"Found {len(urls)} files to download")

    return urls


async def download_file(session: aiohttp.ClientSession, url: str, output_file: str) -> str:
    """
    Download a file from URL to output_file.

    Args:
        session: aiohttp session
        url: URL to download from
        output_file: Path to save file to

    Returns:
        str: Path to downloaded file
    """
    async with session.get(url) as response:
        file_data = await response.read()

    with open(output_file, "wb") as f:
        f.write(file_data)

    return output_file


async def download_new_data(
    from_date: int | str | datetime.datetime | None = None,
    to_date: int | str | datetime.datetime | None = None,
) -> list[str]:
    """
    Download new data files from the Zephyr API.

    Args:
        from_date: Optional start date
        to_date: Optional end date

    Returns:
        list[str]: List of paths to downloaded files
    """
    download_paths = [x["url"] for x in get_download_urls(from_date, to_date)]

    if not download_paths:
        print("No new data to download")
        return []

    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="zephyr_download_")
    print(f"Downloading to temporary directory: {temp_dir}")

    filename_regex = re.compile(r"zephyr-scrape-[0-9]{10}.json")
    filenames = [filename_regex.search(p).group() for p in download_paths]

    downloaded_files = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for url, filename in zip(download_paths, filenames):
            output_path = os.path.join(temp_dir, filename)
            tasks.append(download_file(session, url, output_path))

        downloaded_files = await asyncio.gather(*tasks)
        print(f"Downloaded {len(downloaded_files)} files")

    return downloaded_files


def extract_timestamp_from_filename(filename: str) -> int:
    """
    Extract Unix timestamp from a Zephyr filename.

    Args:
        filename: Filename in format 'zephyr-scrape-{timestamp}.json'

    Returns:
        int: Unix timestamp
    """
    match = re.search(r"zephyr-scrape-(\d{10}).json", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract timestamp from filename: {filename}")


def process_observations(files: list[str], conn: sqlite3.Connection, batch_size: int = 1000):
    """
    Process downloaded JSON files and insert observations into the database.

    Args:
        files: List of paths to JSON files
        conn: Database connection
        batch_size: Number of records to insert at once
    """
    if not files:
        print("No files to process")
        return

    print(f"Processing {len(files)} files")
    data_batch = []
    total_records = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        file_timestamp = extract_timestamp_from_filename(filename)

        with open(file_path, "r") as f:
            observations = json.load(f)

        # Add timestamp to each observation
        for obs in observations:
            obs["timestamp"] = file_timestamp

        # Normalize and batch
        df = pd.json_normalize(observations, sep="_")
        data_batch.append(df)

        if len(data_batch) >= batch_size:
            # Concatenate and write batch
            batch_df = pd.concat(data_batch, ignore_index=True)
            batch_df.to_sql("observations", conn, if_exists="append", index=False)
            total_records += len(batch_df)
            print(f"Inserted batch of {len(batch_df)} observations (total: {total_records})")
            data_batch = []

    # Insert remaining data
    if data_batch:
        batch_df = pd.concat(data_batch, ignore_index=True)
        batch_df.to_sql("observations", conn, if_exists="append", index=False)
        total_records += len(batch_df)
        print(f"Inserted final batch of {len(batch_df)} observations (total: {total_records})")

    conn.commit()
    print(f"Successfully processed {total_records} total observations")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance (in km) between two (lat, lon) pairs.

    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point

    Returns:
        float: Distance in kilometers
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Earth's radius in kilometers

    return distance


def update_stations(conn: sqlite3.Connection):
    """
    Update the stations table with any new stations from observations.

    This identifies stations in the observations table that don't exist in the
    stations table and adds them.
    """
    cursor = conn.cursor()

    # Find new stations (those in observations but not in stations)
    cursor.execute("""
        INSERT OR IGNORE INTO stations (id, name, coordinates_lat, coordinates_lon)
        SELECT DISTINCT id, name, coordinates_lat, coordinates_lon
        FROM observations
        WHERE id NOT IN (SELECT id FROM stations)
    """)

    new_stations = cursor.rowcount
    conn.commit()

    if new_stations > 0:
        print(f"Added {new_stations} new stations")
    else:
        print("No new stations to add")

    return new_stations


def update_station_distances(conn: sqlite3.Connection):
    """
    Update the station_distances table for any new station pairs.

    This computes distances only for station pairs that don't already exist in
    the station_distances table.
    """
    cursor = conn.cursor()

    # Get all stations
    stations_df = pd.read_sql("SELECT * FROM stations", conn)

    if len(stations_df) == 0:
        print("No stations found")
        return

    # Get existing station pairs
    existing_pairs = set()
    cursor.execute("SELECT id_from, id_to FROM station_distances")
    for row in cursor.fetchall():
        existing_pairs.add((row[0], row[1]))

    print(f"Found {len(existing_pairs)} existing station pairs")

    # Compute distances for all pairs
    new_distances = []
    for i, station1 in stations_df.iterrows():
        for j, station2 in stations_df.iterrows():
            if i != j:  # Skip same station
                pair = (station1["id"], station2["id"])
                if pair not in existing_pairs:
                    distance = haversine_distance(
                        station1["coordinates_lat"],
                        station1["coordinates_lon"],
                        station2["coordinates_lat"],
                        station2["coordinates_lon"],
                    )
                    new_distances.append({
                        "id_from": station1["id"],
                        "id_to": station2["id"],
                        "km_between": distance,
                    })

    if new_distances:
        distances_df = pd.DataFrame(new_distances)
        distances_df.to_sql("station_distances", conn, if_exists="append", index=False)
        conn.commit()
        print(f"Added {len(new_distances)} new station distance pairs")
    else:
        print("No new station distances to compute")


def cleanup_temp_files(files: list[str]):
    """
    Clean up temporary downloaded files.

    Args:
        files: List of file paths to delete
    """
    if not files:
        return

    # Get the temporary directory from the first file
    if files:
        temp_dir = os.path.dirname(files[0])
        try:
            for file in files:
                if os.path.exists(file):
                    os.remove(file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            print(f"Cleaned up temporary files from {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")


async def incremental_update(
    from_date: int | str | datetime.datetime | None = None,
    to_date: int | str | datetime.datetime | None = None,
    force_from_date: bool = False,
):
    """
    Perform an incremental update of the Zephyr database.

    Args:
        from_date: Optional start date (overrides automatic detection if force_from_date=True)
        to_date: Optional end date
        force_from_date: If True, use from_date even if database has newer data
    """
    print("=" * 60)
    print("Zephyr Incremental Data Update")
    print("=" * 60)

    # Connect to database and initialize if needed
    conn = get_db_connection()
    initialize_database(conn)

    # Determine the starting point for the update
    latest_timestamp = get_latest_timestamp(conn)

    if not force_from_date and latest_timestamp is not None:
        # Start from the most recent observation + 1 second to avoid duplicates
        from_date = latest_timestamp + 1
        print(f"Incremental update starting from timestamp: {from_date}")
    elif from_date is not None:
        from_date = parse_date(from_date)
        print(f"Using provided from_date: {from_date}")
    else:
        print("Downloading all available data (no from_date specified)")

    # Download new data
    try:
        downloaded_files = await download_new_data(from_date, to_date)

        if not downloaded_files:
            print("\nNo new data available. Database is up to date.")
            return

        # Process observations
        print("\n" + "=" * 60)
        print("Processing Observations")
        print("=" * 60)
        process_observations(downloaded_files, conn)

        # Update stations table
        print("\n" + "=" * 60)
        print("Updating Stations")
        print("=" * 60)
        new_stations = update_stations(conn)

        # Update station distances
        print("\n" + "=" * 60)
        print("Updating Station Distances")
        print("=" * 60)
        update_station_distances(conn)

        # Clean up temporary files
        cleanup_temp_files(downloaded_files)

        print("\n" + "=" * 60)
        print("Update Complete!")
        print("=" * 60)

        # Print summary statistics
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM observations")
        obs_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM stations")
        station_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM station_distances")
        distance_count = cursor.fetchone()[0]

        print(f"Total observations: {obs_count:,}")
        print(f"Total stations: {station_count:,}")
        print(f"Total station pairs: {distance_count:,}")

    finally:
        conn.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Incrementally update Zephyr weather data database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic incremental update (recommended)
  python incremental_update.py

  # Force update from a specific date
  python incremental_update.py --from-date 2024-01-01

  # Download data for a specific date range
  python incremental_update.py --from-date 2024-01-01 --to-date 2024-01-31 --force
        """
    )

    parser.add_argument(
        "--from-date",
        help="Start date (YYYY-MM-DD, Unix timestamp, or ISO8601)",
        default=None,
    )
    parser.add_argument(
        "--to-date",
        help="End date (YYYY-MM-DD, Unix timestamp, or ISO8601)",
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force using --from-date even if database has newer data",
    )

    args = parser.parse_args()

    # Check for API key
    if not ZEPHYR_DATASTORE_KEY:
        print("Error: ZEPHYR_DATASTORE_KEY environment variable not set")
        print("Please set it with your API key:")
        print("  export ZEPHYR_DATASTORE_KEY='your-key-here'")
        return 1

    # Run the incremental update
    asyncio.run(incremental_update(
        from_date=args.from_date,
        to_date=args.to_date,
        force_from_date=args.force,
    ))

    return 0


if __name__ == "__main__":
    exit(main())
