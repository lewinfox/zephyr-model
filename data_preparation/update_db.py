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
import shutil
import sqlite3
import tempfile
from math import atan2, cos, radians, sin, sqrt
from typing import Optional

import aiohttp
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

# Configuration
SQLITE_DB = "zephyr-model.db"
ZEPHYR_DATASTORE_KEY = os.environ.get("ZEPHYR_DATASTORE_KEY")
ZEPHYR_DATASTORE_URL = "https://api.zephyrapp.nz/v1/json-output"


class Wind(BaseModel):
    average: Optional[float] = None
    gust: Optional[float] = None
    bearing: Optional[float] = None

    @field_validator("average", "gust", "bearing", mode="before")
    @classmethod
    def parse_float_or_none(cls, v):
        """Convert to float, or None if not parseable."""
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


class Observation(BaseModel):
    station_id: str
    timestamp: int
    temperature: Optional[float] = None
    wind: Wind

    @field_validator("temperature", mode="before")
    @classmethod
    def parse_float_or_none(cls, v):
        """Convert to float, or None if not parseable."""
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


class Coordinates(BaseModel):
    lat: float
    lon: float

    model_config = ConfigDict(frozen=True)  # Make it hashable


class Station(BaseModel):
    id: str
    name: str
    type: str
    coordinates: Coordinates
    elevation: Optional[float] = None

    model_config = ConfigDict(frozen=True)  # Make it hashable


class StationDistance(BaseModel):
    id_from: str
    id_to: str
    km_between: float


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
    with open(os.path.join(os.path.dirname(__file__), "sql", "observations.sql")) as f:
        sql = f.read()

    cursor.execute(sql)

    # Create index on timestamp for efficient querying
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_observations_timestamp
        ON observations(timestamp)
    """)

    # Create index on station id for efficient joins
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_observations_station_id
        ON observations(station_id)
    """)

    # Create stations table
    with open(os.path.join(os.path.dirname(__file__), "sql", "stations.sql")) as f:
        sql = f.read()

    cursor.execute(sql)

    # Create station_distances table
    with open(
        os.path.join(os.path.dirname(__file__), "sql", "station_distances.sql")
    ) as f:
        sql = f.read()

    cursor.execute(sql)

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
        print(
            f"Most recent observation: {datetime.datetime.fromtimestamp(result)} (timestamp: {result})"
        )
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


async def download_file(
    session: aiohttp.ClientSession, url: str, output_file: str
) -> str:
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
) -> str | None:
    """
    Download new data files from the Zephyr API.

    Args:
        from_date: Optional start date
        to_date: Optional end date

    Returns:
        str: Path to the temporary directory containing the downloaded files. If
        no new files are available, returns `None`.
    """
    download_paths = [x["url"] for x in get_download_urls(from_date, to_date)]

    if not download_paths:
        print("No new data to download")
        return None

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

    return temp_dir


def process_observations(
    source_file_dir: str, conn: sqlite3.Connection, batch_size: int = 1000
) -> set[Station]:
    """
    Process downloaded JSON files and insert observations into the database.

    Args:
        source_file_dir: Directory containing downloaded files
        conn: Database connection
        batch_size: Number of records to insert at once

    Return:
        set[Station]: Set of all the unique stations found in the data
    """

    files = os.listdir(source_file_dir)

    if not files:
        print("No files to process")
        return set()

    print(f"Processing {len(files)} files")

    data_batch = []
    total_records = 0
    stations: set[Station] = set()

    for file_path in files:
        with open(os.path.join(source_file_dir, file_path), "r") as f:
            observations = json.load(f)

        # Validate records
        validated_records: list[dict] = []

        for idx, row in enumerate(observations):
            try:
                station = Station(**row)
                stations.add(station)

                # We want to rename `id` to `station_id` in the observations table
                if "id" in row:
                    row["station_id"] = row.pop("id")

                validated_obs = Observation(**row)
                validated_records.append(validated_obs.model_dump())
            except Exception as e:
                print(f"ERROR: {e}\n{row}")
                raise e

        if validated_records:
            validated_df = pd.json_normalize(validated_records, sep="_")
            data_batch.append(validated_df)

        if len(data_batch) >= batch_size:
            # Concatenate and write batch
            batch_df = pd.concat(data_batch, ignore_index=True)
            batch_df.to_sql("observations", conn, if_exists="append", index=False)
            total_records += len(batch_df)
            print(
                f"Inserted batch of {len(batch_df)} observations (total: {total_records})"
            )
            data_batch = []

    # Insert remaining data
    if data_batch:
        batch_df = pd.concat(data_batch, ignore_index=True)
        batch_df.to_sql("observations", conn, if_exists="append", index=False)
        total_records += len(batch_df)
        print(
            f"Inserted final batch of {len(batch_df)} observations (total: {total_records})"
        )

    conn.commit()
    print(f"Successfully processed {total_records} total observations")

    return stations


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


def update_stations(conn: sqlite3.Connection, stations: set[Station]) -> int:
    """
    Update the stations table with any new stations from observations.

    This identifies stations in the observations table that don't exist in the
    stations table and adds them.
    """

    if len(stations) == 0:
        print("No new stations to add")
        return 0

    df = pd.json_normalize([s.model_dump() for s in stations], sep="_")

    df.to_sql("stations", conn, if_exists="replace", index=False)

    return len(df)


def update_station_distances(conn: sqlite3.Connection):
    """
    Update the station_distances table for any new station pairs.

    This computes distances only for station pairs that don't already exist in
    the station_distances table. Each distance is computed once and inserted
    as both (a,b) and (b,a).
    """
    cursor = conn.cursor()

    # Get all new station pairs that need distance calculations
    # Use id_from < id_to to get each unique pair only once
    query = """
    SELECT
        s1.id as id_from,
        s1.coordinates_lat as lat_from,
        s1.coordinates_lon as lon_from,
        s2.id as id_to,
        s2.coordinates_lat as lat_to,
        s2.coordinates_lon as lon_to
    FROM stations s1
    CROSS JOIN stations s2
    WHERE s1.id < s2.id  -- Only get each pair once (lexicographically)
    AND NOT EXISTS (
        -- Check if either direction already exists
        SELECT 1
        FROM station_distances sd
        WHERE (sd.id_from = s1.id AND sd.id_to = s2.id)
           OR (sd.id_from = s2.id AND sd.id_to = s1.id)
    )
    """

    cursor.execute(query)
    new_pairs = cursor.fetchall()

    if not new_pairs:
        print("No new station distances to compute")
        return

    print(f"Computing {len(new_pairs)} unique distances ({len(new_pairs) * 2} directional pairs)")

    # Calculate distances and create both directional entries
    new_distances = []
    for row in new_pairs:
        id_from, lat_from, lon_from, id_to, lat_to, lon_to = row
        distance = haversine_distance(lat_from, lon_from, lat_to, lon_to)

        # Add both directions
        new_distances.append({
            "id_from": id_from,
            "id_to": id_to,
            "km_between": distance,
        })
        new_distances.append({
            "id_from": id_to,
            "id_to": id_from,
            "km_between": distance,
        })

    # Insert new distances using INSERT OR IGNORE to skip duplicates
    if new_distances:
        # Manually insert with INSERT OR IGNORE instead of pandas to_sql
        cursor.executemany(
            "INSERT OR IGNORE INTO station_distances (id_from, id_to, km_between) VALUES (?, ?, ?)",
            [(d["id_from"], d["id_to"], d["km_between"]) for d in new_distances]
        )
        rows_inserted = cursor.rowcount
        conn.commit()
        print(f"Added {rows_inserted} new station distance pairs")
        print("No new station distances to compute")


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
    elif to_date is None:
        print("Downloading all available data")
    else:
        print(f"Downloading all data to {to_date}")

    # Download new data
    try:
        download_dir = await download_new_data(from_date, to_date)

        if not download_dir:
            print("\nNo new data available. Database is up to date.")
            return

        # Process observations
        print("\n" + "=" * 60)
        print("Processing Observations")
        print("=" * 60)
        stations = process_observations(download_dir, conn)

        if len(stations) == 0:
            return

        # Update stations table
        print("\n" + "=" * 60)
        print("Updating Stations")
        print("=" * 60)
        new_stations = update_stations(conn, stations)

        # Update station distances if new stations were added
        if new_stations > 0:
            print("\n" + "=" * 60)
            print("Updating Station Distances")
            print("=" * 60)
            update_station_distances(conn)

        # Clean up temporary files
        shutil.rmtree(download_dir)

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


def parse_args():
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
        """,
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

    return args


def main():
    """Main entry point for the script."""

    args = parse_args()

    # Check for API key
    if not ZEPHYR_DATASTORE_KEY:
        print("Error: ZEPHYR_DATASTORE_KEY environment variable not set")
        print("Please set it with your API key:")
        print("  export ZEPHYR_DATASTORE_KEY='your-key-here'")
        return 1

    # Run the incremental update
    asyncio.run(
        incremental_update(
            from_date=args.from_date,
            to_date=args.to_date,
            force_from_date=args.force,
        )
    )

    return 0


if __name__ == "__main__":
    main()
