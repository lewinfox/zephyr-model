import argparse
import json
import os
import sqlite3
from math import atan2, cos, radians, sin, sqrt

import pandas as pd

SQLITE_DB = "zephyr-model.db"
ALL_TABLES = ["observations", "stations", "station_distances"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage="python compile-data.py [table]")
    parser.add_argument("-t", "--tables", required=False, nargs="*", default=ALL_TABLES)
    args = parser.parse_args()
    return args


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance (in km) between two (lat, lon) pairs

    This is used for getting the distance between weather stations so we can,
    for example, say "get all stations within `x` km of this one".
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Radius of the Earth in kilometers
    return distance


def unique_stations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique weather stations

    ### Return

    Returns a `DataFrame` containing `id`, `name`, `coordinates_lat` and `coordinates_lon`.
    """
    df = df[["id", "name", "coordinates_lat", "coordinates_lon"]].drop_duplicates()
    return df


def distance_between_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """
    Return a `DataFrame` of all weather stations and the distance (in km) between them

    Each pair of stations appears twice, with each station in both the `from` and
    `to` positions. This is partly laziness and partly for flexibility when querying.
    """

    all_stations = stations.merge(stations, how="cross", suffixes=("_from", "_to"))

    all_stations["km_between"] = all_stations.apply(
        lambda row: haversine_distance(
            row["coordinates_lat_from"],
            row["coordinates_lon_from"],
            row["coordinates_lat_to"],
            row["coordinates_lon_to"],
        ),
        axis=1,
    )

    # Not interested in the distance from a station to itself
    res = all_stations[all_stations["km_between"] > 0.0]

    # Ignore all the columns that already exist in the `stations` table
    res = res[["id_from", "id_to", "km_between"]]

    return res


def _consolidate_files(data_dir: str = "data") -> pd.DataFrame:
    """
    Consolidate all JSON files in `data_dir` into a single `DataFrame`
    """
    data = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            with open(os.path.join(root, f), "r") as fi:
                print(f"Reading {f}")
                data.extend(json.load(fi))

    print("Compiling data")

    df = pd.json_normalize(data, sep="_")
    return df


if __name__ == "__main__":

    args = _parse_args()

    conn = sqlite3.connect(SQLITE_DB)

    df = _consolidate_files("data")

    if "observations" in args.tables:
        print("Writing `observations` table")
        df.to_sql("observations", conn, if_exists="replace")

    if "stations" in args.tables:
        print("writing `stations` table")
        stations = unique_stations(df)
        stations.to_sql("stations", conn, if_exists="replace")

    if "station_distances" in args.tables:
        print("Writing `station_distances` table")
        station_dists = distance_between_stations(stations)
        station_dists.to_sql("station_distances", conn, if_exists="replace")
