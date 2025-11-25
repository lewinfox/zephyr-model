import datetime
import json

import pandas as pd


def _parse_datapoint(dp: dict) -> dict:
    """Parse a Zephyr datapoint into a dict suitable for transformation into a DataFrame"""

    data_dict = {
        "id": dp["id"],
        "name": dp["name"],
        "last_update": datetime.datetime.fromtimestamp(dp["lastUpdate"]["seconds"]),
        "is_error": (
            dp["isError"]
            if "isError" in dp
            else False  # TODO: Is this a sensible default?
        ),  # not present in `navigatus` sources?
        "is_offline": dp["isOffline"],
        "latitude": dp["coordinates"]["_lat"],
        "longitude": dp["coordinates"]["_long"],
        "elevation": dp["elevation"],  # m
        "current_average": dp["currentAverage"],  # km/h
        "current_gust": dp["currentGust"],  # km/h
        "current_bearing": dp["currentBearing"],
        "current_temperature": dp["currentTemperature"],  # Celsius
    }
    return data_dict


def parse_datapoint_json(file: str) -> pd.DataFrame:
    """Parse a Zephyr datapoint file into a Pandas DataFrame"""

    with open(file) as f:
        json_data = json.load(f)
    df_data = [_parse_datapoint(d) for d in json_data]
    df = pd.DataFrame(df_data)
    return df


if __name__ == "__main__":
    df = parse_datapoint_json("data/data.json")
    print(df)
