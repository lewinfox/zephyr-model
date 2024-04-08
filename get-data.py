import asyncio
import datetime
import os
import re
import sys

import aiohttp
import requests

ZEPHYR_DATASTORE_KEY = os.environ["ZEPHYR_DATASTORE_KEY"]
ZEPHYR_FUNCTION_URL = (
    "https://australia-southeast1-zephyr-3fb26.cloudfunctions.net/output"
)


def _parse_date(x: int | str | datetime.datetime | None) -> int | None:
    """
    Parse date

    `x` can be a `datetime.datetime` object, a string in the format
    `yyyy-MM-ddThh:mm:ss` in New Zealand time, or a UNIX timestamp.

    ### Returns

    `int` : POSIX timestamp
    """

    allowed_timestamp_formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]

    if x is None or isinstance(x, int):
        return x
    elif isinstance(x, str):
        for fmt in allowed_timestamp_formats:
            try:
                ts = datetime.datetime.strptime(x, fmt)
                return int(ts.timestamp())
            except Exception as e:
                continue

    raise ValueError(f"Unable to interpret {x} as a timestamp")


def _get_download_urls(
    from_date: int | str | datetime.datetime,
    to_date: int | str | datetime.datetime | None = None,
) -> dict:

    from_date = _parse_date(from_date)  # type: ignore
    to_date = _parse_date(to_date)

    # Request download URLs for all relevant files
    res = requests.get(
        ZEPHYR_FUNCTION_URL,
        params={"key": ZEPHYR_DATASTORE_KEY, "dateFrom": from_date, "dateTo": to_date},  # type: ignore
    )

    res.raise_for_status()

    res_json = res.json()

    return res_json


async def _download_file(
    session: aiohttp.ClientSession, url: str, output_file: str
) -> bytes:
    """
    Download the file at `url` to `output_file`.

    ### Returns

    `bytes`: The file content
    """
    print(f"Requesting {url}")

    async with session.get(url) as response:
        file_data = await response.read()

    with open(output_file, "wb") as f:
        f.write(file_data)

    print(f"Downloaded {output_file}")

    return file_data


async def _get_data(
    from_date: int | str | datetime.datetime,
    to_date: int | str | datetime.datetime | None = None,
    download_dir="./data",
):
    """
    Download Zephyr data
    """

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    download_paths = [x["url"] for x in _get_download_urls(from_date, to_date)]

    filename_regex = re.compile(r"zephyr-scrape-[0-9]{10}.json")
    filenames = [filename_regex.search(p).group() for p in download_paths]  # type: ignore

    async with aiohttp.ClientSession() as session:
        tasks = [
            _download_file(session, url, os.path.join(download_dir, filename))
            for url, filename in zip(download_paths, filenames)
        ]

        result = await asyncio.gather(*tasks)


if __name__ == "__main__":

    if len(sys.argv) == 2:
        from_date = sys.argv[1]
    else:
        print("Usage: python get-data.py '<from-date>'")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_get_data(from_date))
