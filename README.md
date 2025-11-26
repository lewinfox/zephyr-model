# zephyr-model

Predictive model for [zephyrapp.nz](https://zephyrapp.nz).


## Setup

* [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
* Clone repo
* Run `uv sync`
* Create a file `.env` containing `ZEPHYR_DATASTORE_KEY=<your key>`


## Data download

* Set `ZEPHYR_DATASTORE_KEY` env var
* Run `make update-db` or `uv run --env-file .env python data_preparation.py`

This will download JSON files from the Zephyr API and populate a local SQLite
database `zephyr-model.db`.

## Database schema

### `observations`

| station_id | timestamp    | temperature | wind_average | wind_gust | wind_bearing |
|------------|--------------|-------------|--------------|-----------|--------------|
| ab1234     | 1712440200   | 12.34       | 56.78        | 34.78     | 359.9        |

### `stations`

Contains a unique list of all weather stations.


| id  | name         | coordinates_lat | coordinates_lon |
|-----|--------------|-----------------|-----------------|
| 123 | Foo Bar Peak | -43.1           | 172.1           |


### `station_distances`

Contains the distance (in km) between all stations. The `id_[from|to]` fields
join to `stations.id`.

| index | id_from | id_to   | km_between |
|-------|---------|---------|------------|
| 0     | ab12... | cd34... | 123.45     |

Note that each pair of stations appears twice, i.e. from A to B and from B to A,
so it doesn't matter which way round you use `from` and `to`. This is to
facilitate convenient querying at the expense of <5MB extra storage.

``` sql
/* What stations are within 5km of Coronet Summit? */

select
    from_station.name as from_station_name,
    to_station.name as to_station_name,
    dist.km_between
from
    station_distances as dist
    join stations as from_station on from_station.id = dist.id_from
    join stations as to_station on to_station.id = dist.id_to
where
    from_station.name = 'Coronet Summit'
    and dist.km_between <= 5
    order by dist.km_between;
```
