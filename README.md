# zephyr-model

Predictive model for [zephyrapp.nz](https://zephyrapp.nz).

## Setup

* Clone repo
* Create conda env `conda create -n zephyr-model python=3.12`
* `conda activate zephyr-model`
* Set VSCode to use `zephyr-model` environment
* `python -m pip install -r requirements.txt`
* `conda env config vars set ZEPHYR_DATASTORE_KEY="my-datastore-key"`

## Data download

* Set `ZEPHYR_DATASTORE_KEY` env var
* Run `python get-data.py "<from date>"` e.g. `python get-data.py "2023-01-01"`.

This will download JSON files to `./data`.


## Data prep

Run `python compile-data.py`. This will create and populate a sqlite db
containing `stations`, `observations` and `station_distances` tables.

`station_distances` stores the distance in km between each pair of stations, so
we can do things like "find all the stations within 10km of this one".

## Database schema

### `stations`

Contains a unique list of all weather stations.

``` sql
select * from stations limit 1;
```

| index | id  | name         | coordinates_lat | coordinates_lon |
|-------|-----|--------------|-----------------|-----------------|
| 0     | 123 | Foo Bar Peak | -43.1           | 172.1           |


### `station_distances`

Contains the distance (in km) between all stations. The `id_[from|to]` fields
join to `stations.station_id`.

``` sql
select * from station_distances limit 1;
```

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
