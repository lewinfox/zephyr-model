# zephyr-model

Predictive model for [zephyrapp.nz](https://zephyrapp.nz).

## Data download

* Set `ZEPHYR_DATASTORE_KEY` env var
* Run `python get-data.py "<from date>"` e.g. `python get-data.py "2023-01-01"`.

This will download JSON files to `./data`.


## Data prep

Run `python compile-data.py`. This will create and populate a sqlite db
containing `stations`, `observations` and `station_distances` tables.

`station_distances` stores the distance in km between each pair of stations, so
we can do things like "find all the stations within 10km of this one".
