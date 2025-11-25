with local_stations as (
    select
        from_station.name as from_station_name,
        to_station.name as to_station_name,
        from_station.id as from_id,
        to_station.id as to_id,
        dist.km_between
    from
        station_distances as dist
        join stations as from_station on from_station.id = dist.id_from
        join stations as to_station on to_station.id = dist.id_to
    where
        from_station.name = 'Coronet Tandems'
        and dist.km_between <= 20
        order by dist.km_between
    ),
    station_ids as (
        select distinct from_id as station_id
        from local_stations

        union

        select distinct to_id
        from local_stations
    )

select
    o.id,
    o.timestamp,
    o.temperature,
    o.wind_average,
    o.wind_gust,
    o.wind_bearing
from observations as o
join station_ids as id on id.station_id = o.id
