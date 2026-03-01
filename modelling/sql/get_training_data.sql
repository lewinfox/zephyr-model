with
    local_stations as (
        select sd.id_to, sd.km_between
        from station_distances as sd
        join stations as s on s.id = sd.id_from
        where s.name = '{{ station_name }}'
        and sd.km_between <= {{ max_distance }}
    )

select o.*
from observations as o
join local_stations as ls on ls.id_to = o.station_id
