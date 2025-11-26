CREATE TABLE IF NOT EXISTS station_distances (
    id_from TEXT,
    id_to TEXT,
    km_between REAL,
    PRIMARY KEY (id_from, id_to)
)
