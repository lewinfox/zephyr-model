CREATE TABLE IF NOT EXISTS observations (
    station_id TEXT, -- station id
    timestamp INTEGER,
    temperature REAL,
    wind_average REAL,
    wind_gust REAL,
    wind_bearing REAL,
    PRIMARY KEY (station_id, timestamp)
)
