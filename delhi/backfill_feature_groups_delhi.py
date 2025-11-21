import glob
import datetime as dt
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import hopsworks

# CONFIG 
CITY_NAME = "delhi"
DELHI_LAT = 28.6448
DELHI_LON = 77.2167

load_dotenv()
WAQI_TOKEN = os.getenv("WAQI_TOKEN")


def parse_uid_from_filename(path: str) -> int:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    uid_str = stem.split("_")[-1]
    return int(uid_str)


def fetch_sensor_metadata(uid: int):
    if WAQI_TOKEN is None:
        raise RuntimeError("WAQI_TOKEN environment variable is not set")

    url = f"https://api.waqi.info/feed/@A{uid}/?token={WAQI_TOKEN}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"WAQI status not ok for uid={uid}: {data}")

    city = data["data"]["city"]
    lat = float(city["geo"][0])
    lon = float(city["geo"][1])
    station_name = city["name"]

    return station_name, lat, lon


def load_air_quality_multi():
    all_rows = []

    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, "data", "delhi_sensor_*.csv")

    csv_paths = glob.glob(pattern)
    if not csv_paths:
        raise RuntimeError("No files matching data/delhi_sensor_*.csv found")

    for csv_path in csv_paths:
        try:
            sensor_uid = parse_uid_from_filename(csv_path)
        except Exception as e:
            print(f"Could not parse uid from {csv_path}: {e}, skipping.")
            continue

        try:
            station_name, lat, lon = fetch_sensor_metadata(sensor_uid)
        except Exception as e:
            print(f"Could not fetch metadata for uid={sensor_uid}: {e}, skipping.")
            continue

        print(f"Loading {csv_path} for sensor {sensor_uid} ({station_name})")

        df = pd.read_csv(csv_path)

        expected_cols = ["date", "min", "max", "median", "q1", "q3", "stdev", "count"]

        df = df[expected_cols]

        df["date"] = pd.to_datetime(df["date"]).dt.date

        df.rename(columns={"median": "pm2_5"}, inplace=True)

        df["city_name"] = CITY_NAME
        df["sensor_uid"] = sensor_uid
        df["station_name"] = station_name
        df["lat"] = lat
        df["lon"] = lon

        df = df[
            [
                "city_name",
                "sensor_uid",
                "station_name",
                "lat",
                "lon",
                "date",
                "pm2_5",
            ]
        ].dropna(subset=["pm2_5"])

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No AQ CSVs successfully loaded for Delhi.")

    aq_df = pd.concat(all_rows, ignore_index=True)
    return aq_df


def fetch_weather(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    today = dt.date.today()
    hist_end = min(end_date, today - dt.timedelta(days=1))

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": DELHI_LAT,
        "longitude": DELHI_LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": hist_end.strftime("%Y-%m-%d"),
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "windspeed_10m_max",
                "winddirection_10m_dominant",
            ]
        ),
        "timezone": "auto",
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    weather_df = pd.DataFrame(
        {
            "date": pd.Series(pd.to_datetime(daily["time"])).dt.date,
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "wind_speed_max": daily["windspeed_10m_max"],
            "wind_direction_dominant": daily["winddirection_10m_dominant"],
        }
    )

    weather_df["city_name"] = CITY_NAME

    weather_df = weather_df[
        [
            "city_name",
            "date",
            "temp_max",
            "temp_min",
            "wind_speed_max",
            "wind_direction_dominant",
        ]
    ]

    return weather_df


def write_to_hopsworks(aq_df, weather_df):
    project = hopsworks.login()
    fs = project.get_feature_store()

    weather_fg = fs.get_or_create_feature_group(
        name="weather_delhi",
        version=1,
        description="Historical weather for Delhi",
        primary_key=["city_name", "date"],
        event_time="date",
    )

    aq_fg = fs.get_or_create_feature_group(
        name="air_quality_delhi",
        version=1,
        description="Historical PM2.5 for all Delhi sensors",
        primary_key=["city_name", "sensor_uid", "date"],
        event_time="date",
    )

    print("Inserting weather...")
    weather_fg.insert(weather_df)

    print("Inserting air quality...")
    aq_fg.insert(aq_df)

    print("Delhi backfill completed successfully.")

aq_df = load_air_quality_multi()
start_date = aq_df["date"].min()
end_date = aq_df["date"].max()

weather_df = fetch_weather(start_date, end_date)

write_to_hopsworks(aq_df, weather_df)
