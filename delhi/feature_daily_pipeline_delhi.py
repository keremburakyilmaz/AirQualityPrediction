import os
import glob
import datetime as dt
import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# CONFIG
load_dotenv()
CITY_NAME = "delhi"
DELHI_LAT = 28.6448
DELHI_LON = 77.2167

WAQI_TOKEN = os.getenv("WAQI_TOKEN")

def get_sensor_uids():
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, "data", "delhi_sensor_*.csv")
    csv_paths = glob.glob(pattern)
    
    uids = []
    for path in csv_paths:
        base = os.path.basename(path)
        try:
            stem, _ = os.path.splitext(base)
            uid_str = stem.split("_")[-1]
            uids.append(int(uid_str))
        except Exception as e:
            print(f"Skipping file {base}: {e}")
            
    return sorted(list(set(uids)))


def fetch_latest_air_quality(sensor_uid: int):
    try:
        url = f"https://api.waqi.info/feed/@A{sensor_uid}/?token={WAQI_TOKEN}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            print(f"Warning: WAQI status not ok for uid={sensor_uid}: {data}")
            return None

        d = data["data"]

        # Extract timestamp and date
        ts_str = d["time"]["s"]
        ts = dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        date = ts.date()

        # pm25 value
        if "pm25" not in d.get("iaqi", {}):
            print(f"No PM2.5 data for uid={sensor_uid}")
            return None
        
        pm25 = d["iaqi"]["pm25"]["v"]

        # Extract metadata
        city = d.get("city", {})
        geo = city.get("geo", [None, None])
        lat = float(geo[0]) if geo[0] is not None else None
        lon = float(geo[1]) if geo[1] is not None else None
        station_name = city.get("name", f"Sensor {sensor_uid}")

        aq_df = pd.DataFrame(
            {
                "city_name": [CITY_NAME],
                "sensor_uid": [sensor_uid],
                "station_name": [station_name],
                "lat": [lat],
                "lon": [lon],
                "date": [date],
                "pm2_5": [pm25],
            }
        )
        aq_df["pm2_5"] = aq_df["pm2_5"].astype(float)
        return aq_df
        
    except Exception as e:
        print(f"Error parsing data for uid={sensor_uid}: {e}")
        return None


def fetch_weather_window(today: dt.date, days_forecast: int = 10):
    start_date = today - dt.timedelta(days=1)
    end_date = today + dt.timedelta(days=days_forecast)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": DELHI_LAT,
        "longitude": DELHI_LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
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

    dates = pd.to_datetime(daily["time"]).date

    weather_df = pd.DataFrame(
        {
            "date": dates,
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "wind_speed_max": daily["windspeed_10m_max"],
            "wind_direction_dominant": daily["winddirection_10m_dominant"],
        }
    )

    weather_df["city_name"] = CITY_NAME

    return weather_df[
        ["city_name", "date", "temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]
    ]


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

    print("Inserting weather rows...")
    weather_fg.insert(weather_df)

    if not aq_df.empty:
        print("Inserting air quality rows...")
        aq_fg.insert(aq_df)
    else:
        print("No air quality data to insert.")

    print("Daily feature pipeline finished.")


today = dt.date.today()
print(f"Running daily pipeline for Delhi: {today}")

# Identify sensors
uids = get_sensor_uids()
print(f"Found {len(uids)} sensors: {uids}")

# Fetch AQ for each sensor
aq_dfs = []
for uid in uids:
    df = fetch_latest_air_quality(uid)
    if df is not None:
        aq_dfs.append(df)
        
if aq_dfs:
    all_aq_df = pd.concat(aq_dfs, ignore_index=True)
    print(f"Fetched AQ data for {len(aq_dfs)} sensors.")
else:
    all_aq_df = pd.DataFrame()
    print("No AQ data fetched successfully.")

# Fetch Weather
weather_df = fetch_weather_window(today, days_forecast=10)
print(f"Fetched weather forecast for {len(weather_df)} days.")

# Write to Hopsworks
write_to_hopsworks(all_aq_df, weather_df)
