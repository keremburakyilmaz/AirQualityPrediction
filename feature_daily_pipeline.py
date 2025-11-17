import os
import datetime as dt
import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# CONFIG
load_dotenv()
CITY_NAME = "stockholm"
LAT = 59.40217795
LON = 17.86533755

STATION_ID = "A65290"
token = os.getenv("WAQI_TOKEN")

def fetch_latest_air_quality():
    url = f"https://api.waqi.info/feed/{STATION_ID}/?token={token}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"WAQI API status not ok: {data}")

    d = data["data"]

    ts_str = d["time"]["s"]
    ts = dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    date = ts.date()

    # pm25 value
    pm25 = d["iaqi"]["pm25"]["v"]

    aq_df = pd.DataFrame(
        {
            "city_name": [CITY_NAME],
            "date": [date],
            "pm2_5": [pm25],
        }
    )

    aq_df["pm2_5"] = aq_df["pm2_5"].astype(float)

    return aq_df


def fetch_weather_window(today: dt.date, days_forecast: int = 10):
    start_date = today - dt.timedelta(days=1)
    end_date = today + dt.timedelta(days=days_forecast)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT,
        "longitude": LON,
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
        name="weather",
        version=2,
        description=f"Weather for {CITY_NAME}",
        primary_key=["city_name", "date"],
        event_time="date",
    )

    aq_fg = fs.get_or_create_feature_group(
        name="air_quality",
        version=2,
        description=f"PM2.5 for {CITY_NAME}",
        primary_key=["city_name", "date"],
        event_time="date",
    )

    print("Inserting weather rows...")
    weather_fg.insert(weather_df)

    print("Inserting air quality row...")
    aq_fg.insert(aq_df)

    print("Daily feature pipeline finished.")


def main():
    today = dt.date.today()
    print(f"Running daily pipeline for: {today}")

    aq_df = fetch_latest_air_quality()
    print(aq_df)

    weather_df = fetch_weather_window(today, days_forecast=10)
    print(weather_df.head())

    write_to_hopsworks(aq_df, weather_df)


if __name__ == "__main__":
    main()
