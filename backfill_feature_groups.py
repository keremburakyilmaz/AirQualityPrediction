import requests
import pandas as pd
import hopsworks
import datetime as dt

# CONFIG 
CITY_NAME = "stockholm"
LAT = 59.40217795
LON = 17.86533755

AIR_QUALITY_CSV = "data/air_quality_data.csv"


def load_air_quality():
    df = pd.read_csv(AIR_QUALITY_CSV)

    # Assign column names to match your CSV
    expected_cols = ["date", "min", "max", "median", "q1", "q3", "stdev", "count"]
    df = df[expected_cols]

    # Convert timestamp to datetime.date
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%dT%H:%M:%S.%fZ").dt.date

    # Rename median = pm2_5
    df.rename(columns={"median": "pm2_5"}, inplace=True)

    # Add city identifier
    df["city_name"] = CITY_NAME

    # Keep only needed columns for feature group
    aq_df = df[["city_name", "date", "pm2_5"]].dropna()

    return aq_df


def fetch_weather(start_date, end_date):
    today = dt.date.today()
    hist_end = min(end_date, today - dt.timedelta(days=1))

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": hist_end.strftime("%Y-%m-%d"),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "windspeed_10m_max",
            "winddirection_10m_dominant",
        ]),
        "timezone": "auto",
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    weather_df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).date,
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "wind_speed_max": daily["windspeed_10m_max"],
        "wind_direction_dominant": daily["winddirection_10m_dominant"],
    })

    weather_df["city_name"] = CITY_NAME

    return weather_df[
        ["city_name", "date", "temp_max", "temp_min",
         "wind_speed_max", "wind_direction_dominant"]
    ]


def write_to_hopsworks(aq_df, weather_df):
    project = hopsworks.login()
    fs = project.get_feature_store()
    """
        # --- weather FG ---
        weather_fg = fs.get_or_create_feature_group(
            name="weather",
            version=1,
            description=f"Historical weather for {CITY_NAME}",
            primary_key=["city_name"],
            event_time="date"
        )

        # --- air quality FG ---
        aq_fg = fs.get_or_create_feature_group(
            name="air_quality",
            version=1,
            description=f"Historical PM2.5 for {CITY_NAME}",
            primary_key=["city_name"],
            event_time="date"
        )
    """
    weather_fg = fs.get_or_create_feature_group(
        name="weather",
        version=2,
        description=f"Historical weather for {CITY_NAME}",
        primary_key=["city_name", "date"],
        event_time="date",
    )

    aq_fg = fs.get_or_create_feature_group(
        name="air_quality",
        version=2,
        description=f"Historical PM2.5 for {CITY_NAME}",
        primary_key=["city_name", "date"],
        event_time="date",
    )

    print("Inserting weather...")
    weather_fg.insert(weather_df)

    print("Inserting air quality...")
    aq_fg.insert(aq_df)

    print("Backfill completed successfully!")


aq_df = load_air_quality()
start_date = aq_df["date"].min()
end_date = aq_df["date"].max()

print(f"Air quality dates: {start_date} → {end_date}")

weather_df = fetch_weather(start_date, end_date)
write_to_hopsworks(aq_df, weather_df)
