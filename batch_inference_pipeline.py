import os
import datetime as dt

import pandas as pd
import hopsworks
import tensorflow as tf
import matplotlib.pyplot as plt

CITY_NAME = "stockholm"
HORIZON_DAYS = 7  # forecast next 7 days
WEATHER_VERSION = 2 
AIR_QUALITY_VERSION = 2
MODEL_NAME = "pm25_keras_model"
MODEL_VERSION = 4

FEATURE_COLS = [
    "temp_max",
    "temp_min",
    "wind_speed_max",
    "wind_direction_dominant",
    "day_of_week",
    "month",
    "day_of_year",
    "pm2_5_prev",
    "pm2_5_prev_2",
    "pm2_5_prev_3",
]

def get_future_weather(fs):
    weather_fg = fs.get_feature_group(name="weather", version=WEATHER_VERSION)

    df = weather_fg.read()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    today = dt.date.today()
    end_date = today + dt.timedelta(days=HORIZON_DAYS)

    mask = (
        (df["date"] >= today)
        & (df["date"] <= end_date)
    )

    future_df = df.loc[mask, ["date", "temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]].copy().sort_values("date")

    return future_df

def get_latest_pm25_history(fs, n_days=3):
    aq_fg = fs.get_feature_group(name="air_quality", version=AIR_QUALITY_VERSION)
    df = aq_fg.read()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")

    last_rows = df.tail(n_days)
    pm25_values = last_rows["pm2_5"].tolist()
    
    # Pad with None if we don't have enough history
    while len(pm25_values) < n_days:
        pm25_values.insert(0, None)
    
    prev_1 = float(pm25_values[-1]) if pm25_values[-1] is not None else None
    prev_2 = float(pm25_values[-2]) if len(pm25_values) >= 2 and pm25_values[-2] is not None else None
    prev_3 = float(pm25_values[-3]) if len(pm25_values) >= 3 and pm25_values[-3] is not None else None
    
    return prev_1, prev_2, prev_3

def load_keras_model(project):
    mr = project.get_model_registry()

    model_meta = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
    local_dir = model_meta.download()

    print(f"Loaded model from: {local_dir}")

    model = tf.keras.models.load_model(local_dir)
    return model


def build_initial_features(future_weather_df, prev_1, prev_2, prev_3):
    rows = []
    lag_history = [prev_3, prev_2, prev_1]  # [t-3, t-2, t-1]
    lag_history = [v for v in lag_history if v is not None]  # Remove None values

    for _, row in future_weather_df.iterrows():
        d = row["date"]
        d_ts = pd.to_datetime(d)

        pm2_5_prev = lag_history[-1] if len(lag_history) >= 1 else prev_1 if prev_1 is not None else 0.0
        pm2_5_prev_2 = lag_history[-2] if len(lag_history) >= 2 else prev_2 if prev_2 is not None else 0.0
        pm2_5_prev_3 = lag_history[-3] if len(lag_history) >= 3 else prev_3 if prev_3 is not None else 0.0

        feat_row = {
            "date": d,
            "temp_max": float(row["temp_max"]),
            "temp_min": float(row["temp_min"]),
            "wind_speed_max": float(row["wind_speed_max"]),
            "wind_direction_dominant": float(row["wind_direction_dominant"]),
            "day_of_week": float(d_ts.weekday()),
            "month": float(d_ts.month),
            "day_of_year": float(d_ts.dayofyear),
            "pm2_5_prev": float(pm2_5_prev),
            "pm2_5_prev_2": float(pm2_5_prev_2),
            "pm2_5_prev_3": float(pm2_5_prev_3),
        }
        rows.append(feat_row)

    df = pd.DataFrame(rows)
    return df


def autoregressive_forecast(model, features_df):
    df = features_df.copy()
    preds = []
    pm_prev_used = []

    for i in range(len(df)):
        # Update lag features with previous predictions
        if i > 0:
            df.loc[df.index[i], "pm2_5_prev"] = preds[-1]
            if i > 1:
                df.loc[df.index[i], "pm2_5_prev_2"] = preds[-2]
            if i > 2:
                df.loc[df.index[i], "pm2_5_prev_3"] = preds[-3]

        X_row = df.loc[df.index[i], FEATURE_COLS].values.astype("float32").reshape(1, -1)
        y_hat = float(model.predict(X_row, verbose=0)[0][0])

        preds.append(y_hat)
        pm_prev_used.append(float(df.loc[df.index[i], "pm2_5_prev"]))

    df["pm2_5_prev_used"] = pm_prev_used
    df["pm2_5_pred"] = preds

    return df


def plot_forecast(df):
    os.makedirs("docs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["date"], df["pm2_5_pred"], marker="o")

    ax.set_title(f"PM2.5 forecast for {CITY_NAME} (next {HORIZON_DAYS} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted PM2.5")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = os.path.join("docs", "pm25_forecast.png")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved forecast plot to {out_path}")


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    future_weather = get_future_weather(fs)
    if future_weather.empty:
        print("No future weather rows found. Run feature_daily_pipeline first.")
        return

    prev_1, prev_2, prev_3 = get_latest_pm25_history(fs, n_days=3)
    if prev_1 is None:
        print("No recent PM2.5 data found. Run feature_daily_pipeline first.")
        return

    model = load_keras_model(project)

    features_df = build_initial_features(future_weather, prev_1, prev_2, prev_3)
    forecast_df = autoregressive_forecast(model, features_df)

    print("\nForecast results:")
    print(forecast_df[["date", "pm2_5_prev_used", "pm2_5_pred"]])

    plot_forecast(forecast_df)

    print("\nBatch inference with lag feature finished.")


if __name__ == "__main__":
    main()