import os
import glob
import datetime as dt
import pandas as pd
import hopsworks
import tensorflow as tf
import matplotlib.pyplot as plt

# CONFIG
CITY_NAME = "delhi"
HORIZON_DAYS = 7
WEATHER_VERSION = 1
AIR_QUALITY_VERSION = 1

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
        except Exception:
            pass
    return sorted(list(set(uids)))

def get_future_weather(fs):
    weather_fg = fs.get_feature_group(name="weather_delhi", version=WEATHER_VERSION)
    df = weather_fg.read()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    today = dt.date.today()
    end_date = today + dt.timedelta(days=HORIZON_DAYS)
    
    mask = (df["date"] >= today) & (df["date"] <= end_date)
    future_df = df.loc[mask, ["date", "temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]].copy().sort_values("date")
    return future_df

def get_latest_pm25_history(fs, sensor_uid, n_days=3):
    aq_fg = fs.get_feature_group(name="air_quality_delhi", version=AIR_QUALITY_VERSION)
    df = aq_fg.read()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    sensor_df = df[df["sensor_uid"] == sensor_uid].sort_values("date")
    
    if sensor_df.empty:
        return None, None, None
    
    last_rows = sensor_df.tail(n_days)
    pm25_values = last_rows["pm2_5"].tolist()
    
    while len(pm25_values) < n_days:
        pm25_values.insert(0, None)
    
    prev_1 = float(pm25_values[-1]) if pm25_values[-1] is not None else None
    prev_2 = float(pm25_values[-2]) if len(pm25_values) >= 2 and pm25_values[-2] is not None else None
    prev_3 = float(pm25_values[-3]) if len(pm25_values) >= 3 and pm25_values[-3] is not None else None
    
    return prev_1, prev_2, prev_3

def load_model_for_sensor(project, sensor_uid):
    mr = project.get_model_registry()
    model_name = f"pm25_model_delhi_{sensor_uid}"
    
    try:
        models = mr.get_models(model_name)
        
        best_model = max(models, key=lambda m: m.version)
        local_dir = best_model.download()
        model = tf.keras.models.load_model(local_dir)
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

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
        
    return pd.DataFrame(rows)

def autoregressive_forecast(model, features_df):
    df = features_df.copy()
    preds = []
    
    for i in range(len(df)):
        if i > 0:
            df.loc[df.index[i], "pm2_5_prev"] = preds[-1]
            if i > 1:
                df.loc[df.index[i], "pm2_5_prev_2"] = preds[-2]
            if i > 2:
                df.loc[df.index[i], "pm2_5_prev_3"] = preds[-3]
            
        X_row = df.loc[df.index[i], FEATURE_COLS].values.astype("float32").reshape(1, -1)
        y_hat = float(model.predict(X_row, verbose=0)[0][0])
        preds.append(y_hat)
        
    df["pm2_5_pred"] = preds
    return df

def plot_forecasts(all_preds_df):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for sensor_uid in sorted(all_preds_df["sensor_uid"].unique()):
        sensor_df = all_preds_df[all_preds_df["sensor_uid"] == sensor_uid].sort_values("date")
        ax.plot(sensor_df["date"], sensor_df["pm2_5_pred"], marker="o", label=f"Sensor {sensor_uid}", linewidth=2, markersize=6)
    
    ax.set_title(f"PM2.5 Forecast for {CITY_NAME.capitalize()} - All Sensors (next {HORIZON_DAYS} days)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Predicted PM2.5", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    out_path = os.path.join(docs_dir, "pm25_forecast_delhi.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved forecast plot to {out_path}")

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    future_weather = get_future_weather(fs)
    if future_weather.empty:
        print("No future weather found.")
        return

    uids = get_sensor_uids()
    print(f"Running inference for sensors: {uids}")
    
    all_preds = []
    
    for uid in uids:
        print(f"\nProcessing Sensor {uid}...")
        
        prev_1, prev_2, prev_3 = get_latest_pm25_history(fs, uid, n_days=3)
        if prev_1 is None:
            print(f"No recent data for sensor {uid}, skipping.")
            continue
            
        model = load_model_for_sensor(project, uid)
        if model is None:
            continue
            
        features_df = build_initial_features(future_weather, prev_1, prev_2, prev_3)
        preds_df = autoregressive_forecast(model, features_df)
        
        preds_df["sensor_uid"] = uid
        preds_df["city_name"] = CITY_NAME
        
        all_preds.append(preds_df)
        
        print(f"Forecast for {uid}:")
        print(preds_df[["date", "pm2_5_pred"]])

    if all_preds:
        final_df = pd.concat(all_preds, ignore_index=True)
        plot_forecasts(final_df)
        
        print("\nInference completed for all sensors.")
    else:
        print("\nNo predictions generated.")

if __name__ == "__main__":
    main()
