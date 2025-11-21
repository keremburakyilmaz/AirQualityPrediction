import os
import glob
import datetime as dt
import pandas as pd
import hopsworks
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CONFIG
CITY_NAME = "delhi"
WEATHER_VERSION = 1
AIR_QUALITY_VERSION = 1
HINDCAST_DAYS = 10

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

def get_historical_weather(fs, start_date, end_date):
    weather_fg = fs.get_feature_group(name="weather_delhi", version=WEATHER_VERSION)
    df = weather_fg.read()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    hist_df = df.loc[mask, ["date", "temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]].copy().sort_values("date")
    return hist_df

def get_historical_pm25(fs, sensor_uid, start_date, end_date):
    aq_fg = fs.get_feature_group(name="air_quality_delhi", version=AIR_QUALITY_VERSION)
    df = aq_fg.read()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    sensor_df = df[df["sensor_uid"] == sensor_uid].sort_values("date")
    mask = (sensor_df["date"] >= start_date) & (sensor_df["date"] <= end_date)
    hist_df = sensor_df.loc[mask, ["date", "pm2_5"]].copy().sort_values("date")
    return hist_df

def get_pm25_history_at_date(fs, sensor_uid, target_date, n_days=3):
    aq_fg = fs.get_feature_group(name="air_quality_delhi", version=AIR_QUALITY_VERSION)
    df = aq_fg.read()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    sensor_df = df[df["sensor_uid"] == sensor_uid].sort_values("date")
    mask = sensor_df["date"] <= target_date
    filtered = sensor_df.loc[mask]
    
    if filtered.empty:
        return None, None, None
    
    last_rows = filtered.tail(n_days)
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
        if not models:
            return None
        
        best_model = max(models, key=lambda m: m.version)
        local_dir = best_model.download()
        model = tf.keras.models.load_model(local_dir)
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def build_features_for_date(weather_row, prev_1, prev_2, prev_3):
    d = weather_row["date"]
    d_ts = pd.to_datetime(d)
    
    pm2_5_prev = float(prev_1) if prev_1 is not None else 0.0
    pm2_5_prev_2 = float(prev_2) if prev_2 is not None else 0.0
    pm2_5_prev_3 = float(prev_3) if prev_3 is not None else 0.0
    
    feat_row = {
        "date": d,
        "temp_max": float(weather_row["temp_max"]),
        "temp_min": float(weather_row["temp_min"]),
        "wind_speed_max": float(weather_row["wind_speed_max"]),
        "wind_direction_dominant": float(weather_row["wind_direction_dominant"]),
        "day_of_week": float(d_ts.weekday()),
        "month": float(d_ts.month),
        "day_of_year": float(d_ts.dayofyear),
        "pm2_5_prev": pm2_5_prev,
        "pm2_5_prev_2": pm2_5_prev_2,
        "pm2_5_prev_3": pm2_5_prev_3,
    }
    return feat_row

def predict_for_date(model, weather_row, prev_1, prev_2, prev_3):
    feat_row = build_features_for_date(weather_row, prev_1, prev_2, prev_3)
    X_row = pd.DataFrame([feat_row])[FEATURE_COLS].values.astype("float32")
    y_hat = float(model.predict(X_row, verbose=0)[0][0])
    return y_hat

def hindcast_sensor(fs, project, sensor_uid, start_date, end_date):
    """Perform hindcast for a single sensor"""
    print(f"\nHindcasting for Sensor {sensor_uid}...")
    
    model = load_model_for_sensor(project, sensor_uid)
    if model is None:
        print(f"No model found for sensor {sensor_uid}")
        return None
    
    # Get historical weather and actual PM2.5
    weather_df = get_historical_weather(fs, start_date, end_date)
    actual_df = get_historical_pm25(fs, sensor_uid, start_date, end_date)
    
    if weather_df.empty or actual_df.empty:
        print(f"Insufficient data for sensor {sensor_uid}")
        return None
    
    # Merge on date
    merged = weather_df.merge(actual_df, on="date", how="inner").sort_values("date")
    
    if merged.empty:
        return None
    
    predictions = []
    actuals = []
    dates = []
    
    # For each date, simulate what prediction would have been made
    for idx, row in merged.iterrows():
        date = row["date"]
        actual_pm25 = row["pm2_5"]
        
        # Get the PM2.5 history (1, 2, 3 days before) - what would have been available for prediction
        prev_date = date - dt.timedelta(days=1)
        prev_1, prev_2, prev_3 = get_pm25_history_at_date(fs, sensor_uid, prev_date, n_days=3)
        
        if prev_1 is None:
            continue
        
        # Make prediction using weather and lag features
        weather_row = row[["date", "temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]]
        pred_pm25 = predict_for_date(model, weather_row, prev_1, prev_2, prev_3)
        
        predictions.append(pred_pm25)
        actuals.append(actual_pm25)
        dates.append(date)
    
    if not predictions:
        return None
    
    result_df = pd.DataFrame({
        "date": dates,
        "predicted": predictions,
        "actual": actuals,
        "sensor_uid": sensor_uid
    })
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    print(f"Sensor {sensor_uid} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    result_df["rmse"] = rmse
    result_df["mae"] = mae
    
    return result_df

def plot_hindcast(all_results):
    """Plot hindcast results for all sensors"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    n_sensors = len(all_results)
    if n_sensors == 0:
        print("No hindcast data to plot")
        return
    
    # Create subplots - arrange in a grid
    cols = 3
    rows = (n_sensors + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if n_sensors == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (sensor_uid, result_df) in enumerate(all_results.items()):
        ax = axes[idx]
        
        ax.plot(result_df["date"], result_df["predicted"], marker="o", label="Predicted", linewidth=2, markersize=4, alpha=0.7)
        ax.plot(result_df["date"], result_df["actual"], marker="s", label="Actual", linewidth=2, markersize=4, alpha=0.7)
        
        rmse = result_df["rmse"].iloc[0]
        mae = result_df["mae"].iloc[0]
        
        ax.set_title(f"Sensor {sensor_uid} - RMSE: {rmse:.2f}, MAE: {mae:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("PM2.5", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # Hide unused subplots
    for idx in range(n_sensors, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"PM2.5 Hindcast: Predictions vs Actuals ({CITY_NAME.capitalize()})", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    out_path = os.path.join(docs_dir, "pm25_hindcast_delhi.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved hindcast plot to {out_path}")

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    end_date = dt.date.today() - dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=HINDCAST_DAYS)
    
    print(f"Hindcast period: {start_date} to {end_date}")
    
    uids = get_sensor_uids()
    print(f"Running hindcast for sensors: {uids}")
    
    all_results = {}
    
    for uid in uids:
        result_df = hindcast_sensor(fs, project, uid, start_date, end_date)
        if result_df is not None and not result_df.empty:
            all_results[uid] = result_df
    
    if all_results:
        plot_hindcast(all_results)
        
        for sensor_uid, result_df in all_results.items():
            rmse = result_df["rmse"].iloc[0]
            mae = result_df["mae"].iloc[0]
            print(f"Sensor {sensor_uid}: RMSE={rmse:.2f}, MAE={mae:.2f}, Samples={len(result_df)}")
    else:
        print("\nNo hindcast results generated.")

if __name__ == "__main__":
    main()

