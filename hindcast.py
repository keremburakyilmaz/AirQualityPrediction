import os
import datetime as dt
import pandas as pd
import hopsworks
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CONFIG
CITY_NAME = "stockholm"
WEATHER_VERSION = 2
AIR_QUALITY_VERSION = 2
MODEL_NAME = "pm25_keras_model"
MODEL_VERSION = 4
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

def get_historical_weather(fs, start_date, end_date):
    weather_fg = fs.get_feature_group(name="weather", version=WEATHER_VERSION)
    df = weather_fg.read()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    hist_df = df.loc[mask, ["date", "temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]].copy().sort_values("date")
    return hist_df

def get_historical_pm25(fs, start_date, end_date):
    aq_fg = fs.get_feature_group(name="air_quality", version=AIR_QUALITY_VERSION)
    df = aq_fg.read()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    hist_df = df.loc[mask, ["date", "pm2_5"]].copy().sort_values("date")
    return hist_df

def get_pm25_history_at_date(fs, target_date, n_days=3):
    aq_fg = fs.get_feature_group(name="air_quality", version=AIR_QUALITY_VERSION)
    df = aq_fg.read()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    mask = df["date"] <= target_date
    filtered = df.loc[mask].sort_values("date")
    
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

def load_keras_model(project):
    mr = project.get_model_registry()
    
    try:
        model_meta = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
        local_dir = model_meta.download()
        model = tf.keras.models.load_model(local_dir)
        return model
    except Exception as e:
        print(f"Error loading model {MODEL_NAME} v{MODEL_VERSION}: {e}")
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

def hindcast(fs, project, start_date, end_date):    
    model = load_keras_model(project)
    
    # Get historical weather and actual PM2.5
    weather_df = get_historical_weather(fs, start_date, end_date)
    actual_df = get_historical_pm25(fs, start_date, end_date)
    
    if weather_df.empty or actual_df.empty:
        print(f"Insufficient data for hindcast")
        return None
    
    merged = weather_df.merge(actual_df, on="date", how="inner").sort_values("date")
    
    predictions = []
    actuals = []
    dates = []
    
    for idx, row in merged.iterrows():
        date = row["date"]
        actual_pm25 = row["pm2_5"]
        
        prev_date = date - dt.timedelta(days=1)
        prev_1, prev_2, prev_3 = get_pm25_history_at_date(fs, prev_date, n_days=3)
        
        if prev_1 is None:
            continue
        
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
    })
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
        
    result_df["rmse"] = rmse
    result_df["mae"] = mae
    
    return result_df

def plot_hindcast(result_df):
    os.makedirs("docs", exist_ok=True)
    
    if result_df is None or result_df.empty:
        print("No hindcast data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(result_df["date"], result_df["predicted"], marker="o", label="Predicted", linewidth=2, markersize=4, alpha=0.7)
    ax.plot(result_df["date"], result_df["actual"], marker="s", label="Actual", linewidth=2, markersize=4, alpha=0.7)
    
    rmse = result_df["rmse"].iloc[0]
    mae = result_df["mae"].iloc[0]
    
    ax.set_title(f"PM2.5 Hindcast: {CITY_NAME.capitalize()} - RMSE: {rmse:.2f}, MAE: {mae:.2f}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("PM2.5", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    out_path = os.path.join("docs", "pm25_hindcast_stockholm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved hindcast plot to {out_path}")

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    end_date = dt.date.today() - dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=HINDCAST_DAYS)
    
    print(f"Hindcast period: {start_date} to {end_date}")
    
    result_df = hindcast(fs, project, start_date, end_date)
    
    if result_df is not None and not result_df.empty:
        plot_hindcast(result_df)
        
        rmse = result_df["rmse"].iloc[0]
        mae = result_df["mae"].iloc[0]
        print(f"{CITY_NAME.capitalize()}: RMSE={rmse:.2f}, MAE={mae:.2f}, Samples={len(result_df)}")
    else:
        print("\nNo hindcast results generated.")

if __name__ == "__main__":
    main()

