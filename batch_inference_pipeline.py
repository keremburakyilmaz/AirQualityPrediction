import os
import datetime as dt

import pandas as pd
import hopsworks
import tensorflow as tf
import matplotlib.pyplot as plt


CITY_NAME = "stockholm"
HORIZON_DAYS = 7  # forecast next 7 days
WEATHER_VERSION = 2 
MODEL_NAME = "pm25_keras_model"
MODEL_VERSION = 1 

def get_future_weather(fs):
    weather_fg = fs.get_feature_group(name="weather", version=WEATHER_VERSION)

    df = weather_fg.read()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    today = dt.date.today()
    end_date = today + dt.timedelta(days=HORIZON_DAYS)

    mask = (
        (df["city_name"] == CITY_NAME)
        & (df["date"] > today)
        & (df["date"] <= end_date)
    )

    future_df = df.loc[mask].copy().sort_values("date")

    return future_df


def load_keras_model(project):
    mr = project.get_model_registry()

    model_meta = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
    local_dir = model_meta.download()

    print(f"Loaded model from: {local_dir}")

    model = tf.keras.models.load_model(local_dir)
    return model


def plot_forecast(future_df):
    os.makedirs("docs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(future_df["date"], future_df["pm2_5_pred"], marker="o")

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

    # Get future weather features
    future_df = get_future_weather(fs)

    if future_df.empty:
        print("No future weather rows found for the next days. ")
        return

    print("Future weather rows:")
    print(future_df[["date", "temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]])

    # Load model
    model = load_keras_model(project)

    # Prepare features for prediction
    X_future = future_df[
        ["temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]
    ].astype("float32").to_numpy()

    # Predict PM2.5
    preds = model.predict(X_future).reshape(-1)
    future_df["pm2_5_pred"] = preds

    print("\nForecast:")
    print(future_df[["date", "pm2_5_pred"]])

    # Plot and save to docs/
    plot_forecast(future_df)

    print("\nBatch inference pipeline finished.")


if __name__ == "__main__":
    main()
