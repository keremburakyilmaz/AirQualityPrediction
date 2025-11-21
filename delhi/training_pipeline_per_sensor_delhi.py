import os
import numpy as np
import hopsworks
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

FEATURE_COLS = [
    "temp_max",
    "temp_min",
    "wind_speed_max",
    "wind_direction_dominant",
    "day_of_week",
    "month",
    "day_of_year",
    "pm2_5_prev",
]

def train_model_for_sensor(sensor_uid, df_sensor, project):
    print(f"\n--- Training model for Sensor {sensor_uid} ---")
    
    df = df_sensor.copy()
    df = df.sort_values("date")
    
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    
    # Lag feature
    df["pm2_5_prev"] = df["pm2_5"].shift(1)
    df = df.dropna(subset=["pm2_5_prev"])

    # Train/Test Split
    X_all = df[FEATURE_COLS].astype("float32")
    y_all = df["pm2_5"].astype("float32")
    
    n = len(X_all)
    split_idx = int(n * 0.8)
    
    X_train = X_all.iloc[:split_idx]
    X_test = X_all.iloc[split_idx:]
    y_train = y_all.iloc[:split_idx]
    y_test = y_all.iloc[split_idx:]

    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    
    # Model Building
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(X_train_np)
    
    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    
    _ = model.fit(
        X_train_np, y_train_np,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0 
    )
    
    # Evaluate
    preds = model.predict(X_test_np, verbose=0).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_test_np, preds)))
    mae = float(mean_absolute_error(y_test_np, preds))
    
    print(f"Sensor {sensor_uid} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Save and Register
    model_name = f"pm25_model_delhi_{sensor_uid}"
    export_dir = f"model/{model_name}"
    os.makedirs(export_dir, exist_ok=True)
    model.save(export_dir, include_optimizer=False)
    
    mr = project.get_model_registry()
    
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    
    aq_model = mr.python.create_model(
        name=model_name,
        metrics={"rmse": rmse, "mae": mae},
        description=f"Keras MLP model for Delhi Sensor {sensor_uid}",
        model_schema=model_schema,
        input_example=X_test.iloc[:1],
    )
    
    aq_model.save(export_dir)
    print(f"Registered model: {model_name}")


project = hopsworks.login()
fs = project.get_feature_store()

aq_fg = fs.get_feature_group(name="air_quality_delhi", version=1)
w_fg = fs.get_feature_group(name="weather_delhi", version=1)

print("Reading feature groups...")
aq = aq_fg.read()
w = w_fg.read()

aq["date"] = pd.to_datetime(aq["date"])
w["date"] = pd.to_datetime(w["date"])

df = aq.merge(
    w[["city_name", "date", "temp_max", "temp_min", "wind_speed_max", "wind_direction_dominant"]],
    on=["city_name", "date"],
    how="inner"
)

unique_sensors = df["sensor_uid"].unique()

for uid in unique_sensors:
    df_sensor = df[df["sensor_uid"] == uid]
    train_model_for_sensor(uid, df_sensor, project)


