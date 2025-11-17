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


def main():
    # Login and get Feature Store
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Load Feature Groups
    aq_fg = fs.get_feature_group(name="air_quality", version=2)
    w_fg = fs.get_feature_group(name="weather", version=2)

    aq = aq_fg.read()
    w = w_fg.read()

    aq["date"] = pd.to_datetime(aq["date"])
    w["date"] = pd.to_datetime(w["date"])

    df = aq.merge(
        w[
            [
                "city_name",
                "date",
                "temp_max",
                "temp_min",
                "wind_speed_max",
                "wind_direction_dominant",
            ]
        ],
        on=["city_name", "date"],
        how="inner",
    )

    df["pm2_5"] = df["pm2_5"].astype("float32")
    df = df.sort_values("date")

    df["day_of_week"] = df["date"].dt.weekday      # 0–6
    df["month"] = df["date"].dt.month             # 1–12
    df["day_of_year"] = df["date"].dt.dayofyear   # 1–366

    # lag feature: yesterday's PM2.5 
    df["pm2_5_prev"] = df["pm2_5"].shift(1)

    df = df.dropna(subset=["pm2_5_prev"])


    # Get training data
    X_all = df[FEATURE_COLS].astype("float32")
    y_all = df["pm2_5"].astype("float32")

    print("Features used:", list(X_all.columns))
    print("Total samples after lagging:", len(X_all))

    # 6. Time-based train/test split (80/20)
    n = len(X_all)
    split_idx = int(n * 0.8)

    X_train = X_all.iloc[:split_idx]
    X_test = X_all.iloc[split_idx:]
    y_train = y_all.iloc[:split_idx]
    y_test = y_all.iloc[split_idx:]

    # Convert to numpy
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    
    # Build Keras model (MLP)

    # Normalization layer learns mean/std from training data
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(X_train_np)

    model = keras.Sequential(
        [
            normalizer,
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae", "mse"],
    )

    model.summary()
    
    # Train
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    _ = model.fit(
        X_train_np,
        y_train_np,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluate
    preds = model.predict(X_test_np).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_test_np, preds)))
    mae = float(mean_absolute_error(y_test_np, preds))

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

    # Save model locally
    export_dir = "model/pm25_keras_model"
    os.makedirs("model", exist_ok=True)
    model.save(export_dir, include_optimizer=False)

    # Register model in Hopsworks (generic Python model)
    mr = project.get_model_registry()

    # Use train data schema for input, label schema for output
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)

    model_schema = ModelSchema(
        input_schema=input_schema,
        output_schema=output_schema,
    )

    aq_model = mr.python.create_model(
        name="pm25_keras_model",
        metrics={"rmse": rmse, "mae": mae},
        description="Keras MLP model predicting PM2.5 from daily weather",
        model_schema=model_schema,
        input_example=X_test.iloc[:1],
    )

    # export_dir already contains the SavedModel
    aq_model.save(export_dir)

    print("\nKeras model registered successfully.")

if __name__ == "__main__":
    main()
