import os
import numpy as np
import hopsworks
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow import keras
from tensorflow.keras import layers

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

def main():
    # Login and get Feature Store
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Load Feature View
    fv = fs.get_feature_view(name="air_quality_fv", version=1)

    # Get training data
    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_size=0.2,
        seed=42,
    )

    # Drop non-numeric / non-useful columns
    X_train = X_train.drop(columns=["city_name", "date"], errors="ignore")
    X_test = X_test.drop(columns=["city_name", "date"], errors="ignore")

    # Ensure float32
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    feature_names = list(X_train.columns)
    print("Features used:", feature_names)

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
