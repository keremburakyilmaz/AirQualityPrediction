# Air Quality Prediction System

A machine learning system for predicting PM2.5 air quality using historical weather and air quality data. The system supports two cities: **Stockholm** (single sensor) and **Delhi** (multi-sensor).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Pipelines](#pipelines)
- [Automated Workflows](#automated-workflows)
- [Model Information](#model-information)
- [Dashboard](#dashboard)

## Overview

This project implements a complete ML pipeline for air quality forecasting:

- **Data Collection**: Historical and real-time air quality and weather data
- **Feature Engineering**: Weather features + temporal features + lag features (1, 2, 3 days)
- **Model Training**: TensorFlow/Keras neural networks for PM2.5 prediction
- **Inference**: 7-day autoregressive forecasts
- **Monitoring**: Hindcast analysis comparing predictions vs actuals
- **Visualization**: Automated dashboard with daily forecast updates

## Features

### Completed Tasks

#### Grade 'E' Tasks
1. **Backfill Feature Pipeline**: Downloads historical weather data and air quality data, registers as Feature Groups in Hopsworks
2. **Daily Feature Pipeline**: Scheduled daily pipeline that downloads yesterday's data and 7-10 day weather forecasts, updates Feature Groups
3. **Training Pipeline**: Selects features, reads training data via Feature View, trains regression models, registers with Hopsworks
4. **Batch Inference Pipeline**: Downloads models from Hopsworks, generates 7-10 day forecasts, creates dashboard visualizations
5. **Hindcast Monitoring**: Plots hindcast graphs showing predictions vs actual measured air quality

#### Grade 'C' Tasks
6. **Multiple Lag Features**: Added 1-day, 2-day, and 3-day lagged PM2.5 features with performance analysis

#### Grade 'A' Tasks
7. **Multi-Sensor Support**: Provides predictions for all air quality sensors in Delhi (15 sensors)

**Noteworthy Limitation**: The city Delhi has disadvantage in prediction due to low air quality overall the year and limited amount of historical data (only 38 days)

## Project Structure

```
AirQualityPrediction/
├── data/                          # Stockholm historical data
│   └── air_quality_data.csv
├── delhi/                         # Delhi-specific pipelines
│   ├── data/                      # Delhi sensor CSV files (15 sensors)
│   │   └── delhi_sensor_*.csv
│   ├── backfill_feature_groups_delhi.py
│   ├── feature_daily_pipeline_delhi.py
│   ├── training_pipeline_per_sensor_delhi.py
│   ├── batch_inference_per_sensor_delhi.py
│   └── hindcast_delhi.py
├── docs/                          # Dashboard and visualizations
│   ├── index.html                 # Web dashboard
│   ├── pm25_forecast.png          # Stockholm forecast
│   └── pm25_forecast_delhi.png    # Delhi forecast
├── model/                         # Saved models (local)
├── .github/workflows/             # GitHub Actions workflows
│   ├── pm2_5_prediction.yml       # Stockholm daily pipeline
│   └── daily_pm2_5_delhi_multisensor.yml  # Delhi daily pipeline
├── backfill_feature_groups.py     # Stockholm backfill
├── feature_daily_pipeline.py      # Stockholm daily features
├── training_pipeline_tf.py        # Stockholm training
├── batch_inference_pipeline.py    # Stockholm inference
└── hindcast.py                    # Stockholm hindcast
```

## Pipelines

### Feature Pipelines

**Stockholm** (`feature_daily_pipeline.py`):
- Fetches latest PM2.5 from WAQI API (station A65290)
- Fetches weather forecast (yesterday + next 10 days)
- Updates Feature Groups in Hopsworks

**Delhi** (`delhi/feature_daily_pipeline_delhi.py`):
- Fetches latest PM2.5 for all 15 sensors from WAQI API
- Fetches weather forecast for Delhi
- Updates Feature Groups in Hopsworks

### Training Pipelines

**Stockholm** (`training_pipeline_tf.py`):
- Single model for Stockholm sensor
- Features: weather (4) + temporal (3) + lag (3) = 10 features
- Model architecture: Normalization -> Dense(64) -> Dense(32) -> Dense(1)

**Delhi** (`delhi/training_pipeline_per_sensor_delhi.py`):
- Separate model for each sensor (15 models)
- Same feature set and architecture
- Each model registered independently

### Inference Pipelines

**Stockholm** (`batch_inference_pipeline.py`):
- Uses fixed model version (v4)
- Autoregressive forecasting with lag feature updates
- Single forecast plot

**Delhi** (`delhi/batch_inference_per_sensor_delhi.py`):
- Automatically uses latest model version for each sensor
- Autoregressive forecasting for all sensors
- Multi-sensor forecast plot

## Automated Workflows

### GitHub Actions

Two workflows run daily at **6 AM UTC**:

1. **Stockholm Pipeline** (`.github/workflows/pm2_5_prediction.yml`)
   - Runs `feature_daily_pipeline.py`
   - Runs `batch_inference_pipeline.py`
   - Commits updated forecast PNG

2. **Delhi Pipeline** (`.github/workflows/daily_pm2_5_delhi_multisensor.yml`)
   - Runs `delhi/feature_daily_pipeline_delhi.py`
   - Runs `delhi/batch_inference_per_sensor_delhi.py`
   - Commits updated forecast PNG

Both workflows can also be triggered manually via `workflow_dispatch`.

## Model Information

### Features

All models use the following 10 features:

**Weather Features (4)**:
- `temp_max`: Maximum temperature
- `temp_min`: Minimum temperature
- `wind_speed_max`: Maximum wind speed
- `wind_direction_dominant`: Dominant wind direction

**Temporal Features (3)**:
- `day_of_week`: Day of week (0-6)
- `month`: Month (1-12)
- `day_of_year`: Day of year (1-366)

**Lag Features (3)**:
- `pm2_5_prev`: PM2.5 from 1 day ago
- `pm2_5_prev_2`: PM2.5 from 2 days ago
- `pm2_5_prev_3`: PM2.5 from 3 days ago

### Model Architecture

```
Input (10 features) -> Normalization Layer -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1)  # PM2.5
```

**Training**:
- Optimizer: Adam
- Loss: MSE
- Metrics: MAE, MSE
- Early stopping: patience=10 (Delhi) / 5 (Stockholm)
- Train/Test split: 80/20 (time-based)

### Model Registry

**Stockholm**:
- Model name: `pm25_keras_model`
- Current version: 4
- Fixed version in inference

**Delhi**:
- Model names: `pm25_model_delhi_{sensor_uid}`
- Auto-increments version on each training
- Inference automatically uses latest version

## Dashboard

The dashboard is available at `docs/index.html` and displays:
- **Stockholm Forecast**: 7-day PM2.5 forecast for Stockholm
- **Delhi Forecast**: 7-day PM2.5 forecasts for all 15 Delhi sensors

The dashboard automatically updates daily at 6 AM UTC via GitHub Actions workflows.

## Monitoring

### Hindcast Analysis

Both pipelines include hindcast scripts that:
- Compare historical predictions vs actual measured values
- Calculate RMSE and MAE metrics
- Generate visualization plots

**Stockholm**: `hindcast.py` -> `docs/pm25_hindcast_stockholm.png`
**Delhi**: `delhi/hindcast_delhi.py` -> `docs/pm25_hindcast_delhi.png`