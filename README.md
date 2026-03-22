# 🌫️ London NO2 Air Pollution Forecasting

A machine learning project that predicts London's NO2 (nitrogen dioxide) air pollution levels using **Random Forest** and **XGBoost** regressors, with time-series feature engineering.



##  Project Overview

This project uses hourly NO2 measurements from London and Paris monitoring stations to forecast London's pollution levels. It compares two ensemble models to determine which better captures air quality patterns.

**Target variable:** `station_london` (NO2 concentration, µg/m³)



## 📁 Project Structure


├── no2_forecasting.ipynb          # Main Jupyter notebook
├── README.md                      # Project documentation
├── plots/
│   ├── Random_Forest_actual_vs_predicted.png
│   ├── Random_Forest_Error_Distribution.png
│   ├── Random_Forest_Feature_importance.png
│   ├── xgboost_actual_vs_predicted.png
│   ├── xgboost_feature_importance.png
│   ├── xgboost_error_distribution.png
│   ├── comparison_actual_vs_predicted.png
│   └── pollution_trend.png
```



##  Dataset

- **Source:** [pandas sample data – air_quality_no2.csv](https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2.csv)
- **Coverage:** May – June 2019 (hourly measurements)
- **Stations:** London, Paris (Antwerp was dropped due to missing values)



##  Feature Engineering

Raw datetime was expanded into predictive features:

| Feature | Description |
|---|---|
| `hour` | Hour of the day (0–23) |
| `day` | Day of the month |
| `month` | Month of the year |
| `london_lag1` | London NO2 value 1 hour ago |
| `london_lag2` | London NO2 value 2 hours ago |
| `london_lag3` | London NO2 value 3 hours ago |
| `london_roll3` | 3-hour rolling mean of London NO2 |
| `london_roll6` | 6-hour rolling mean of London NO2 |
| `station_paris` | Concurrent Paris NO2 reading |

> `shuffle=False` was used in the train/test split to preserve temporal order.



##  Models

### Random Forest
```python
RandomForestRegressor(n_estimators=500, random_state=42)
```

### XGBoost
```python
XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


##  Results

| Model | RMSE | R² Score |
|---|---|---|
| Random Forest | 2.40 | 0.9482 |
| **XGBoost** | **2.35** | **0.9505** |

Both models achieved excellent performance (~95% variance explained). XGBoost edged ahead with a slightly lower RMSE and higher R².

**Key finding:** The 3-hour rolling mean (`london_roll3`) was the most important feature for both models, confirming that recent pollution history is the strongest predictor of current NO2 levels.


## Visualizations & Plot Explanations

### 1. Actual vs Predicted London NO2 (Random Forest & XGBoost — separate)

Both plots show the blue line as ground truth NO2 and the orange line as the model's prediction across ~280 test samples. Where the lines overlap closely, the model performs well. Both models track cyclical rises and falls accurately, but occasionally diverge at sharp spikes (around index 150) — sudden pollution events that are inherently hard to predict. The tighter overlap in the XGBoost plot is consistent with its lower RMSE.


### 2. Prediction Error Distribution (Random Forest & XGBoost)

These histograms show `actual - predicted` for every test sample. A perfect model would stack all errors at zero. Both distributions are bell-shaped and centered near zero — a good sign. The RF distribution spans roughly -8 to +8, while XGBoost's is slightly tighter and more centered. The slight left skew in both plots means the models occasionally over-predict the NO2 level. Overall, errors are small relative to the NO2 range.


### 3. Feature Importance — Random Forest

This horizontal bar chart ranks features by how much they reduce impurity across all 500 decision trees. `london_roll3` dominates massively, accounting for the vast majority of predictive power. `london_lag1` and `london_lag2` follow at a distance. Temporal features (`month`, `day`, `hour`) and `station_paris` contribute almost nothing. This confirms the model relies almost entirely on very recent pollution history rather than time-of-day or cross-city patterns.


### 4. Feature Importance — XGBoost

XGBoost uses a gain-based importance metric, so the ranking shifts slightly from Random Forest. `london_roll3` remains dominant, but `london_lag1` is notably more important here, and `london_roll6` also contributes more. This suggests XGBoost learns slightly richer temporal patterns — part of why it edges ahead on RMSE. Both models agree that `station_paris`, `hour`, `day`, and `month` are weak predictors.


### 5. NO2 Pollution Trend Over Time (London & Paris)

This plot shows the raw data across the full dataset (May–June 2019) — not predictions. Paris (orange) consistently spikes higher and more erratically than London (blue). Both cities show rhythmic patterns likely tied to traffic cycles (weekday peaks, weekend dips). This plot justifies including `station_paris` as a feature, though the importance charts reveal it ended up being weak — the two cities correlate loosely but not tightly enough to be a strong standalone predictor.


### 6. Actual vs Predicted Comparison (RF vs XGBoost — combined)

The most informative comparison plot. The black line is actual NO2, the dashed blue is Random Forest, and the dotted orange is XGBoost. For most of the test period the two models are nearly indistinguishable. Differences become visible toward the right side (later test samples), where XGBoost tracks the actual line slightly more closely during low-value stretches. This visually confirms that XGBoost's numerical advantage is real but modest — both are strong models.


##  Possible Improvements

- Add cross-validation for more robust evaluation
- Try LSTM or Prophet for deeper time-series modeling
- Include weather data (temperature, wind) as external features
- Tune hyperparameters with GridSearchCV or Optuna
