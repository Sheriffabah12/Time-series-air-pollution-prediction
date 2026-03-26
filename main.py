# London NO2 Air Pollution Forecasting
# Models: Random Forest vs XGBoost
# Target: Predict hourly NO2 levels at London monitoring station


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load the air quality dataset directly from the pandas GitHub repo.
# This dataset contains hourly NO2 readings from three European cities:
# London, Paris, and Antwerp — recorded between May and June 2019.

df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2.csv")    

# Quick look at the first few rows to understand structure
df.head()

# Check dimensions: how many rows (hours) and columns (stations + datetime)
df.shape

# See all column names
df.columns

# Check data types — datetime should be object here, we'll convert it below
df.dtypes

# Count missing values per column
# Antwerp has many NaN values, making it unreliable for modeling
df.isna().sum()

# Drop the Antwerp column — too many missing values to be useful as a feature
df = df.drop(columns=["station_antwerp"])

# Convert the datetime column from string to proper datetime format
# This enables us to extract time-based features (hour, day, month)

df["datetime"] = pd.to_datetime(df["datetime"])

# Extract time components as separate numeric features.
# These capture cyclical patterns in pollution:
# - hour: rush hour spikes (morning/evening traffic)
# - day: within-month variation
# - month: seasonal differences between May and June
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month

# Lag features capture the "memory" of the time series.
# london_lag1 = what was the NO2 level 1 hour ago?
# This is the single most informative type of feature for short-term forecasting
# because pollution levels don't change drastically from one hour to the next.

df["london_lag1"] = df["station_london"].shift(1)
df["london_lag2"] = df["station_london"].shift(2)
df["london_lag3"] = df["station_london"].shift(3)

# Rolling means smooth out noise and capture short-term trends.
# london_roll3 = average of the last 3 hours → captures very recent trend
# london_roll6 = average of the last 6 hours → captures slightly longer trend

df["london_roll3"] = df["station_london"].rolling(3).mean()
df["london_roll6"] = df["station_london"].rolling(6).mean()

# Shifting and rolling create NaN values at the start of the series
# (e.g., lag1 has no value for the first row). Drop those rows.

df = df.dropna()

# Features used for prediction:
# - station_paris: cross-city signal (Paris NO2 as a proxy for regional air quality)
# - hour, day, month: temporal patterns
# - lag and rolling features: recent London pollution history
X = df[[
"station_paris",
"hour",
"day",
"month",
"london_lag1",
"london_lag2",
"london_lag3",
"london_roll3",
"london_roll6"
]]

# Target: London NO2 concentration (what we want to predict)
Y = df["station_london"]


# Split data chronologically 
# IMPORTANT: shuffle=False preserves the time order.
# Shuffling would let the model "see the future" during training,
# which would produce misleadingly high accuracy scores.

X_train, X_test, Y_train, Y_test = train_test_split( X,Y, test_size = 1/3, shuffle = False)

# Random Forest builds 500 independent decision trees and averages their predictions.
# More trees = more stable predictions, but slower training.
# random_state=42 ensures reproducible results.

rf = RandomForestRegressor(n_estimators = 500, random_state = 42)

rf.fit(X_train, Y_train)

# Generate predictions on the held-out test set

rf_predictions = rf.predict(X_test)

# MSE: average squared difference between actual and predicted values.
# Lower is better. Squaring penalises large errors more than small ones.

rf_mse = mean_squared_error(Y_test, rf_predictions)
print("Random Forest MSE:", rf_mse)

# RMSE: square root of MSE — brings the error back to the original unit (µg/m³).
# An RMSE of 2.40 means predictions are off by ~2.4 µg/m³ on average.
rmse = np.sqrt(rf_mse)
print(rmse)

# R² (coefficient of determination): proportion of variance explained by the model.
# 0.948 means the model explains ~94.8% of the variation in London NO2 levels.

r2 = r2_score(Y_test, rf_predictions)
print("R2 score:", r2)


# PLOT 1: Actual vs Predicted — Random Forest
# Blue line = actual NO2 values from the test set.
# Orange line = model's predicted values.
# The x-axis is the sequential test sample index (not real datetime).
# Where the lines overlap closely, the model is performing well.
# Divergences at peaks indicate the model struggles with sudden pollution spikes,
# which are difficult to predict from historical patterns alone.
plt.figure(figsize=(10,5))

plt.plot(Y_test.values, label="Actual London NO2")
plt.plot(rf_predictions, label="Predicted London NO2")

plt.title("Actual vs Predicted London NO2")
plt.xlabel("Time")
plt.ylabel("NO2 Level")
plt.legend()
plt.savefig("Random_Forest_actual_vs_predicted.png")
plt.show()

# PLOT 2: Prediction Error Distribution — Random Forest
# Error = actual - predicted for each test sample.
# Positive error → model under-predicted (actual was higher than forecast).
# Negative error → model over-predicted (actual was lower than forecast).
# A good model produces errors centered at 0, roughly bell-shaped.
# The slight left skew here means the model tends to over-predict slightly.
# Most errors fall between -4 and +4, with few large outliers.

errors = Y_test - rf_predictions

plt.figure(figsize=(8,5))

plt.hist(errors, bins=30)

plt.title("Prediction Error Distribution")
plt.xlabel("Error_distribution.png")
plt.savefig("Random_Forest_Error_Distribution.png")
plt.show()

# PLOT 3: Feature Importance — Random Forest
# Feature importance in Random Forest = how much each feature reduces
# impurity (variance) across all 500 trees, averaged and normalised to sum to 1.
# london_roll3 dominates — the 3-hour rolling mean captures recent pollution
# trend so well that the model barely needs anything else.
# lag1 and lag2 follow distantly. Time features (hour, day, month) and
# station_paris contribute almost nothing, suggesting London's NO2 is
# largely self-contained in the short term.

importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", figsize=(8,5))

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.savefig("Random_Forest_Feature_importance.png")
plt.show()

# PLOT 4: NO2 Pollution Trend Over Time (London & Paris)
# This plot shows the raw, unmodified NO2 readings across the full dataset
# (May–June 2019) — not predictions. It gives context for what the models
# are trying to learn.
# Paris (orange) shows higher and more erratic spikes than London (blue).
# Both cities share rhythmic patterns likely tied to traffic cycles
# (weekday rush hours, quieter weekends).
# This justifies including station_paris as a feature, though the importance
# charts show it ended up being a weak predictor — the cities correlate
# loosely but not tightly enough for Paris readings to reliably forecast London.
plt.figure(figsize=(12,6))

plt.plot(df["datetime"], df["station_london"], label="London")
plt.plot(df["datetime"], df["station_paris"], label="Paris")

plt.title("NO2 Pollution Trend Over Time")
plt.xlabel("Date")
plt.ylabel("NO2 Concentration")
plt.legend()
plt.savefig("pollution_trend.png")

plt.show()

# SECTION 7: MODEL 2 — XGBOOST

# XGBoost builds trees sequentially — each new tree corrects the errors
# of the previous one (gradient boosting). Key hyperparameters:
# - n_estimators=2000: up to 2000 boosting rounds (more than RF's 500 trees)
# - learning_rate=0.01: small steps per round → slower but more accurate
# - max_depth=6: each tree can split up to 6 levels deep
# - subsample=0.8: each tree sees 80% of training rows (reduces overfitting)
# - colsample_bytree=0.8: each tree sees 80% of features (reduces overfitting)

xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, Y_train)
xgb_predictions = xgb.predict(X_test)

# ---- Evaluation ----
# XGBoost RMSE: 2.35 — slightly lower than RF's 2.40
# XGBoost R²: 0.9505 — slightly higher than RF's 0.9482
# Both differences are small but consistent, confirming XGBoost as the better model.

xgb_rmse = np.sqrt(mean_squared_error(Y_test, xgb_predictions))
xgb_r2 = r2_score(Y_test, xgb_predictions)

print("XGBoost RMSE:", xgb_rmse)
print("XGBoost R2:", xgb_r2)

# PLOT 5: Actual vs Predicted — XGBoost

# Same structure as the Random Forest plot.
# The orange (predicted) line tracks the blue (actual) line very closely.
# XGBoost handles the mid-range values slightly better than RF,
# particularly in low-pollution periods toward the end of the test set.
# Sharp spikes remain the hardest to predict for both models.

plt.figure(figsize=(10,5))
plt.plot(Y_test.values, label="Actual London NO2")
plt.plot(xgb_predictions, label="Predicted London NO2 (XGBoost)")

plt.title("Actual vs Predicted (XGBoost)")
plt.xlabel("Time")
plt.ylabel("NO2 Level")
plt.legend()
plt.savefig("xgboost_actual_vs_predicted.png")
plt.show()

# Feature Importance — XGBoost

# XGBoost importance = gain-based (how much each feature improves predictions
# when used in a split), unlike RF which uses impurity reduction.
# london_roll3 still dominates, but london_lag1 is notably more prominent
# here than in the RF chart — XGBoost is extracting more value from the
# immediate 1-hour lag. london_roll6 also contributes more than in RF.
# This richer use of temporal features partly explains XGBoost's edge.
# station_paris, hour, day, and month remain weak in both models.
xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns)

xgb_importance.sort_values().plot(kind='barh', figsize=(8,5))

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.savefig("xgboost_feature_importance.png")
plt.show()

# Prediction Error Distribution — XGBoost

# Same interpretation as the Random Forest error plot.
# XGBoost's distribution is slightly tighter and more centered at zero,
# reflecting its marginally lower RMSE.
# The slight left skew is still present — both models share the tendency
# to over-predict slightly, likely because the training data has more
# mid-to-high pollution samples than low ones.
xgb_errors = Y_test - xgb_predictions

plt.figure(figsize=(8,5))

plt.hist(xgb_errors, bins=30)

plt.title("XGBoost Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.savefig("xgboost_error_distribution.png")
plt.show()


# 1. ACTUAL vs PREDICTED (COMPARISON)

# This is the most informative comparison plot.
# Black = actual NO2 values (ground truth)
# Blue dashed = Random Forest predictions
# Orange dotted = XGBoost predictions
#
# For most of the test period the two models are nearly indistinguishable —
# both track the actual line very well. Differences become visible toward
# the right side (later test samples), where XGBoost tracks low-pollution
# stretches slightly more accurately.
# This confirms XGBoost's numerical advantage is real but modest —
# both models are strong for this forecasting task
plt.figure(figsize=(12,6))

plt.plot(Y_test.values, label="Actual", color="black")
plt.plot(rf_predictions, label="Random Forest", linestyle="--")
plt.plot(xgb_predictions, label="XGBoost", linestyle=":")

plt.title("Actual vs Predicted Comparison (RF vs XGBoost)")
plt.xlabel("Time")
plt.ylabel("NO2 Level")
plt.legend()

plt.savefig("comparison_actual_vs_predicted.png")
plt.show()


# PLOT 9: Error Distribution Comparison (RF vs XGBoost)

# This plot overlays both models' error histograms on the same axes
# using alpha=0.5 transparency so both distributions are visible simultaneously.
# Blue (semi-transparent) = Random Forest errors
# Orange (semi-transparent) = XGBoost errors
#
# Key things to read from this plot:
# - The XGBoost distribution (orange) has a taller, narrower peak near zero,
#   meaning more of its predictions land very close to the actual value.
# - The Random Forest distribution (blue) is slightly wider and flatter,
#   indicating more spread in its errors — consistent with its higher RMSE.
# - Both distributions share the same left skew (negative errors are more
#   spread out than positive ones), meaning both models occasionally
#   over-predict NO2 levels but neither under-predicts consistently.
# - The large overlap between the two confirms that the performance
#   difference between the models is real but small.
# This overlaid view is more honest than two side-by-side histograms
# because it forces a direct visual alignment on the same scale.

rf_errors = Y_test - rf_predictions
xgb_errors = Y_test - xgb_predictions

plt.figure(figsize=(10,5))

plt.hist(rf_errors, bins=30, alpha=0.5, label="Random Forest")
plt.hist(xgb_errors, bins=30, alpha=0.5, label="XGBoost")

plt.title("Error Distribution Comparison")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.legend()

plt.savefig("comparison_error_distribution.png")
plt.show()



# PLOT 10: Feature Importance Comparison (RF vs XGBoost)

# This plot overlays both models' feature importances on the same bar chart
# using alpha=0.6 transparency so both sets of bars are readable together.
# Light blue = Random Forest importance (impurity/variance reduction based)
# Light orange = XGBoost importance (gain based — improvement per split)
#
# Key things to read from this plot:
# - london_roll3 dominates in BOTH models, but XGBoost's bar is longer —
#   it assigns even more weight to the 3-hour rolling mean, suggesting it
#   exploits this feature more aggressively than Random Forest does.
# - london_lag1 is the biggest difference between the two models: the
#   XGBoost bar is noticeably longer while the RF bar is barely visible.
#   This means XGBoost recognises the immediate previous hour as a strong
#   independent signal, while RF absorbs it into the rolling mean instead.
# - london_roll6 follows the same pattern — XGBoost values it more than RF,
#   showing it spreads importance across multiple temporal features rather
#   than concentrating almost everything in a single one.
# - For weaker features (station_paris, hour, day, month, london_lag3),
#   both models agree — their bars are similarly negligible.
# - Note: the two importance scales are not directly comparable (impurity
#   vs gain), but the relative ranking within each model is still valid.
# Overall this chart explains WHY XGBoost performs slightly better:
# it distributes importance more evenly across several temporal features,
# making it more robust than RF's near-total reliance on london_roll3 alone

# Random Forest importance
rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

# XGBoost importance
xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns)

plt.figure(figsize=(10,6))

rf_importance.sort_values().plot(kind='barh', alpha=0.6, label="Random Forest")
xgb_importance.sort_values().plot(kind='barh', alpha=0.6, label="XGBoost")

plt.title("Feature Importance Comparison")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.legend()

plt.savefig("comparison_feature_importance.png")
plt.show()

# SUMMARY

# | Model         | RMSE  | R²     |
# |---------------|-------|--------|
# | Random Forest | 2.40  | 0.9482 |
# | XGBoost       | 2.35  | 0.9505 |
#
# Both models explain ~95% of variance in London NO2 levels.
# XGBoost wins by a small margin on both metrics.
# The dominant feature for both models is london_roll3 —
# the 3-hour rolling mean — confirming that very recent pollution
# history is the strongest signal for short-term NO2 forecasting.
