import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2.csv")    
df.head()

df.shape
df.columns
df.dtypes
df.isna().sum()
df = df.drop(columns=["station_antwerp"])

df["datetime"] = pd.to_datetime(df["datetime"])

df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month


# lag features (previous pollution values)

df["london_lag1"] = df["station_london"].shift(1)
df["london_lag2"] = df["station_london"].shift(2)
df["london_lag3"] = df["station_london"].shift(3)


df["london_roll3"] = df["station_london"].rolling(3).mean()
df["london_roll6"] = df["station_london"].rolling(6).mean()

# remove rows created by shifting
df = df.dropna()

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

Y = df["station_london"]

# Train / Test split

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X,Y, test_size = 1/3, shuffle = False)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 500, random_state = 42)

rf.fit(X_train, Y_train)

rf_predictions = rf.predict(X_test)

from sklearn.metrics import mean_squared_error

rf_mse = mean_squared_error(Y_test, rf_predictions)

print("Random Forest MSE:", rf_mse)

import numpy as np

rmse = np.sqrt(rf_mse)

print(rmse)

from sklearn.metrics import r2_score

r2 = r2_score(Y_test, rf_predictions)
print("R2 score:", r2)

plt.figure(figsize=(10,5))

plt.plot(Y_test.values, label="Actual London NO2")
plt.plot(rf_predictions, label="Predicted London NO2")

plt.title("Actual vs Predicted London NO2")
plt.xlabel("Time")
plt.ylabel("NO2 Level")
plt.legend()

plt.show()

errors = Y_test - rf_predictions

plt.figure(figsize=(8,5))

plt.hist(errors, bins=30)

plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.show()

importance = pd.Series(rf.feature_importances_, index=X.columns)

importance.sort_values().plot(kind="barh", figsize=(8,5))

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.show()

plt.figure(figsize=(12,6))

plt.plot(df["datetime"], df["station_london"], label="London")
plt.plot(df["datetime"], df["station_paris"], label="Paris")

plt.title("NO2 Pollution Trend Over Time")
plt.xlabel("Date")
plt.ylabel("NO2 Concentration")
plt.legend()

plt.show()

from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, Y_train)

xgb_predictions = xgb.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

xgb_rmse = np.sqrt(mean_squared_error(Y_test, xgb_predictions))

# XGBoost predictions
xgb_predictions = xgb.predict(X_test)

# Plot Actual vs Predicted for XGBoost
plt.figure(figsize=(10,5))
plt.plot(Y_test.values, label="Actual London NO2")
plt.plot(xgb_predictions, label="Predicted London NO2 (XGBoost)")

plt.title("Actual vs Predicted (XGBoost)")
plt.xlabel("Time")
plt.ylabel("NO2 Level")
plt.legend()
plt.savefig("xgboost_actual_vs_predicted.png")
plt.show()

# Get feature importance
xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns)

# Plot
xgb_importance.sort_values().plot(kind='barh', figsize=(8,5))

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

# Save image
plt.savefig("xgboost_feature_importance.png")

plt.show()

xgb_errors = Y_test - xgb_predictions

plt.figure(figsize=(8,5))

plt.hist(xgb_errors, bins=30)

plt.title("XGBoost Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.savefig("xgboost_error_distribution.png")

plt.show()


# 1. ACTUAL vs PREDICTED (COMPARISON)

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
xgb_r2 = r2_score(Y_test, xgb_predictions)

print("XGBoost RMSE:", xgb_rmse)
print("XGBoost R2:", xgb_r2)
