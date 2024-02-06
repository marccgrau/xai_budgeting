import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

import xgboost as xgb
from src.feature_engineering import (
    apply_feature_engineering,
    choose_acc_ids,
    drop_all_zero_entries,
)
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Load the best hyperparameters
hyperparams_path = Path("hyperparameters/hyperparams_xgboost.json")
with open(hyperparams_path, "r") as file:
    best_hyperparams = json.load(file)

# Load the dataset
data_path = Path("data/final/merged_double_digit.csv")
df = pd.read_csv(data_path)

# Convert 'Year' to a relative year
df["Year"] = df["Year"] - df["Year"].min()

# Sort dataframe and apply feature engineering
df = df.sort_values(by="Year")
df = apply_feature_engineering(df)
df = drop_all_zero_entries(df)
df = choose_acc_ids(df=df, acc_ids=[40, 41, 42, 43, 45, 46, 47, 49])

# Load the label encoders
region_encoder_path = Path("models/le_region.joblib")
acc_id_encoder_path = Path("models/le_acc_id.joblib")
le_region = joblib.load(region_encoder_path)
le_acc_id = joblib.load(acc_id_encoder_path)

# Apply the label encoders to the dataframe
df["Region"] = le_region.transform(df["Region"])
df["Acc-ID"] = le_acc_id.transform(df["Acc-ID"])

# Define the cutoff year
cutoff_year = df["Year"].max() - 1

train_data = df[df["Year"] <= cutoff_year]
test_data = df[df["Year"] > cutoff_year]

X_train = train_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
y_train = train_data["Realized"]

X_test = test_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
y_test = test_data["Realized"]
budget_y_test = test_data["Budget y"]

# Load the model
model_save_path = Path("models/best_model_xgboost.json")
model = xgb.XGBRegressor(**best_hyperparams)
model.load_model(model_save_path)

# Predict on X_test
y_pred = model.predict(X_test)

# Evaluate the model
mae = round(mean_absolute_error(y_test, y_pred), 2)
mse = round(mean_squared_error(y_test, y_pred), 2)
rmse = round(np.sqrt(mse), 2)
mape = round(mean_absolute_percentage_error(y_test, y_pred), 2)

# Compare predictions with budget_y_test
comparison_mae = round(mean_absolute_error(y_test, budget_y_test), 2)
comparison_mse = round(mean_squared_error(y_test, budget_y_test), 2)
comparison_rmse = round(np.sqrt(comparison_mse), 2)
comparison_mape = round(mean_absolute_percentage_error(y_test, budget_y_test), 2)

# Log and save the results
logger.info(f"Model Evaluation: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}")
logger.info(
    f"Budget Comparison: MAE: {comparison_mae}, MSE: {comparison_mse}, RMSE: {comparison_rmse}, MAPE: {comparison_mape}"
)

results_file_path = Path("evaluations/evaluation_xgboost.txt")
with open(results_file_path, "w") as file:
    file.write(f"Model saved successfully at {model_save_path}\n")
    file.write(
        f"Model Evaluation: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}\n"
    )
    file.write(
        f"Budget Comparison: MAE: {comparison_mae}, MSE: {comparison_mse}, \
            RMSE: {comparison_rmse}, MAPE: {comparison_mape}\n"
    )

logger.info(f"Evaluation results saved successfully at {results_file_path}")
