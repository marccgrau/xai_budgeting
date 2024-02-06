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

import catboost as cb
import xgboost as xgb
from src.feature_engineering import (
    apply_feature_engineering,
    choose_acc_ids,
    drop_all_zero_entries,
)
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Load the best hyperparameters for both models
xgb_hyperparams_path = Path("hyperparameters/hyperparams_xgboost.json")
cat_hyperparams_path = Path("hyperparameters/hyperparams_catboost.json")
with open(xgb_hyperparams_path, "r") as file:
    xgb_best_hyperparams = json.load(file)
with open(cat_hyperparams_path, "r") as file:
    cat_best_hyperparams = json.load(file)

# Load the dataset
data_path = Path("data/final/merged_double_digit.csv")
df = pd.read_csv(data_path)

# Convert 'Year' to a relative year and apply feature engineering
df["Year"] = df["Year"] - df["Year"].min()
df = df.sort_values(by="Year")
df = apply_feature_engineering(df)
df = drop_all_zero_entries(df)
df = choose_acc_ids(df=df, acc_ids=[40, 41, 42, 43, 45, 46, 47, 49])

df_cb = df.copy()
df_xgb = df.copy()

# Load the label encoders
region_encoder_path = Path("models/le_region.joblib")
acc_id_encoder_path = Path("models/le_acc_id.joblib")
le_region = joblib.load(region_encoder_path)
le_acc_id = joblib.load(acc_id_encoder_path)

# Apply the label encoders to the dataframe
df_xgb["Region"] = le_region.transform(df_xgb["Region"])
df_xgb["Acc-ID"] = le_acc_id.transform(df_xgb["Acc-ID"])

# Define the cutoff year and split the data for XGBoost
cutoff_year_xgb = df_xgb["Year"].max() - 1
train_data_xgb = df_xgb[df_xgb["Year"] <= cutoff_year_xgb]
test_data_xgb = df_xgb[df_xgb["Year"] > cutoff_year_xgb]

# Define the cutoff year and split the data for CatBoost
cutoff_year_cb = df_cb["Year"].max() - 1
train_data_cb = df_cb[df_cb["Year"] <= cutoff_year_cb]
test_data_cb = df_cb[df_cb["Year"] > cutoff_year_cb]

X_test_xgb = test_data_xgb.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
y_test_xgb = test_data_xgb["Realized"]
budget_y_test = test_data_xgb["Budget y"]
y_test = test_data_xgb["Realized"]

X_test_cb = test_data_cb.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
y_test_cb = test_data_cb["Realized"]


# Load and predict with the XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model_path = Path("models/best_model_xgboost.json")
xgb_model.load_model(xgb_model_path)
xgb_preds = xgb_model.predict(X_test_xgb)

# Load and predict with the CatBoost model
cat_model = cb.CatBoostRegressor()
cat_model_path = Path("models/best_model_catboost.cbm")
cat_model.load_model(cat_model_path)
cat_preds = cat_model.predict(X_test_cb)

# Create a simple average ensemble of the predictions
ensemble_preds = (xgb_preds + cat_preds) / 2

# Evaluate the ensemble
ensemble_mae = mean_absolute_error(y_test, ensemble_preds)
ensemble_mse = mean_squared_error(y_test, ensemble_preds)
ensemble_rmse = np.sqrt(ensemble_mse)
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_preds)

# Compare ensemble predictions with budget_y_test
comparison_mae = mean_absolute_error(y_test, budget_y_test)
comparison_mse = mean_squared_error(y_test, budget_y_test)
comparison_rmse = np.sqrt(comparison_mse)
comparison_mape = mean_absolute_percentage_error(y_test, budget_y_test)

# Log and save the results
logger.info(
    f"Ensemble Evaluation: MAE: {ensemble_mae}, MSE: {ensemble_mse}, \
        RMSE: {ensemble_rmse}, MAPE: {ensemble_mape}"
)
logger.info(
    f"Budget Comparison: MAE: {comparison_mae}, MSE: {comparison_mse}, \
        RMSE: {comparison_rmse}, MAPE: {comparison_mape}"
)

# Save the evaluation results
results_file_path = Path("evaluations/evaluation_ensemble.txt")
with open(results_file_path, "w") as file:
    file.write(
        f"Model Evaluation: MAE: {ensemble_mae}, MSE: {ensemble_mse},  \
            RMSE: {ensemble_rmse}, MAPE: {ensemble_mape}\n"
    )
    file.write(
        f"Budget Comparison: MAE: {comparison_mae}, MSE: {comparison_mse}, \
            RMSE: {comparison_rmse}, MAPE: {comparison_mape}\n"
    )

logger.info(f"Evaluation results saved successfully at {results_file_path}")
