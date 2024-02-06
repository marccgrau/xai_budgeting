import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

import catboost as cb
from src.feature_engineering import (
    apply_feature_engineering,
    choose_acc_ids,
    drop_all_zero_entries,
)
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Load the best hyperparameters
hyperparams_path = Path("hyperparameters/hyperparams_catboost.json")
with open(hyperparams_path, "r") as file:
    best_hyperparams = json.load(file)

# Load the dataset
data_path = Path("data/final/merged_double_digit.csv")
df = pd.read_csv(data_path)

# Convert 'Year' to a relative year
df["Year"] = df["Year"] - df["Year"].min()

# Sort dataframe and apply feature engineerings
df = df.sort_values(by="Year")
df = apply_feature_engineering(df)
df = drop_all_zero_entries(df)
df = choose_acc_ids(df=df, acc_ids=[40, 41, 42, 43, 45, 46, 47, 49])

# Define the cutoff year
cutoff_year = df["Year"].max() - 1
logger.info(f"Cutoff year: {cutoff_year}")

train_data = df[df["Year"] <= cutoff_year]
test_data = df[df["Year"] > cutoff_year]

logger.info(f"Train data shape: {train_data.shape}")
logger.info(f"Test data shape: {test_data.shape}")
logger.info(f"Train years: {train_data['Year'].unique()}")
logger.info(f"Test years: {test_data['Year'].unique()}")
logger.info(f"Train columns: {train_data.columns}")
logger.info(f"Test columns: {test_data.columns}")
logger.info(f"Train data head: {train_data.head()}")
logger.info(f"Test data head: {test_data.head()}")

X_train = train_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
y_train = train_data["Realized"]
budget_y_train = train_data["Budget y"]

X_test = test_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
y_test = test_data["Realized"]
budget_y_test = test_data["Budget y"]

# Load the model
model_save_path = Path("models/best_model_catboost.cbm")
model = cb.CatBoostRegressor()
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

# Log the results
logger.info(f"Model Evaluation: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}")
logger.info(
    f"Budget Comparison: MAE: {comparison_mae}, MSE: {comparison_mse}, \
        RMSE: {comparison_rmse}, MAPE: {comparison_mape}"
)

results_file_path = Path("evaluations/evaluation_catboost.txt")

# Write the evaluation results to the file
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
