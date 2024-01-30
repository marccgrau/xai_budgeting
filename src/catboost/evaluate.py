import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import catboost as cb
from src.catboost.feature_engineering import apply_feature_engineering
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

df = apply_feature_engineering(df)

# Split the data into training and test sets
train_data, test_data = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=False
)

X_train = train_data.drop(columns=["Realized", "Budget y", "Budget y+1"])
y_train = train_data["Realized"]
budget_y_train = train_data["Budget y"]

X_test = test_data.drop(columns=["Realized", "Budget y", "Budget y+1"])
y_test = test_data["Realized"]
budget_y_test = test_data["Budget y"]

# Load the model
model_save_path = Path("models/best_model_catboost.cbm")
model = cb.CatBoostRegressor()
model.load_model(model_save_path)

# Predict on X_test
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Compare predictions with budget_y_test
comparison_mae = mean_absolute_error(y_test, budget_y_test)
comparison_mse = mean_squared_error(y_test, budget_y_test)
comparison_rmse = np.sqrt(comparison_mse)

# Log the results
logger.info(f"Model Evaluation: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
logger.info(
    f"Budget Comparison: MAE: {comparison_mae}, MSE: {comparison_mse}, RMSE: {comparison_rmse}"
)

results_file_path = Path("evaluations/evaluation_catboost.txt")

# Write the evaluation results to the file
with open(results_file_path, "w") as file:
    file.write(f"Model saved successfully at {model_save_path}\n")
    file.write(f"Model Evaluation: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}\n")
    file.write(
        f"Budget Comparison: MAE: {comparison_mae}, MSE: {comparison_mse}, RMSE: {comparison_rmse}\n"
    )

logger.info(f"Evaluation results saved successfully at {results_file_path}")
