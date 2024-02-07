import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def evaluate_model(y_test, y_pred, budget_y_test):
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    rmse = round(np.sqrt(mse), 2)
    mape = round(mean_absolute_percentage_error(y_test, y_pred), 2)

    comparison_mae = round(mean_absolute_error(y_test, budget_y_test), 2)
    comparison_mse = round(mean_squared_error(y_test, budget_y_test), 2)
    comparison_rmse = round(np.sqrt(comparison_mse), 2)
    comparison_mape = round(mean_absolute_percentage_error(y_test, budget_y_test), 2)

    return (
        mae,
        mse,
        rmse,
        mape,
        comparison_mae,
        comparison_mse,
        comparison_rmse,
        comparison_mape,
    )


def log_and_save_evaluation_results(results_file_path: Path, **kwargs):
    with open(results_file_path, "w") as file:
        for key, value in kwargs.items():
            logger.info(f"{key}: {value}")
            file.write(f"{key}: {value}\n")


def main(
    file_path: Path = Path("data/final/merged_double_digit.csv"), category: str = "Alle"
):
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    df = engineer_df(df, acc_config.get(category))

    preprocessor = joblib.load(Path("models/preprocessor_hgb.joblib"))

    cutoff_year = df["Year"].max() - 1
    test_data = df[df["Year"] > cutoff_year]
    X_test = test_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
    y_test = test_data["Realized"]

    X_test = preprocessor.transform(X_test)

    model = joblib.load(Path("models/best_model_hgb.joblib"))
    y_pred = model.predict(X_test)

    evaluation_results = evaluate_model(y_test, y_pred, test_data["Budget y"])

    log_and_save_evaluation_results(
        Path("evaluations/evaluation_hgb.txt"),
        **{
            "Model Evaluation - MAE": evaluation_results[0],
            "Model Evaluation - MSE": evaluation_results[1],
            "Model Evaluation - RMSE": evaluation_results[2],
            "Model Evaluation - MAPE": evaluation_results[3],
            "Budget Comparison - MAE": evaluation_results[4],
            "Budget Comparison - MSE": evaluation_results[5],
            "Budget Comparison - RMSE": evaluation_results[6],
            "Budget Comparison - MAPE": evaluation_results[7],
            "Model saved successfully at": str(Path("models/best_model_hgb.joblib")),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the HistGradientBoostingRegressor model."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/final/merged_double_digit.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--category", type=str, default="Alle", help="Category to evaluate"
    )
    args = parser.parse_args()
    main(file_path=args.file_path, category=args.category)
