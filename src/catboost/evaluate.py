import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

import catboost as cb
from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def evaluate_model(
    y_test: np.ndarray, y_pred: np.ndarray, budget_y_test: np.ndarray
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Evaluate the model performance and compare it against budget values."""
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


def log_and_save_evaluation_results(
    results_file_path: Path, kwargs: Dict[str, object]
) -> None:
    """Log and save the evaluation results to a file."""
    with open(results_file_path, "w", encoding="utf-8") as file:
        for key, value in kwargs.items():
            logger.info(f"{key}: {value}")
            file.write(f"{key}: {value}\n")


def main(
    file_path: Path = Path("data/final/merged_double_digit.csv"), category: str = "Alle"
) -> None:

    hyperparams_path: Path = Path("hyperparameters/hyperparams_catboost.json")
    model_save_path: Path = Path("models/best_model_catboost.cbm")
    results_file_path: Path = Path("evaluations/evaluation_catboost.txt")

    with open(hyperparams_path, "r") as file:
        best_hyperparams: Dict[str, Any] = json.load(file)

    df: pd.DataFrame = pd.read_csv(file_path)

    df = engineer_df(df, acc_config.get(category))

    cutoff_year: int = df["Year"].max() - 1
    test_data: pd.DataFrame = df[df["Year"] > cutoff_year]
    X_test: pd.DataFrame = test_data.drop(
        columns=["Realized", "Budget y", "Budget y+1", "Slack"]
    )
    y_test: np.ndarray = test_data["Realized"].values
    budget_y_test: np.ndarray = test_data["Budget y"].values

    model: cb.CatBoostRegressor = cb.CatBoostRegressor(**best_hyperparams)
    model.load_model(model_save_path)
    y_pred: np.ndarray = model.predict(X_test)

    evaluation_results: Tuple[
        float, float, float, float, float, float, float, float
    ] = evaluate_model(y_test, y_pred, budget_y_test)

    log_and_save_evaluation_results(
        results_file_path,
        {
            "Model Evaluation - MAE": evaluation_results[0],
            "Model Evaluation - MSE": evaluation_results[1],
            "Model Evaluation - RMSE": evaluation_results[2],
            "Model Evaluation - MAPE": evaluation_results[3],
            "Budget Comparison - MAE": evaluation_results[4],
            "Budget Comparison - MSE": evaluation_results[5],
            "Budget Comparison - RMSE": evaluation_results[6],
            "Budget Comparison - MAPE": evaluation_results[7],
            "Model saved successfully at": str(model_save_path),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the CatBoost model.")
    parser.add_argument(
        "--file_path",
        type=str,
        default="data/final/merged_double_digit.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Alle",
        help="Category of the dataset to use for training",
    )
    args = parser.parse_args()
    main(file_path=args.file_path, category=args.category)
