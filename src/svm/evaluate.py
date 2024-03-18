import argparse
import logging
from pathlib import Path
from typing import Optional

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


def save_results_to_csv(
    file_path: Path, y_test, y_pred, budget_y, category, acc_id, region, **kwargs
):
    results_df = pd.DataFrame(
        {"y_test": y_test, "y_pred": y_pred, "Budget y": budget_y}
    )
    results_df.to_csv(file_path, index=False)
    log_and_save_evaluation_results(
        Path(f"evaluations/evaluation_hgb_{category}_{acc_id}_{region}.txt"), **kwargs
    )


def main(
    file_path: Path = Path("data/final/merged_double_digit.csv"),
    category: str = "Alle",
    acc_id: Optional[str] = None,
    region: Optional[str] = None,
):
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    df = engineer_df(df, acc_config.get(category))

    if acc_id is not None:
        df = df[df["Acc-ID"] == int(acc_id)]
    if region is not None:
        df = df[df["Region"] == region]

    preprocessor = joblib.load(Path(f"models/preprocessor_hgb_{category}.joblib"))

    cutoff_year = df["Year"].max() - 1
    test_data = df[df["Year"] > cutoff_year]
    X_test = test_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
    y_test = test_data["Realized"]

    X_test = preprocessor.transform(X_test)

    model = joblib.load(Path(f"models/best_model_hgb_{category}.joblib"))
    y_pred = model.predict(X_test)

    evaluation_results = evaluate_model(y_test, y_pred, test_data["Budget y"])

    save_results_to_csv(
        Path(f"evaluations/hbg_predictions_{acc_id}_{region}.csv"),
        y_test=y_test,
        y_pred=y_pred,
        budget_y=test_data["Budget y"],
        category=category,
        acc_id=acc_id,
        region=region,
        **{
            "Model Evaluation - MAE": evaluation_results[0],
            "Model Evaluation - MSE": evaluation_results[1],
            "Model Evaluation - RMSE": evaluation_results[2],
            "Model Evaluation - MAPE": evaluation_results[3],
            "Budget Comparison - MAE": evaluation_results[4],
            "Budget Comparison - MSE": evaluation_results[5],
            "Budget Comparison - RMSE": evaluation_results[6],
            "Budget Comparison - MAPE": evaluation_results[7],
            "Model saved successfully at": str(
                Path(f"models/best_model_hgb_{category}.joblib")
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the HistGradientBoostingRegressor model."
    )

    def none_or_str(value):
        if value == "None":
            return None
        return value

    parser.add_argument(
        "--file_path",
        type=str,
        default="data/final/merged_double_digit.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--category", type=str, default="Alle", help="Category to evaluate"
    )
    parser.add_argument(
        "--acc_id", type=none_or_str, default=None, help="Account ID to evaluate"
    )
    parser.add_argument(
        "--region", type=none_or_str, default=None, help="Region to evaluate"
    )
    args = parser.parse_args()
    main(
        file_path=args.file_path,
        category=args.category,
        acc_id=args.acc_id,
        region=args.region,
    )
