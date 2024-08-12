import argparse
import json
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


def load_label_encoders(region_encoder_path: Path, acc_id_encoder_path: Path):
    le_region = joblib.load(region_encoder_path)
    le_acc_id = joblib.load(acc_id_encoder_path)
    return le_region, le_acc_id


def encode_features(df: pd.DataFrame, le_region, le_acc_id) -> pd.DataFrame:
    df["Region"] = le_region.transform(df["Region"])
    df["Acc-ID"] = le_acc_id.transform(df["Acc-ID"])
    return df


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
        Path(f"evaluations/evaluation_rf_{category}_{acc_id}_{region}.txt"),
        **kwargs,
    )


def main(
    file_path: Path = Path("data/final/merged_complete.csv"),
    category: str = "Alle",
    acc_id: Optional[str] = None,
    region: Optional[str] = None,
):
    # Configuration and Paths
    hyperparams_path = Path(f"hyperparameters/hyperparams_rf_{category}.json")
    file_path = Path(file_path)
    model_save_path = Path(f"models/best_model_rf_{category}.joblib")
    column_transformer_path = Path(f"models/column_transformer_{category}.joblib")

    # Load hyperparameters and data
    with open(hyperparams_path, "r") as file:
        best_hyperparams = json.load(file)
    df = pd.read_csv(file_path)

    # Feature Engineering
    df = engineer_df(df, acc_config.get(category))

    categorical_columns = ["Region", "Acc-ID"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    if acc_id is not None:
        df = df[df["Acc-ID"] == int(acc_id)]
    if region is not None:
        df = df[df["Region"] == region]
    df = pd.get_dummies(df, columns=["Region", "Acc-ID"])

    # Split Data
    cutoff_year = df["Year"].max() - 1
    test_data = df[df["Year"] > cutoff_year]

    X_test, y_test = (
        test_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"]),
        test_data["Realized"],
    )
    budget_y_test = test_data["Budget y"]

    # Load Model and Predict
    model = joblib.load(model_save_path)
    y_pred = model.predict(X_test)

    # Evaluate
    evaluation_results = evaluate_model(y_test, y_pred, budget_y_test)

    # Log and Save Results
    save_results_to_csv(
        Path(f"evaluations/rf_predictions_{acc_id}_{region}.csv"),
        y_test=y_test,
        y_pred=y_pred,
        budget_y=budget_y_test,
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
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RandomForest model.")

    def none_or_str(value):
        if value == "None":
            return None
        return value

    parser.add_argument(
        "--file_path",
        type=str,
        default="data/final/merged_complete.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Alle",
        help="Category to evaluate",
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
