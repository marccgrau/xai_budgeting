import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from prophet import Prophet
from prophet.serialize import model_from_json

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
    r2 = round(r2_score(y_test, y_pred), 2)

    comparison_mae = round(mean_absolute_error(y_test, budget_y_test), 2)
    comparison_mse = round(mean_squared_error(y_test, budget_y_test), 2)
    comparison_rmse = round(np.sqrt(comparison_mse), 2)
    comparison_mape = round(mean_absolute_percentage_error(y_test, budget_y_test), 2)
    comparison_r2 = round(r2_score(y_test, budget_y_test), 2)

    return (
        mae,
        mse,
        rmse,
        mape,
        r2,
        comparison_mae,
        comparison_mse,
        comparison_rmse,
        comparison_mape,
        comparison_r2,
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
        Path(f"evaluations/evaluation_prophet_{category}_{acc_id}_{region}_with_r2.txt"),
        **kwargs,
    )


def main(
    file_path: Path = Path("data/final/merged_complete_preprocessed.csv"),
    category: str = "Alle",
    acc_id: Optional[str] = None,
    region: Optional[str] = None,
):
    # Configuration and Paths
    model_save_path = Path(f"models/best_model_prophet_{category}.json")
    file_path = Path(file_path)

    # Load data
    df = pd.read_csv(file_path)
    df = engineer_df(df, acc_config.get(category))

    if acc_id is not None:
        df = df[df["Acc-ID"] == int(acc_id)]
    if region is not None:
        df = df[df["Region"] == region]

    # Preprocess 'Year' column
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')[0]
    df['ds'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
    df = df.rename(columns={'Realized': 'y'})

    # Remove rows with invalid 'ds' values
    df = df.dropna(subset=['ds'])

    # Split Data
    cutoff_year = df['ds'].max() - pd.DateOffset(years=1)
    train_data = df[df['ds'] <= cutoff_year]
    test_data = df[df['ds'] > cutoff_year]

    budget_y_test = test_data["Budget y"]
    y_test = test_data["y"]

    # Load Model and Predict
    with open(model_save_path, 'r') as fin:
        model = model_from_json(json.load(fin))

    forecast = model.predict(test_data)
    y_pred = forecast['yhat']

    # Evaluate
    evaluation_results = evaluate_model(y_test, y_pred, budget_y_test)

    # Log and Save Results
    save_results_to_csv(
        Path(f"evaluations/prophet_predictions_{acc_id}_{region}_with_r2.csv"),
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
            "Model Evaluation - R²": evaluation_results[4],
            "Budget Comparison - MAE": evaluation_results[5],
            "Budget Comparison - MSE": evaluation_results[6],
            "Budget Comparison - RMSE": evaluation_results[7],
            "Budget Comparison - MAPE": evaluation_results[8],
            "Budget Comparison - R²": evaluation_results[9],
        },
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Prophet model.")

    def none_or_str(value):
        if value == "None":
            return None
        return value

    parser.add_argument(
        "--file_path",
        type=str,
        default="data/final/merged_complete_preprocessed.csv",
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
