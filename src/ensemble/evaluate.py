import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

import catboost as cb
import xgboost as xgb
from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def load_model_hyperparams(hyperparams_path: Path) -> Dict[str, float]:
    """Load model hyperparameters from JSON file."""
    with open(hyperparams_path, "r") as file:
        hyperparams = json.load(file)
    return hyperparams


def prepare_data(
    data_path: Path, encode_for_xgb: bool = False, category: str = "Alle"
) -> pd.DataFrame:
    """Prepare the dataset for evaluation."""
    df = pd.read_csv(data_path)
    df["Year"] = df["Year"] - df["Year"].min()
    df = df.sort_values(by="Year")
    df = engineer_df(df, acc_config.get(category))

    if encode_for_xgb:
        # Assuming label encoders are stored and loaded similarly to the XGBoost training script
        region_encoder_path = Path("models/le_Region.joblib")
        acc_id_encoder_path = Path("models/le_Acc-ID.joblib")
        le_region = joblib.load(region_encoder_path)
        le_acc_id = joblib.load(acc_id_encoder_path)
        df["Region"] = le_region.transform(df["Region"])
        df["Acc-ID"] = le_acc_id.transform(df["Acc-ID"])

    return df


def load_and_predict(model_path: Path, X_test: pd.DataFrame) -> np.ndarray:
    """Load a trained model and predict."""
    if "catboost" in str(model_path):
        model = cb.CatBoostRegressor()
    else:
        model = xgb.XGBRegressor()
    model.load_model(model_path)
    predictions = model.predict(X_test)
    return predictions


def evaluate_ensemble(
    predictions: Tuple[np.ndarray, np.ndarray], y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate the ensemble model and return metrics."""
    ensemble_preds = np.mean(predictions, axis=0)
    metrics = {
        "MAE": mean_absolute_error(y_test, ensemble_preds),
        "MSE": mean_squared_error(y_test, ensemble_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, ensemble_preds)),
        "MAPE": mean_absolute_percentage_error(y_test, ensemble_preds),
    }
    return metrics


def log_and_save_evaluation_results(
    results_file_path: Path,
    metrics: Dict[str, float],
    comparison_metrics: Dict[str, float],
) -> None:
    """Log the evaluation and comparison results."""
    with open(results_file_path, "w", encoding="utf-8") as file:
        for key, value in metrics.items():
            logger.info(f"Ensemble {key}: {value}")
            file.write(f"Ensemble {key}: {value}\n")
        for key, value in comparison_metrics.items():
            logger.info(f"Budget Comparison {key}: {value}")
            file.write(f"Budget Comparison {key}: {value}\n")


def main(
    file_path: Path = Path("data/final/merged_double_digit.csv"), category: str = "Alle"
) -> None:

    # Paths configuration
    xgb_model_path = Path("models/best_model_xgboost.json")
    cat_model_path = Path("models/best_model_catboost.cbm")
    file_path = Path(file_path)
    results_file_path = Path("evaluations/evaluation_ensemble.txt")

    # Data preparation
    df_xgb = prepare_data(file_path, encode_for_xgb=True, category=category)
    df_cat = prepare_data(file_path, encode_for_xgb=False, category=category)

    cutoff_year = df_xgb["Year"].max() - 1
    X_test_xgb = df_xgb[df_xgb["Year"] > cutoff_year].drop(
        columns=["Realized", "Budget y", "Budget y+1", "Slack"]
    )
    X_test_cat = df_cat[df_cat["Year"] > cutoff_year].drop(
        columns=["Realized", "Budget y", "Budget y+1", "Slack"]
    )
    y_test = df_xgb[df_xgb["Year"] > cutoff_year]["Realized"].values

    # Model predictions
    xgb_preds = load_and_predict(xgb_model_path, X_test_xgb)
    cat_preds = load_and_predict(cat_model_path, X_test_cat)

    # Ensemble evaluation
    ensemble_metrics = evaluate_ensemble((xgb_preds, cat_preds), y_test)

    # Compare ensemble predictions with budget_y_test
    budget_y_test = df_xgb[df_xgb["Year"] > cutoff_year]["Budget y"].values
    budget_comparison_metrics = {
        "MAE": mean_absolute_error(y_test, budget_y_test),
        "MSE": mean_squared_error(y_test, budget_y_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, budget_y_test)),
        "MAPE": mean_absolute_percentage_error(y_test, budget_y_test),
    }

    # Log and save the results
    log_and_save_evaluation_results(
        results_file_path, ensemble_metrics, budget_comparison_metrics
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an ensemble model")
    parser.add_argument(
        "--file_path",
        type=Path,
        default=Path("data/final/merged_double_digit.csv"),
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
