import argparse
import json
import logging
import random
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def fit_encode_categorical_columns(
    df: pd.DataFrame, categorical_columns: list
) -> tuple:
    encoders = {}
    for col in categorical_columns:
        encoder = OneHotEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    return df, encoders


def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    return df


def main(
    file_path: Path = Path("data/final/merged_double_digit.csv"), category: str = "Alle"
) -> None:
    random.seed(acc_config.get("SEED"))
    np.random.seed(acc_config.get("SEED"))

    file_path = Path(file_path)
    df = pd.read_csv(file_path, index_col=None, header=0)

    # Apply feature engineering based on YAML configuration
    df = engineer_df(df, acc_config.get(category))

    # Define the cutoff year and split the dataset
    cutoff_year = df["Year"].max() - 1
    train_data = df[df["Year"] <= cutoff_year]
    test_data = df[df["Year"] > cutoff_year]

    categorical_features = ["Region", "Acc-ID"]
    numeric_features = [
        col for col in df.columns if col not in categorical_features + ["Realized", "Budget y", "Budget y+1", "Slack"]
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    target_column = "Realized"
    exclude_columns = ["Budget y", "Budget y+1", "Slack"]

    feature_columns = [col for col in df.columns if col not in exclude_columns + [target_column]]

    # Then use these lists to split your data
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    X_train = preprocessor.fit_transform(X_train)

    X_test = preprocessor.transform(X_test)

    # Save preprocessor
    joblib.dump(preprocessor, Path(f"models/preprocessor_hgb_{category}.joblib"))

    def objective(trial: optuna.Trial) -> float:
        param = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 200),
        }

        model = HistGradientBoostingRegressor(
            **param, random_state=acc_config.get("SEED")
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        budget_y_test = test_data["Budget y"]
        results = pd.DataFrame({
            "Realized": y_test,
            "Predicted": preds,
            "Budget y": budget_y_test
        })

        results.to_csv('predictions.csv', index=False)

        return mse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=acc_config.get("SEED")),
    )
    study.optimize(objective, n_trials=100, timeout=3600)

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"Value (MSE): {trial.value}")
    for key, value in trial.params.items():
        logger.info(f"{key}: {value}")

    # Save the best hyperparameters
    best_hyperparams = trial.params
    with open(f"hyperparameters/hyperparams_hgb_{category}.json", "w") as f:
        json.dump(best_hyperparams, f)
    logger.info("Best hyperparameters saved successfully")

    # Train and save the best model
    best_model = HistGradientBoostingRegressor(
        **best_hyperparams, random_state=acc_config.get("SEED")
    )
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, Path(f"models/best_model_hgb_{category}.joblib"))
    logger.info("Best model saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and tune a HistGradientBoostingRegressor model"
    )
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
    main(file_path=Path(args.file_path), category=args.category)
