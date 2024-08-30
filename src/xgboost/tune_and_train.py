import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.integration import XGBoostPruningCallback
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb
from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def apply_one_hot_encoding(df, categorical_columns):
    # Apply OneHotEncoding to categorical columns
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        ],
        remainder="passthrough",
    )
    df_transformed = column_transformer.fit_transform(df)
    # feature_names = column_transformer.get_feature_names_out()
    # df_encoded = pd.DataFrame(df_transformed, columns=feature_names)
    return df_transformed, column_transformer


def main(
    file_path: Path = Path("data/final/merged_double_digit.csv"), category: str = "Alle"
) -> None:
    random.seed(acc_config.get("SEED"))
    np.random.seed(acc_config.get("SEED"))

    file_path = Path(file_path)
    df = pd.read_csv(file_path, index_col=None, header=0)

    # Apply feature engineering based on YAML configuration
    df = engineer_df(df, acc_config.get(category))

    # Encode categorical variables
    categorical_columns = ["Region", "Acc-ID"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Define the cutoff year
    cutoff_year = df["Year"].max() - 1

    # Split the data into train and test sets
    train_data = df[df["Year"] <= cutoff_year]
    test_data = df[df["Year"] > cutoff_year]

    X_train = train_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
    y_train = train_data["Realized"]

    X_test = test_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
    y_test = test_data["Realized"]

    def objective(trial: optuna.Trial) -> float:
        param = {
            # XGBoost specific parameters
            "objective": "reg:squarederror",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }

        model = xgb.XGBRegressor(
            **param, enable_categorical=True, random_state=acc_config.get("SEED")
        )
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",
            early_stopping_rounds=100,
            verbose=False,
            callbacks=[pruning_callback],
        )

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return mse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=acc_config.get("SEED")),
    )
    study.optimize(objective, n_trials=100, timeout=1200)

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"Value (MSE): {trial.value}")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save the best hyperparameters
    best_hyperparams = trial.params
    with open(f"hyperparameters/hyperparams_xgboost_{category}.json", "w") as f:
        json.dump(best_hyperparams, f)
    logger.info("Best hyperparameters of XGBoost saved successfully")

    # Train and save the best model
    best_model = xgb.XGBRegressor(
        **best_hyperparams, enable_categorical=True, random_state=acc_config.get("SEED")
    )
    best_model.fit(X_train, y_train)
    best_model.save_model(f"models/best_model_xgboost_{category}.json")
    logger.info(f'Max year in dataframe: {df["Year"].max()}')
    logger.info(f"Cutoff year: {cutoff_year}")
    logger.info(f'Train years: {train_data["Year"].unique()}')
    logger.info("Best XGBoost model saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune an XGBoost model")
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
