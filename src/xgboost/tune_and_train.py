import argparse
import json
import logging
from pathlib import Path

import joblib
import optuna
import pandas as pd
import yaml
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


# Function to apply label encoding
def encode_categorical_columns(df, categorical_columns):
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def main(
    file_path: Path = Path("data/final/merged_double_digit.csv"), category: str = "Alle"
) -> None:
    file_path = Path(file_path)
    df = pd.read_csv(file_path, index_col=None, header=0)

    # Apply feature engineering based on YAML configuration
    df = engineer_df(df, acc_config.get(category))

    # Encode categorical variables
    categorical_columns = ["Region", "Acc-ID"]
    df, encoders = encode_categorical_columns(df, categorical_columns)

    # Save the encoders for future use (prediction phase)
    for col, le in encoders.items():
        joblib.dump(le, Path(f"models/le_{col}.joblib"))

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

        model = xgb.XGBRegressor(**param)
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

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=1200)

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"Value (MSE): {trial.value}")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save the best hyperparameters
    best_hyperparams = trial.params
    with open("hyperparameters/hyperparams_xgboost.json", "w") as f:
        json.dump(best_hyperparams, f)
    logger.info("Best hyperparameters of XGBoost saved successfully")

    # Train and save the best model
    best_model = xgb.XGBRegressor(**best_hyperparams)
    best_model.fit(X_train, y_train)
    best_model.save_model("models/best_model_xgboost.json")
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
