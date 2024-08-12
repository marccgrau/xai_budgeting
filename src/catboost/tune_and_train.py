import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import mean_squared_error

import catboost as cb
from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def main(
    file_path: Path = Path("data/final/merged_double_digit.csv"),
    category: str = "Alle",
) -> None:
    random.seed(acc_config.get("SEED"))
    np.random.seed(acc_config.get("SEED"))
    # Replace with your actual data loading code
    file_path = Path(file_path)
    df = pd.read_csv(file_path, index_col=None, header=0)

    # Sort dataframe and apply feature engineering
    df = engineer_df(df, acc_config.get(category))
    cat_features = ["Region", "Acc-ID"]
    df[cat_features] = df[cat_features].astype(str)

    # Define the cutoff year
    cutoff_year = df["Year"].max() - 1
    logger.info(f"Cutoff year: {cutoff_year}")

    # Split the data into train and test sets depending on the cutoff year
    train_data = df[df["Year"] <= cutoff_year]
    test_data = df[df["Year"] > cutoff_year]

    X_train = train_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
    y_train = train_data["Realized"]

    X_test = test_data.drop(columns=["Realized", "Budget y", "Budget y+1", "Slack"])
    y_test = test_data["Realized"]

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=acc_config.get("SEED")),
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        param = {
            # CatBoost regression parameters
            "objective": "RMSE",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.01, 0.2, log=True
            ),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "iterations": trial.suggest_int("iterations", 100, 1500),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "10gb",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

        gbm = cb.CatBoostRegressor(
            **param,
            cat_features=["Region", "Acc-ID"],
            random_seed=acc_config.get("SEED"),
        )

        pruning_callback = CatBoostPruningCallback(trial, metric="RMSE")
        gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=0,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
        )

        preds = gbm.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        return mse

    study.optimize(objective, n_trials=100, timeout=1200)

    logger.info("Number of finished trials: {}".format(len(study.trials)))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("Value (MSE): {}".format(trial.value))

    logger.info("Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # Save the best hyperparameters
    best_hyperparams = trial.params
    with open(f"hyperparameters/hyperparams_catboost_{category}.json", "w") as f:
        json.dump(best_hyperparams, f)
    logger.info("Best hyperparameters of catboost saved successfully")

    best_model = cb.CatBoostRegressor(
        **best_hyperparams,
        cat_features=["Region", "Acc-ID"],
        random_seed=acc_config.get("SEED"),
    )
    best_model.fit(X_train, y_train)
    best_model.save_model(f"models/best_model_catboost_{category}.cbm", format="cbm")
    logger.info("Best model saved successfully")
    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Train years: {X_train['Year'].unique()}")
    logger.info(f"Test years: {X_test['Year'].unique()}")
    logger.info(f"Train columns: {X_train.columns}")
    logger.info(f"Test columns: {X_test.columns}")
    logger.info(f"Train data head: {X_train.head()}")
    logger.info(f"Test data head: {X_test.head()}")
    logger.info(f"X_train years: {X_train['Year'].unique()}")
    logger.info(f"X_test years: {X_test['Year'].unique()}")
    logger.info(f"Train columns: {X_train.columns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune a CatBoost model")
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
