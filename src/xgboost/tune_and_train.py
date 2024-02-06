import json
import logging
from pathlib import Path

import joblib
import optuna
import pandas as pd
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from src.feature_engineering import (
    apply_feature_engineering,
    choose_acc_ids,
    drop_all_zero_entries,
)
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Replace with your actual data loading code
file_path = Path("data/final/merged_double_digit.csv")
df = pd.read_csv(file_path, index_col=None, header=0)

# Convert 'Year' to a relative year and apply feature engineering
df["Year"] = df["Year"] - df["Year"].min()
df = df.sort_values(by="Year")
df = apply_feature_engineering(df)
df = drop_all_zero_entries(df)
df = choose_acc_ids(df=df, acc_ids=[40, 41, 42, 43, 45, 46, 47, 49])
df = df.dropna()

# Encode categorical variables
le_region = LabelEncoder()
df["Region"] = le_region.fit_transform(df["Region"])

le_acc_id = LabelEncoder()
df["Acc-ID"] = le_acc_id.fit_transform(df["Acc-ID"])

# Save the encoders to file
joblib.dump(le_region, "models/le_region.joblib")
joblib.dump(le_acc_id, "models/le_acc_id.joblib")

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
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
    }

    model = xgb.XGBRegressor(**param)

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False,
        callbacks=[pruning_callback],
    )

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse


if __name__ == "__main__":
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
