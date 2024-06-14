import argparse
import json
import logging
import os
import yaml

from pathlib import Path

import optuna
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from prophet import Prophet
from prophet.serialize import model_to_json

from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

# Setup logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# Load account configuration
acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def main(file_path: Path = Path("data/final/merged_complete_preprocessed.csv"), category: str = "Alle") -> None:
    """
    Main function to train and tune a Prophet model.

    Parameters:
    file_path (Path): The path to the dataset.
    category (str): The category of the dataset to use for training.

    Returns:
    None
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, index_col=None, header=0)
    df['Year'] = df['Year'].astype(int)
    df = engineer_df(df, acc_config.get(category))

    df = df[df['Year'].apply(lambda x: str(x).isdigit())]

    df['Year'] = df['Year'].astype(str)
    df['ds'] = pd.to_datetime(df['Year'], format='%Y')
    df = df.rename(columns={'Realized': 'y'})

    cutoff_year = df['ds'].max() - pd.DateOffset(years=1)
    train_data = df[df['ds'] <= cutoff_year]
    test_data = df[df['ds'] > cutoff_year]

    exclude_columns = ["Budget y", "Budget y+1", "Slack"]
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    train_data = train_data[feature_columns]
    test_data = test_data[feature_columns]

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna hyperparameter optimization.

        Parameters:
        trial (optuna.Trial): A trial object for hyperparameter suggestions.

        Returns:
        float: Mean Squared Error of the forecast.
        """
        param = {
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": 0.01,
            "holidays_prior_scale": 0.01,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False
        }
        model = Prophet(**param)
        model.fit(train_data)

        forecast = model.predict(train_data)
        mse = ((forecast['yhat'] - train_data['y']) ** 2).mean()
        return mse

    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=1200, n_jobs=-1)

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"Value (MSE): {trial.value}")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save best hyperparameters
    best_hyperparams = trial.params
    with open(f"hyperparameters/hyperparams_prophet_{category}.json", "w") as f:
        json.dump(best_hyperparams, f)
    logger.info("Best hyperparameters of Prophet saved successfully")

    # Train best model and save it
    best_model = Prophet(**best_hyperparams)
    best_model.fit(train_data)
    with open(f"models/best_model_prophet_{category}.json", 'w') as fout:
        json.dump(model_to_json(best_model), fout)
    logger.info("Best Prophet model saved successfully")

    # Forecast and plot results
    future_periods = min(len(test_data), 1)
    future = best_model.make_future_dataframe(periods=future_periods, freq='Y')
    forecast = best_model.predict(future)
    fig = best_model.plot(forecast)
    plt.title('Forecasted vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(['Actual', 'Forecasted'])
    plt.show()

    # Save the plot
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plot_path = os.path.join(plots_dir, "Prophet_Model_Optimization_Forecast.png")
    fig.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune a Prophet model")
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
