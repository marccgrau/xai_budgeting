import argparse
import json
import logging
import os
from pathlib import Path
import optuna
import pandas as pd
import yaml
from matplotlib import pyplot as plt, ticker
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

import seaborn as sns

setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def apply_one_hot_encoding(df, categorical_columns):
    column_transformer = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ],
        remainder="passthrough",
    )
    df_transformed = column_transformer.fit_transform(df)
    return df_transformed, column_transformer


def main(
        file_path: Path = Path("data/final/merged_double_digit.csv"), category: str = "Alle"
) -> None:
    file_path = Path(file_path)
    df = pd.read_csv(file_path, index_col=None, header=0)

    # Apply feature engineering based on YAML configuration
    df = engineer_df(df, acc_config.get(category))

    categorical_columns = ["Region", "Acc-ID"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")
    df = pd.get_dummies(df, columns=["Region", "Acc-ID"])

    # Define the cutoff year
    cutoff_year = df["Year"].max() - 1

    # Split the data into train and test sets
    train_data = df[df["Year"] <= cutoff_year]
    test_data = df[df["Year"] > cutoff_year]

    target_column = "Realized"
    exclude_columns = ["Budget y", "Budget y+1", "Slack"]

    feature_columns = [col for col in df.columns if col not in exclude_columns + [target_column]]

    # Then use these lists to split your data
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_train.to_csv("data/final/X_train.csv", index=False)

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    mse_values = []

    # Save the ColumnTransformer
    # joblib.dump(column_transformer, f"models/column_transformer_{category}.joblib")

    def objective(trial: optuna.Trial) -> float:
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        model = RandomForestRegressor(**param, random_state=acc_config.get("SEED"))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mse_values.append(mse)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=1200, n_jobs=-1)

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"Value (MSE): {trial.value}")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save the best hyperparameters
    best_hyperparams = trial.params
    with open(f"hyperparameters/hyperparams_rf_{category}.json", "w") as f:
        json.dump(best_hyperparams, f)
    logger.info("Best hyperparameters of RandomForest saved successfully")

    # Train and save the best model
    best_model = RandomForestRegressor(**best_hyperparams)
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f"models/best_model_rf_{category}.joblib")
    logger.info("Best RandomForest model saved successfully")

    sorted_mse_values = sorted(mse_values, reverse=True)

    # Convert the list into a DataFrame for seaborn
    df = pd.DataFrame({'Trial': range(1, len(sorted_mse_values) + 1), 'MSE': sorted_mse_values})

    # Set the seaborn style
    sns.set(style='whitegrid')

    # Create the plot
    plt.figure(figsize=(12, 7))
    plot = sns.lineplot(x='Trial', y='MSE', data=df, marker='o', color='coral', linewidth=2.5, markersize=8, alpha=0.85,
                        label='MSE per Trial')

    # Filling the area under the plot line
    plt.fill_between(df['Trial'], df['MSE'], color='coral', alpha=0.3)

    # Enhancing the plot with seaborn's and matplotlib's functionalities
    plot.set_title('Optuna Optimization Progress - MSE Minimization', fontsize=16, fontweight='bold', color='#333333')
    plot.set_xlabel('Trial', fontsize=14, labelpad=15, color='#333333')
    plot.set_ylabel('MSE', fontsize=14, labelpad=15, color='#333333')

    # Formatting the y-axis to display large numbers without scientific notation
    plot.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
    plt.xlim(left=1)

    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Save the plot
    plot_path = os.path.join(plots_dir, "RF Model Optimization Progress.png")
    plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune a RandomForest model")
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
