import argparse
import logging
import os
import yaml

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

# Setup logging configuration
setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)


def prepare_data(df, acc_config):
    # Apply feature engineering
    df = engineer_df(df, acc_config)

    # Ensure 'ds' is datetime
    df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-01-01')
    df = df.rename(columns={'Realized': 'y'})

    # Select a subset of potentially important features
    # selected_features = ['Alter 20–64', 'Staatsangehörigkeit Ausländer', 'Einwanderung',
    #                          'BWS Transport, IT-Dienstleistung', 'BIP']

    # Prepare the dataframe for Prophet
    prophet_columns = ['ds', 'y'] #+ selected_features
    df = df[prophet_columns].sort_values('ds')

    # Remove duplicate dates, keeping the last occurrence
    df = df.drop_duplicates(subset='ds', keep='last')

    # Scale the features
    # scaler = StandardScaler()
    # df[selected_features] = scaler.fit_transform(df[selected_features])

    # Log the columns being used
    logger.info(f"Columns used for Prophet: {df.columns.tolist()}")

    return df


def create_prophet_forecast(train_data, test_data):
    # Create and fit the model
    model = Prophet()

    # Add all columns except 'ds' and 'y' as regressors
    for column in train_data.columns:
        if column not in ['ds', 'y']:
            model.add_regressor(column)

    model.fit(train_data)

    # Create future dataframe for predictions
    future = model.make_future_dataframe(periods=len(test_data), freq='YS')

    # Add regressor values to the future dataframe
    for column in train_data.columns:
        if column not in ['ds', 'y']:
            # For historical dates, use values from train_data
            future.loc[future['ds'].isin(train_data['ds']), column] = train_data[column].values
            # For forecast dates, use values from test_data if available
            if column in test_data.columns:
                future.loc[future['ds'].isin(test_data['ds']), column] = test_data[column].values
            else:
                # If the column is not in test_data, use the last value from train_data
                future.loc[future['ds'].isin(test_data['ds']), column] = train_data[column].iloc[-1]

    # Make predictions
    forecast = model.predict(future)

    return forecast


def main(file_path: Path = Path("data/final/merged_complete_preprocessed.csv"), category: str = "Alle"):
    file_path = Path(file_path)
    df = pd.read_csv(file_path, index_col=None, header=0)
    df['Year'] = df['Year'].astype(int)

    total_combinations = df.groupby(['Region', 'Acc-ID']).ngroups
    processed_combinations = 0

    # DataFrame to store all test data and forecasts
    all_results = pd.DataFrame()

    for (region, acc_id), group_df in df.groupby(['Region', 'Acc-ID']):
        processed_combinations += 1
        logger.info(f"Processing combination {processed_combinations} of {total_combinations}")
        logger.info(f"Processing data for Region: {region}, Account ID: {acc_id}")

        group_df = group_df[group_df['Year'].apply(lambda x: str(x).isdigit())]

        if len(group_df) < 2:
            logger.warning(f"Insufficient data for Region: {region}, Account ID: {acc_id}. Skipping.")
            continue

        try:
            group_df = prepare_data(group_df, acc_config.get(category))

            # Split into train and test
            train_data = group_df[group_df['ds'].dt.year < 2022]
            test_data = group_df[group_df['ds'].dt.year >= 2022]

            if len(train_data) < 2 or len(test_data) == 0:
                logger.warning(f"Insufficient train or test data for Region: {region}, Account ID: {acc_id}. Skipping.")
                continue

            # Create forecast
            forecast = create_prophet_forecast(train_data, test_data)

            # Merge forecast with actual values
            results = pd.merge(test_data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
            results['Region'] = region
            results['Acc-ID'] = acc_id

            # Append to all_results
            all_results = pd.concat([all_results, results], ignore_index=True)

            all_results.to_csv("data/final/all_results.csv", index=False)

        except Exception as e:
            logger.error(f"Error processing Region: {region}, Account ID: {acc_id}. Error: {str(e)}")
            continue

    # Calculate combined metrics
    if not all_results.empty:
        mse = np.mean((all_results['y'] - all_results['yhat']) ** 2)
        mae = np.mean(np.abs(all_results['y'] - all_results['yhat']))
        rmse = np.sqrt(mse)

        logger.info("Combined metrics across all processed combinations:")
        logger.info(f"Combined MSE: {mse}")
        logger.info(f"Combined MAE: {mae}")
        logger.info(f"Combined RMSE: {rmse}")

    else:
        logger.warning("No results were calculated. Check if any combinations were successfully processed.")

    logger.info("Processing completed for all regions and account combinations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Prophet forecast and calculate combined metrics")
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
