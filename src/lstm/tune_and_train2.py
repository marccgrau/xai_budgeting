import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.feature_engineering import engineer_df
from src.logging_config import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

acc_config_path = Path("config/acc_config.yaml")
with open(acc_config_path, "r") as yaml_file:
    acc_config = yaml.safe_load(yaml_file)

def prepare_data(df, acc_config):
    df = engineer_df(df, acc_config)
    df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-01-01')
    df = df.rename(columns={'Realized': 'y'})
    for i in range(1, 4):
        df[f'y(t-{i})'] = df['y'].shift(i)

    df.dropna(inplace=True)
    logger.info("Data preparation complete.")

    return df


def create_rnn_forecast(train_data, test_data):
    X_train = train_data[[f'y(t-{i})' for i in range(1, 4)]].values
    y_train = train_data['y'].values

    X_test = test_data[[f'y(t-{i})' for i in range(1, 4)]].values
    y_test = test_data['y'].values

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    yhat = model.predict(X_test)
    test_data.loc[:, 'yhat'] = yhat
    logger.info("LSTM model training and prediction complete.")

    return test_data



def main(file_path: Path, category: str):
    df = pd.read_csv(file_path, index_col=None, header=0)
    df['Year'] = df['Year'].astype(int)

    total_combinations = df.groupby(['Region', 'Acc-ID']).ngroups
    processed_combinations = 0

    all_results = pd.DataFrame()

    for (region, acc_id), group_df in df.groupby(['Region', 'Acc-ID']):
        processed_combinations += 1
        logger.info(f"Processing combination {processed_combinations} of {total_combinations}")
        logger.info(f"Processing data for Region: {region}, Account ID: {acc_id}")

        group_df = group_df[group_df['Year'].apply(lambda x: str(x).isdigit())]

        if len(group_df) < 4:
            logger.warning(f"Insufficient data for Region: {region}, Account ID: {acc_id}. Skipping.")
            continue

        try:
            group_df = prepare_data(group_df, acc_config.get(category))

            train_data = group_df[group_df['ds'].dt.year < 2022]
            test_data = group_df[group_df['ds'].dt.year >= 2022]

            if len(train_data) < 4 or len(test_data) == 0:
                logger.warning(f"Insufficient train or test data for Region: {region}, Account ID: {acc_id}. Skipping.")
                continue

            forecast = create_rnn_forecast(train_data, test_data)

            forecast.loc[:, 'Region'] = region
            forecast.loc[:, 'Acc-ID'] = acc_id
            all_results = pd.concat([all_results, forecast], ignore_index=True)

            all_results.to_csv("data/final/all_results.csv", index=False)

        except Exception as e:
            logger.error(f"Error processing Region: {region}, Account ID: {acc_id}. Error: {str(e)}")
            continue

    if not all_results.empty:
        mse = mean_squared_error(all_results['y'], all_results['yhat'])
        mae = mean_absolute_error(all_results['y'], all_results['yhat'])
        rmse = np.sqrt(mse)

        logger.info("Combined metrics across all processed combinations:")
        logger.info(f"Combined MSE: {mse}")
        logger.info(f"Combined MAE: {mae}")
        logger.info(f"Combined RMSE: {rmse}")

    else:
        logger.warning("No results were calculated. Check if any combinations were successfully processed.")

    logger.info("Processing completed for all regions and account combinations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create RNN forecast and calculate combined metrics")
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
    logger.info("Starting main function.")
    main(file_path=Path(args.file_path), category=args.category)
    logger.info("Main function completed.")



