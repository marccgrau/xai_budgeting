import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import yaml
import optuna
from src.feature_engineering import engineer_df
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

def preprocess_data(df, categorical_features, numerical_features):
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])
    X = preprocessor.fit_transform(df.drop(['Realized'], axis=1))
    y = df['Realized'].values
    return X, y

def train_model_with_optuna(trial, X_train, y_train, X_val, y_val):
    hidden_size = trial.suggest_int('hidden_size', 20, 100)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    model = LSTMModel(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(output, torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
        if torch.isnan(loss):
            logger.warning(f"NaN loss encountered at epoch {epoch}")
            return float('inf')
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_val, dtype=torch.float32))
        mse = criterion(predictions, torch.tensor(y_val, dtype=torch.float32).view(-1, 1))
        if torch.isnan(mse):
            logger.warning(f"NaN MSE encountered")
            return float('inf')

    # Save predictions and actual values to a CSV file
    result_df = pd.DataFrame({
        'Actual': y_val,
        'Predicted': predictions.numpy().flatten()
    })
    result_df.to_csv('predictions_vs_actuals.csv', index=False)
    logger.info("Predictions and actual values saved to predictions_vs_actuals.csv")

    return mse.item()

def objective(trial, X_train, y_train, X_val, y_val):
    return train_model_with_optuna(trial, X_train, y_train, X_val, y_val)

def main(file_path, category):
    acc_config_path = Path("config/acc_config.yaml")
    with open(acc_config_path, "r") as yaml_file:
        acc_config = yaml.safe_load(yaml_file)

    df = pd.read_csv(file_path)
    df = engineer_df(df, acc_config.get(category, {}))
    df = df.dropna()

    cutoff_year = df["Year"].max() - 1

    train_data = df[df["Year"] <= cutoff_year]
    test_data = df[df["Year"] > cutoff_year]

    target_column = "Realized"
    exclude_columns = ["Budget y", "Budget y+1", "Slack"]

    feature_columns = [col for col in df.columns if col not in exclude_columns + [target_column]]
    categorical_features = [col for col in feature_columns if df[col].dtype == 'object']
    numerical_features = [col for col in feature_columns if col not in categorical_features]

    X_train, y_train = preprocess_data(train_data, categorical_features=categorical_features,
                                       numerical_features=numerical_features)
    X_val, y_val = preprocess_data(test_data, categorical_features=categorical_features,
                                   numerical_features=numerical_features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=10)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value (MSE): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {study.best_trial.params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model with Optuna optimization")
    parser.add_argument("--file_path", type=str, default="data/final/merged_complete_preprocessed.csv",
                        help="Path to the dataset")
    parser.add_argument("--category", type=str, default="Alle", help="Category of the dataset to use for training")
    args = parser.parse_args()

    main(args.file_path, args.category)
