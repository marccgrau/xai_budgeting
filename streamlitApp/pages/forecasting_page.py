import streamlit as st
import pandas as pd
import numpy as np
import os
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import joblib
from models.catboost_model import train_catboost
from models.random_forest_model import train_random_forest
from models.xgboost_model import train_xgboost
from models.lstm_model import run_lstm_model
from utils.feature_engineering import engineer_df
from utils.metrics import calculate_metrics
from models.prophet_model import run_prophet_model
import logging

logging.basicConfig(level=logging.INFO)


def list_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]


def prepare_data(df, acc_ids=None):
    df = engineer_df(df, acc_ids)
    categorical_columns = ["Region", "Acc-ID"]
    numerical_columns = df.columns.difference(categorical_columns + ["Realized", "Budget y", "Budget y+1", "Slack"])
    df[numerical_columns] = df[numerical_columns].fillna(0)
    df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]).astype("category"))

    cutoff_year = df["Year"].max() - 1
    train_data = df[df["Year"] <= cutoff_year]
    test_data = df[df["Year"] > cutoff_year]
    target_column = "Realized"
    exclude_columns = ["Budget y", "Budget y+1", "Slack"]
    feature_columns = [col for col in df.columns if col not in exclude_columns + [target_column]]

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    budget_y_test = test_data["Budget y"]

    return X_train, X_test, y_train, y_test, budget_y_test, feature_columns


def evaluate_model(y_test, y_pred, budget_y_test):
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    rmse = round(np.sqrt(mse), 2)
    comparison_mae = round(mean_absolute_error(y_test, budget_y_test), 2)
    comparison_mse = round(mean_squared_error(y_test, budget_y_test), 2)
    comparison_rmse = round(np.sqrt(comparison_mse), 2)
    return mae, mse, rmse, comparison_mae, comparison_mse, comparison_rmse


def forecasting_page():
    st.title("Forecasting")
    st.write("Upload a dataset or choose from existing datasets and choose a model to train.")

    data_directory = 'data'
    available_files = list_csv_files(data_directory)
    file_choice = st.selectbox("Select a file or upload a new one", ["Upload a new file"] + available_files)

    if file_choice == "Upload a new file":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(os.path.join(data_directory, file_choice))

    if 'df' in locals():
        st.write("Dataset:")
        st.write(df.head())

        model_type = st.selectbox("Select model", ["XGBoost", "RandomForest", "CatBoost", "LSTM", "Prophet"])

        if model_type == "XGBoost":
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, (0.01, 0.3))
            n_estimators = st.slider("Number of Estimators", 100, 1000, (100, 1000))
            max_depth = st.slider("Max Depth", 3, 12, (3, 12))
            subsample = st.slider("Subsample", 0.5, 1.0, (0.5, 1.0))
            colsample_bytree = st.slider("Colsample by Tree", 0.1, 1.0, (0.1, 1.0))
        elif model_type == "RandomForest":
            n_estimators = st.slider("Number of Estimators", 100, 500, (100, 500))
            max_depth = st.slider("Max Depth", 3, 20, (3, 10))
            min_samples_split = st.slider("Min Samples Split", 2, 20, (5, 20))
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, (5, 20))
        elif model_type == "CatBoost":
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, (0.01, 0.3))
            iterations = st.slider("Iterations", 100, 1000, (100, 1000))
            depth = st.slider("Depth", 4, 10, (4, 10))
        elif model_type == "LSTM":
            num_layers = st.slider("Number of LSTM Layers", 1, 100, 50)
            activation = st.selectbox("Activation Function", ["relu", "tanh", "sigmoid"],
                                      index=0)
        if model_type == "Prophet":
            st.write("Prophet model does not require hyperparameters.")
        if model_type != "Prophet":
            n_trials = st.slider("Number of Trials", 1, 200, 1)
        else:
            n_trials = 1

        X_train, X_test, y_train, y_test, budget_y_test, feature_columns = prepare_data(df)

        if model_type == "RandomForest":
            X_train = pd.get_dummies(X_train, columns=["Region", "Acc-ID"])
            X_test = pd.get_dummies(X_test, columns=["Region", "Acc-ID"])
            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            y_train = y_train.fillna(0)
            y_test = y_test.fillna(0)
        elif model_type == "CatBoost":
            cat_features = ["Region", "Acc-ID"]
            X_train[cat_features] = X_train[cat_features].astype(str)
            X_test[cat_features] = X_test[cat_features].astype(str)

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                def objective(trial):
                    try:
                        if model_type == "XGBoost":
                            params = {
                                "learning_rate": trial.suggest_float("learning_rate", learning_rate[0],
                                                                     learning_rate[1]),
                                "n_estimators": trial.suggest_int("n_estimators", n_estimators[0], n_estimators[1]),
                                "max_depth": trial.suggest_int("max_depth", max_depth[0], max_depth[1]),
                                "subsample": trial.suggest_float("subsample", subsample[0], subsample[1]),
                                "colsample_bytree": trial.suggest_float("colsample_bytree", colsample_bytree[0],
                                                                        colsample_bytree[1]),
                            }
                            preds, model = train_xgboost(X_train, y_train, X_test, y_test, params)
                        elif model_type == "RandomForest":
                            params = {
                                "n_estimators": trial.suggest_int("n_estimators", n_estimators[0], n_estimators[1]),
                                "max_depth": trial.suggest_int("max_depth", max_depth[0], max_depth[1]),
                                "min_samples_split": trial.suggest_int("min_samples_split", min_samples_split[0],
                                                                       min_samples_split[1]),
                                "min_samples_leaf": trial.suggest_int("min_samples_leaf", min_samples_leaf[0],
                                                                      min_samples_leaf[1]),
                            }
                            preds, model = train_random_forest(X_train, y_train, X_test, y_test, params)
                        elif model_type == "CatBoost":
                            params = {
                                "learning_rate": trial.suggest_float("learning_rate", learning_rate[0],
                                                                     learning_rate[1]),
                                "iterations": trial.suggest_int("iterations", iterations[0], iterations[1]),
                                "depth": trial.suggest_int("depth", depth[0], depth[1]),
                            }
                            preds, model = train_catboost(X_train, y_train, X_test, y_test, cat_features, params)

                        if np.isnan(preds).any() or np.isinf(preds).any():
                            raise ValueError("Predictions contain NaN or infinity values")
                        mse = mean_squared_error(y_test, preds)
                        return mse
                    except Exception as e:
                        logging.error(f"Error during trial: {str(e)}")
                        return np.inf

                if model_type != "Prophet" and model_type != "LSTM":
                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=n_trials, timeout=1200, n_jobs=-1)

                    if not study.trials:
                        st.error("No trials were completed successfully.")
                        return

                    trial = study.best_trial
                    best_hyperparams = trial.params

                    os.makedirs('hyperparameters', exist_ok=True)
                    with open(f"hyperparameters/hyperparams_{model_type.lower()}.json", "w") as f:
                        json.dump(best_hyperparams, f)
                    st.write(f"Best hyperparameters of {model_type} saved successfully")

                    if model_type == "XGBoost":
                        best_model = xgb.XGBRegressor(
                            **best_hyperparams,
                            objective="reg:squarederror",
                            eval_metric="rmse",
                            enable_categorical=True,
                            random_state=42
                        )
                    elif model_type == "RandomForest":
                        best_model = RandomForestRegressor(**best_hyperparams, random_state=42)
                    elif model_type == "CatBoost":
                        best_model = CatBoostRegressor(cat_features=cat_features, **best_hyperparams, verbose=0)
                    best_model.fit(X_train, y_train)
                    os.makedirs('models', exist_ok=True)
                    if model_type == "XGBoost":
                        best_model.save_model(f"models/best_model_{model_type.lower()}.json")
                    elif model_type == "RandomForest":
                        joblib.dump(best_model, f"models/best_model_{model_type.lower()}.joblib")
                    elif model_type == "CatBoost":
                        best_model.save_model(f"models/best_model_{model_type.lower()}.cbm")

                    st.success(f"Training of {model_type} model completed successfully!")

                    preds = best_model.predict(X_test)
                    results = pd.DataFrame({
                        "Realized": y_test,
                        "Predicted": preds,
                        "Budget y": budget_y_test
                    })
                    results.to_csv(f"results/{model_type.lower()}_results.csv", index=False)
                    mae, mse, rmse = calculate_metrics(y_test, preds)
                    comparison_mae, comparison_mse, comparison_rmse = evaluate_model(y_test, preds, budget_y_test)[3:]

                    with st.expander("Best Trial Results", expanded=True):
                        st.subheader("Best Trial")
                        st.write(f"Value (MSE): {trial.value:.4f}")

                        st.subheader("Best Hyperparameters")
                        for key, value in trial.params.items():
                            st.write(f"{key}: {value}")

                    with st.expander("Model Performance", expanded=True):
                        st.subheader("Evaluation Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("MAE", f"{mae:.4f}", delta=f"{comparison_mae - mae:.4f}", delta_color="normal")
                        col2.metric("MSE", f"{mse:.4f}", delta=f"{comparison_mse - mse:.4f}", delta_color="normal")
                        col3.metric("RMSE", f"{rmse:.4f}", delta=f"{comparison_rmse - rmse:.4f}", delta_color="normal")

                        st.subheader("Baseline (Budget) Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Baseline MAE", f"{comparison_mae:.4f}")
                        col2.metric("Baseline MSE", f"{comparison_mse:.4f}")
                        col3.metric("Baseline RMSE", f"{comparison_rmse:.4f}")

                    with st.expander("Feature Importances", expanded=True):
                        st.subheader("Feature Importances")
                        if hasattr(best_model, "feature_importances_"):
                            if model_type == "RandomForest":
                                feature_columns = list(X_train.columns)
                            importances = best_model.feature_importances_
                            importance_df = pd.DataFrame({
                                "Feature": feature_columns,
                                "Importance": importances
                            }).sort_values(by="Importance", ascending=False)

                            st.bar_chart(importance_df.set_index('Feature'))

                            st.table(importance_df.style.format({"Importance": "{:.4f}"})
                                     .bar(subset=["Importance"], color="#5fba7d", align="zero"))
                        else:
                            st.write("This model doesn't provide feature importances.")

                    st.markdown(
                        f"<p style='color:green; font-size:16px;'>✅ Best hyperparameters of {model_type} saved successfully</p>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='color:green; font-size:16px;'>✅ Best {model_type} model saved successfully</p>",
                        unsafe_allow_html=True)
                else:
                    with st.spinner("Running model..."):
                        if model_type == "Prophet":
                            try:
                                mae, mse, rmse = run_prophet_model(df)
                                st.success("Prophet model run successfully!")
                                st.subheader("Evaluation Metrics")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("MAE", f"{mae:.4f}")
                                col2.metric("MSE", f"{mse:.4f}")
                                col3.metric("RMSE", f"{rmse:.4f}")
                            except Exception as e:
                                st.error(f"Error running Prophet model: {e}")
                        elif model_type == "LSTM":
                            try:
                                mae, mse, rmse = run_lstm_model(df, n_trials=n_trials, num_layers=num_layers,
                                                                 activation=activation)
                                if mae is None or mse is None or rmse is None:
                                    st.error("No valid data groups to process.")
                                else:
                                    st.success("LSTM model run successfully!")
                                    st.subheader("Evaluation Metrics")
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("MAE", f"{mae:.4f}")
                                    col2.metric("MSE", f"{mse:.4f}")
                                    col3.metric("RMSE", f"{rmse:.4f}")
                            except Exception as e:
                                st.error(f"Error running LSTM model: {e}")


if __name__ == "__main__":
    forecasting_page()
