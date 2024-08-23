import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)

def combine_predictions():
    # Load RNN results
    df = pd.read_csv('results/rnn_results.csv')
    df = df[['Year', 'Region', 'Acc-ID', 'y', 'Budget y', 'yhat']]
    df = df.rename(columns={'yhat': 'yhat_rnn'})

    # Load CatBoost results
    dfcat = pd.read_csv('results/catboost_results.csv')
    dfcat = dfcat[['Realized', 'Predicted', 'Budget y']]
    dfcat = dfcat.rename(columns={'Predicted': 'yhat_cat'})
    df = pd.merge(df, dfcat, how='inner', left_on=['y', 'Budget y'], right_on=['Realized', 'Budget y'])
    df = df.drop(columns='Realized')

    # Load XGBoost results
    dfXg = pd.read_csv('results/xgboost_results.csv')
    dfXg = dfXg[['Realized', 'Predicted', 'Budget y']]
    dfXg = dfXg.rename(columns={'Predicted': 'yhat_xg'})
    df = pd.merge(df, dfXg, how='inner', left_on=['y', 'Budget y'], right_on=['Realized', 'Budget y'])
    df = df.drop(columns='Realized')

    # Load RandomForest results
    dfRandom = pd.read_csv('results/randomforest_results.csv')
    dfRandom = dfRandom[['Realized', 'Predicted', 'Budget y']]
    dfRandom = dfRandom.rename(columns={'Predicted': 'yhat_rf'})
    df = pd.merge(df, dfRandom, how='inner', left_on=['y', 'Budget y'], right_on=['Realized', 'Budget y'])
    df = df.drop(columns='Realized')

    # Load Prophet results
    dfProph = pd.read_csv('results/prophet_results.csv')
    dfProph = dfProph[['Region', 'Acc-ID', 'yhat']]
    dfProph = dfProph.rename(columns={'yhat': 'yhat_proph'})
    df = pd.merge(df, dfProph, how='inner', on=['Region', 'Acc-ID'])

    # Drop duplicates
    df = df.drop_duplicates()
    return df

# Function to load historical data
def load_historical_data(file_path):
    df_hist = pd.read_csv(file_path)
    df_hist = df_hist[['Year', 'Region', 'Acc-ID', 'Realized', 'Budget y']]
    df_hist = df_hist.rename(columns={'Realized': 'y'})
    return df_hist

# Function to plot forecasts
def plot_forecasts(region, acc_id, df, df_hist):
    # Filter the dataframe based on selected region and Acc-ID
    filtered_df = df[(df['Region'] == region) & (df['Acc-ID'] == acc_id)]
    filtered_hist = df_hist[(df_hist['Region'] == region) & (df_hist['Acc-ID'] == acc_id)]

    if filtered_df.empty and filtered_hist.empty:
        st.write("No data available for the selected Region and Acc-ID.")
        return

    # Sort the filtered dataframes by Year
    filtered_df = filtered_df.sort_values(by='Year')
    filtered_hist = filtered_hist.sort_values(by='Year')

    # Plot the actual budget values, historical data, and predictions
    plt.figure(figsize=(12, 6))
    if not filtered_hist.empty:
        plt.plot(filtered_hist['Year'], filtered_hist['Budget y'], label='Historical Budget y', marker='o')
        plt.plot(filtered_hist['Year'], filtered_hist['y'], label='Historical Realized', marker='x')
    if not filtered_df.empty:
        plt.plot(filtered_df['Year'], filtered_df['Budget y'], label='Actual Budget y', marker='o')
        plt.plot(filtered_df['Year'], filtered_df['yhat_rnn'], label='RNN Prediction', marker='x')
        plt.plot(filtered_df['Year'], filtered_df['yhat_cat'], label='CatBoost Prediction', marker='x')
        plt.plot(filtered_df['Year'], filtered_df['yhat_xg'], label='XGBoost Prediction', marker='x')
        plt.plot(filtered_df['Year'], filtered_df['yhat_rf'], label='RandomForest Prediction', marker='x')
        plt.plot(filtered_df['Year'], filtered_df['yhat_proph'], label='Prophet Prediction', marker='x')

    plt.xlabel('Year')
    plt.ylabel('Budget Value')
    plt.title(f'Budget and Predictions for Region: {region}, Acc-ID: {acc_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def results_page():
    st.title("Results Page")
    st.write("Select Region and Acc-ID to see the predictions of different models.")

    # Combine predictions from different models
    df = combine_predictions()
    df_hist = load_historical_data('data/merged_double_digit.csv')

    # Create dropdowns for selecting region and Acc-ID
    regions = df['Region'].unique()
    acc_ids = df['Acc-ID'].unique()

    selected_region = st.selectbox("Select Region", regions)
    selected_acc_id = st.selectbox("Select Acc-ID", acc_ids)

    # Plot the forecasts
    if st.button("Show Predictions"):
        plot_forecasts(selected_region, selected_acc_id, df, df_hist)

if __name__ == "__main__":
    results_page()
