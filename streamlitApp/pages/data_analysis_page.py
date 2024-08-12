import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
from pathlib import Path
import pygwalker as pyg
import streamlit.components.v1 as components
from utils.feature_engineering import apply_feature_engineering, drop_all_zero_entries, choose_acc_ids


def load_config(config_path='config/acc_config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def list_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]


def data_analysis_page():
    st.title("Data Analysis")
    st.write("Upload a dataset or choose from existing datasets for analysis.")

    config_path = Path("config/acc_config.yaml")
    acc_config = load_config(config_path)
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
        df = apply_feature_engineering(df)
        df = drop_all_zero_entries(df)

        st.subheader("Dataset Overview")
        st.write("Processed and filtered data:")
        st.write(df)

        st.subheader("Interactive Data Exploration")
        pyg_html = pyg.to_html(df)
        components.html(pyg_html, height=1000, scrolling=True)

        category_choice = st.selectbox("Select category", list(acc_config.keys()))
        df = choose_acc_ids(df, acc_config.get(category_choice))

        df['Year'] = df['Year'].astype(int)
        df['Realized'] = pd.to_numeric(df['Realized'], errors='coerce') / 1000000
        df['Budget y'] = pd.to_numeric(df['Budget y'], errors='coerce') / 1000000

        aggregated_data = df.groupby(['Year', 'Region'])[['Realized', 'Budget y']].sum().reset_index()
        aggregated_data['Percentage Difference'] = ((aggregated_data['Budget y'] - aggregated_data['Realized']) /
                                                    aggregated_data['Realized']).replace([np.inf, -np.inf],
                                                                                         np.nan) * 100

        region_options = ['All Regions'] + list(aggregated_data['Region'].unique())
        region_choice = st.selectbox("Select region", region_options)

        st.subheader(f"Percentage Difference (Budget y to Realized)")
        fig, ax = plt.subplots(figsize=(10, 6))  # Increased figure size for better visibility

        if region_choice == 'All Regions':
            for region in aggregated_data['Region'].unique():
                region_data = aggregated_data[aggregated_data['Region'] == region]
                sns.lineplot(data=region_data, x='Year', y='Percentage Difference', label=region, ax=ax)
        else:
            canton_data = aggregated_data[aggregated_data['Region'] == region_choice]
            sns.lineplot(data=canton_data, x='Year', y='Percentage Difference', label=region_choice, ax=ax,
                         color='purple', linewidth=2)
            for region in aggregated_data['Region'].unique():
                if region != region_choice:
                    region_data = aggregated_data[aggregated_data['Region'] == region]
                    sns.lineplot(data=region_data, x='Year', y='Percentage Difference', ax=ax, color='gray', alpha=0.3,
                                 linewidth=0.8, label="_nolegend_")

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Percentage Difference', fontsize=12)
        ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.legend(title='Region', fontsize=10, title_fontsize='11', loc='best', bbox_to_anchor=(1.05, 1),
                  borderaxespad=0.)
        ax.tick_params(axis='y', labelsize=10)
        years = aggregated_data['Year'].unique()
        ax.set_xticks(years)
        ax.set_xticklabels([int(year) for year in years], fontsize=10, rotation=45)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Heatmap of Percentage Differences")
        df_pivot = aggregated_data.pivot(index='Region', columns='Year', values='Percentage Difference')

        fig, ax = plt.subplots(figsize=(12, 10))
        heatmap = sns.heatmap(df_pivot, annot=True, fmt=".1f", cmap='coolwarm', linewidths=.5, center=0,
                              cbar_kws={'label': 'Percentage Difference'})

        ax.set_title(f'Percent Deviation by Region and Year for Category {category_choice}', fontsize=18)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Region', fontsize=14)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.yaxis.label.set_color('black')
        cbar.ax.tick_params(labelsize=12, colors='black')

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Custom Plot by Region, ACC ID, and Variable")

        custom_region_choice = st.selectbox("Select region for custom plot", region_options)
        y_axis_columns = [col for col in df.columns if col not in ['Year', 'Region', 'Acc-ID']]
        y_axis_choice = st.selectbox("Select y-axis variable", y_axis_columns)

        if y_axis_choice in ['Realized', 'Budget y']:
            acc_id_options = df['Acc-ID'].unique()
            acc_id_choice = st.selectbox("Select ACC ID", acc_id_options)
            filtered_df = df[df['Acc-ID'] == acc_id_choice]
        else:
            filtered_df = df

        fig, ax = plt.subplots(figsize=(10, 6))
        if custom_region_choice == 'All Regions':
            sns.lineplot(data=filtered_df, x='Year', y=y_axis_choice, hue='Region', ax=ax)
        else:
            custom_region_data = filtered_df[filtered_df['Region'] == custom_region_choice]
            sns.lineplot(data=custom_region_data, x='Year', y=y_axis_choice, label=custom_region_choice, ax=ax)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(y_axis_choice, fontsize=12)
        if custom_region_choice == 'All Regions':
            ax.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend(title=None, loc='upper left')
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xticks(years)
        ax.set_xticklabels([int(year) for year in years], fontsize=10, rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    data_analysis_page()
