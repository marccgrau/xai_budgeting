import pandas as pd
import os


def process_file(year, input_dir, output_dir):
    extension = '.xls' if year <= 2017 else '.xlsx'
    df = pd.read_excel(os.path.join(input_dir, f"Kantonsdaten_{year}_raw{extension}"))

    if year != 2013:
        df = df.T
        df.columns = df.iloc[0]
        df = df.iloc[1:]

    df.insert(0, 'Year', year)

    if year != 2013:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Canton'}, inplace=True)

    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name != 'Canton' else x).fillna(0)

    df.to_csv(os.path.join(output_dir, f"Kantonsdaten_{year}_processed.csv"), index=False)


input_directory = "../data/raw_canton"
output_directory = "../data/processed_canton"
output_directory_merged = "../data/merged_canton"

years = range(2013, 2022)

for year in years:
    process_file(year, input_directory, output_directory)

dataframes = []
for year in years:
    processed_file_path = os.path.join(output_directory, f"Kantonsdaten_{year}_processed.csv")
    df = pd.read_csv(processed_file_path)
    dataframes.append(df)

final_df = pd.concat(dataframes, axis=0, ignore_index=True)
final_df.fillna(0, inplace=True)

final_df.to_csv(os.path.join(output_directory_merged, "Kantonsdaten_2013_to_2021_merged.csv"), index=False)
