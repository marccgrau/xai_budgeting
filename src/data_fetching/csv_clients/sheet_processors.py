from typing import Tuple

import numpy as np
import pandas as pd


class SheetProcessor:
    def __init__(self, file_path: str, sheet_name: str, region: str, current_year: str):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.region = region
        if current_year is None:
            raise ValueError("current_year cannot be None")
        self.current_year = current_year

    def process_sheet(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class HRM1Processor(SheetProcessor):
    def convert_hrm1_to_hrm2(self):
        pass

    def process_sheet(self):
        # Logic for processing HRM1 sheets
        pass


class HRM2Processor2019Plus(SheetProcessor):
    def process_sheet(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Read the Excel sheet
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, header=4)
        df = df.drop(df.index[0])

        if self.current_year is None:
            raise ValueError("current_year cannot be None")

        # Rename columns based on year
        df = self.rename_columns_based_on_year(df, self.current_year)

        # Handle NaNs
        df = df.fillna("")

        # Filter rows containing 'HRM'
        df = df[df["Ref-ID"].str.contains("HRM")]
        df["Acc-ID"] = df["Acc-ID"].astype(str)

        # Keep only rows with numeric 'Acc-ID'
        df = df[df["Acc-ID"].str.isnumeric()]

        # Add region and year columns
        df["Region"] = self.region
        df["Year"] = self.current_year
        df["Slack"] = np.where(df["Budget y"] != 0, df["Budget y"] - df["Realized"], 0)

        # Separate into single and double digit DataFrames
        df_single_digit = df[df["Acc-ID"].str.len() == 1].copy()
        df_double_digit = df[df["Acc-ID"].str.len() == 2].copy()

        return (df_single_digit, df_double_digit)

    def rename_columns_based_on_year(
        self, df: pd.DataFrame, base_year: str
    ) -> pd.DataFrame:
        # Define the base renaming dictionary
        renaming_dict = {
            "Referenz-ID": "Ref-ID",
            "HRM 2": "Acc-ID",
            "in 1 000 Franken": "Name",
            "en 1 000 frs.": "Name",
            f"{base_year}": "Budget y",
            f"{base_year}.3": "Realized",
            str(int(base_year) + 1): "Budget y+1",
        }

        df.columns = df.columns.map(str)

        # Filter out columns not in the renaming dictionary
        df = df[df.columns.intersection(renaming_dict.keys())]

        # Rename the columns
        df = df.rename(columns=renaming_dict)

        return df


class HRM2Processor2018(SheetProcessor):
    def process_sheet(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Read the Excel sheet
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, header=4)
        df = df.drop(df.index[0])

        # Rename columns based on year
        df = self.rename_columns_based_on_year(df, self.current_year)

        # Handle NaNs
        df = df.fillna("")

        # Filter rows containing 'HRM'
        df = df[df["Ref-ID"].str.contains("HRM")]
        df["Acc-ID"] = df["Acc-ID"].astype(str)

        # Keep only rows with numeric 'Acc-ID'
        df = df[df["Acc-ID"].str.isnumeric()]

        # Add region and year columns
        df["Region"] = self.region
        df["Year"] = self.current_year
        df["Slack"] = np.where(df["Budget y"] != 0, df["Budget y"] - df["Realized"], 0)

        # Separate into single and double digit DataFrames
        df_single_digit = df[df["Acc-ID"].str.len() == 1].copy()
        df_double_digit = df[df["Acc-ID"].str.len() == 2].copy()

        return (df_single_digit, df_double_digit)

    def rename_columns_based_on_year(
        self, df: pd.DataFrame, base_year: str
    ) -> pd.DataFrame:
        # Define the base renaming dictionary
        renaming_dict = {
            "Referenz-ID": "Ref-ID",
            "HRM 2": "Acc-ID",
            "in 1 000 Franken": "Name",
            "en 1 000 frs.": "Name",
            f"{base_year}": "Budget y",
            f"{base_year}.3": "Realized",
            str(int(base_year) + 1): "Budget y+1",
        }

        df.columns = df.columns.map(str)

        # Filter out columns not in the renaming dictionary
        df = df[df.columns.intersection(renaming_dict.keys())]

        # Rename the columns
        df = df.rename(columns=renaming_dict)

        return df


def get_hrm_processor(
    hrm_type: str,
    file_path: str,
    sheet_name: str,
    region: str,
    current_year: str,
) -> SheetProcessor:
    if hrm_type == "HRM1":
        return HRM1Processor(file_path, sheet_name, region, current_year)
    elif hrm_type == "HRM2" and int(current_year) >= 2019:
        return HRM2Processor2019Plus(file_path, sheet_name, region, current_year)
    elif hrm_type == "HRM2" and int(current_year) <= 2018:
        return HRM2Processor2018(file_path, sheet_name, region, current_year)
    else:
        raise ValueError(f"Unknown HRM type: {hrm_type}")
