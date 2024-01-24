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
    def __init__(self, file_path: str, sheet_name: str, region: str, current_year: str):
        super().__init__(file_path, sheet_name, region, current_year)
        self.convert_dict = {
            "1": (["1"], [1.0]),
            "2": (["2"], [1.0]),
            "3": (["3"], [1.0]),
            "4": (["4"], [1.0]),
            "5": (["5"], [1.0]),
            "6": (["6"], [1.0]),
            "30": (["30"], [1.0]),
            "31": (["31"], [1.0]),
            "33": (["33"], [1.0]),
            "38": (["35"], [1.0]),
            "34 - 37": (["34", "35", "36", "37"], [0.25, 0.25, 0.25, 0.25]),
            "39": (["39"], [1.0]),
            "40": (["40"], [1.0]),
            "41 / 43": (["41", "43"], [0.5, 0.5]),
            "48": (["45"], [1.0]),
            "44 - 47": (["44", "45", "46", "47"], [0.25, 0.25, 0.25, 0.25]),
            "49": (["49"], [1.0]),
            "32": (["34"], [1.0]),
            "50": (["50"], [1.0]),
            "52": (["54"], [1.0]),
            "56 - 58": (["56", "57", "58"], [1 / 3, 1 / 3, 1 / 3]),
            "60 - 61": (["60", "61"], [0.5, 0.5]),
            "62 - 67": (
                ["62", "63", "64", "65", "66", "67"],
                [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            ),
        }

    def convert_hrm1_to_hrm2(self, df: pd.DataFrame) -> pd.DataFrame:
        new_rows = []

        for index, row in df.iterrows():
            old_acc_id = row["Acc-ID"]

            # Check if the Acc-ID is a range and needs splitting
            if old_acc_id in self.convert_dict:
                new_acc_ids, ratios = self.convert_dict[old_acc_id]

                for new_acc_id, ratio in zip(new_acc_ids, ratios):
                    new_row = row.copy()
                    new_row["Acc-ID"] = new_acc_id
                    new_row["Budget y"] *= ratio
                    new_row["Realized"] *= ratio
                    new_row["Budget y+1"] *= ratio
                    new_rows.append(new_row.to_dict())
            else:
                # Handle cases where Acc-ID is not in the mapping
                new_rows.append(row)

        return pd.DataFrame(new_rows, index=None)

    def process_sheet(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, header=1)
        df = df.drop(df.index[0])

        df = self.rename_columns_based_on_year(df, self.current_year)

        df = self.convert_hrm1_to_hrm2(df)

        df["Acc-ID"] = df["Acc-ID"].astype(str)

        # Keep only rows with numeric 'Acc-ID'
        df = df[df["Acc-ID"].str.isnumeric()]
        df = df[df["Acc-ID"] != "0"]

        # Add region and year columns
        df["Region"] = self.region
        df["Year"] = self.current_year
        df["Slack"] = np.where(df["Budget y"] != 0, df["Budget y"] - df["Realized"], 0)

        # Separate into single and double digit DataFrames
        df_single_digit = df[df["Acc-ID"].str.len() == 1].copy()
        df_double_digit = df[df["Acc-ID"].str.len() == 2].copy()

        return (df_single_digit, df_double_digit)

    @staticmethod
    def rename_columns_based_on_year(df: pd.DataFrame, base_year: str) -> pd.DataFrame:
        # Define the base renaming dictionary
        renaming_dict = {
            "0": "Acc-ID",
            "0.1": "Name",
            f"{base_year}": "Budget y",
            f"{base_year}.1": "Realized",
            str(int(base_year) + 1): "Budget y+1",
        }

        df.columns = df.columns.map(str)

        # Filter out columns not in the renaming dictionary
        df = df[df.columns.intersection(renaming_dict.keys())]

        # Rename the columns
        df = df.rename(columns=renaming_dict)

        return df


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

    @staticmethod
    def rename_columns_based_on_year(df: pd.DataFrame, base_year: str) -> pd.DataFrame:
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
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, header=1)
        df = df.drop(df.index[0])

        # Rename columns based on year
        df = self.rename_columns_based_on_year(df, self.current_year)

        # Filter rows containing characters
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

    @staticmethod
    def rename_columns_based_on_year(df: pd.DataFrame, base_year: str) -> pd.DataFrame:
        # Define the base renaming dictionary
        renaming_dict = {
            "Unnamed: 0": "Acc-ID",
            "in 1000 Franken": "Name",
            "en 1000 frs.": "Name",
            f"{base_year}": "Budget y",
            f"{base_year}.1": "Realized",
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
