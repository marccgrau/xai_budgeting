from typing import List, Optional

import pandas as pd


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with at least the columns 'Year' and 'Realized'.

    Returns:
    pandas.DataFrame: The DataFrame with added lag features.

    Raises:
    ValueError: If required columns are not in the DataFrame.
    """
    df_mod = df.copy()
    # Check if necessary columns are present
    required_columns = ["Year", "Realized"]
    missing_columns = [col for col in required_columns if col not in df_mod.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Define the lag periods
    one_year_lag = 1
    two_year_lag = 2

    # Create lag features for 'Realized' depending on the region and account ID
    df_mod["Realized_1yr_lag"] = df_mod.groupby(["Region", "Acc-ID"])["Realized"].shift(
        one_year_lag
    )
    df_mod["Realized_2yr_lag"] = df_mod.groupby(["Region", "Acc-ID"])["Realized"].shift(
        two_year_lag
    )

    return df_mod


def drop_all_zero_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows from the DataFrame where:
    - Both 'Realized' and 'Budget y' are zero, or
    - 'Budget y' is zero for specific Account IDs (10 and 20).

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with specified rows removed.
    """
    # Drop rows where both 'Realized' and 'Budget y' are zero
    condition1 = (df["Realized"] != 0) | (df["Budget y"] != 0)

    # Drop rows where 'Budget y' is zero for specific Account IDs
    condition2 = ~((df["Budget y"] == 0) & (df["Acc-ID"].isin([10, 14, 20, 29])))

    # Combine conditions using bitwise '&' operator for AND operation
    filtered_df = df[condition1 & condition2]

    return filtered_df


def choose_acc_ids(df: pd.DataFrame, acc_ids: Optional[List[int]]) -> pd.DataFrame:
    """
    Choose specific Account IDs from the DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    acc_ids (list): The list of Account IDs to choose.

    Returns:
    pandas.DataFrame: The DataFrame with specified Account IDs.
    """
    df_mod = df.copy()
    if acc_ids is None:
        return df_mod
    return df_mod[df_mod["Acc-ID"].isin(acc_ids)]


def engineer_df(df: pd.DataFrame, acc_ids: Optional[List[int]]) -> pd.DataFrame:
    """
    Apply feature engineering to the DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    acc_ids (list): The list of Account IDs to choose.

    Returns:
    pandas.DataFrame: The DataFrame with added lag features and specified Account IDs.
    """
    df_mod = df.copy()
    # df_mod["Year"] = df_mod["Year"] - df_mod["Year"].min() #Comment this line when using Prophet
    df_mod = df_mod.sort_values(by="Year")
    df_mod = apply_feature_engineering(df_mod)
    df_mod = drop_all_zero_entries(df_mod)
    df_mod = choose_acc_ids(df_mod, acc_ids)
    return df_mod
