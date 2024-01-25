import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a DataFrame.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        return pd.read_csv(file_path, index_col=None, header=0)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()  # Returning an empty DataFrame in case of error


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in specific columns of the DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    df["Ref-ID"] = df["Ref-ID"].fillna("HRM1")
    df["Name"] = df["Name"].fillna("Missing Name")
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data types of specific columns in the DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The DataFrame with converted data types.
    """
    df["Acc-ID"] = df["Acc-ID"].astype("category")
    df["Region"] = df["Region"].astype("category")
    df["Year"] = pd.to_datetime(df["Year"], format="%Y").dt.year
    return df


def transform_data(file_path: str) -> pd.DataFrame:
    """
    Process a single file: load data, fill missing values, and convert dtypes.

    Args:
    file_path (str): Path to the file to be processed.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    df = load_data(file_path)
    df = fill_missing_values(df)
    df = convert_dtypes(df)
    df = df[["Year", "Region", "Acc-ID", "Realized", "Budget y", "Budget y+1", "Slack"]]
    return df
