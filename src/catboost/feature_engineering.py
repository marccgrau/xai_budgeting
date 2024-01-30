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

    # Check if necessary columns are present
    required_columns = ["Year", "Realized"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Sort the data by 'Year' to ensure correct lag feature calculation
    df = df.sort_values(by="Year")

    # Define the lag periods
    one_year_lag = 1
    two_year_lag = 2
    five_year_lag = 5

    # Create lag features for 'Realized' depending on the region and account ID
    df["Realized_1yr_lag"] = df.groupby(["Region", "Acc-ID"])["Realized"].shift(
        one_year_lag
    )
    df["Realized_2yr_lag"] = df.groupby(["Region", "Acc-ID"])["Realized"].shift(
        two_year_lag
    )

    # Create a 5-year lag rolling mean feature for 'Realized' without including the current year's value
    df["Realized_5yr_lag"] = (
        df.groupby(["Region", "Acc-ID"])["Realized"]
        .shift(one_year_lag)
        .rolling(five_year_lag)
        .mean()
    )

    # Drop rows with NaN values resulting from the lag operations
    df = df.dropna()

    return df
