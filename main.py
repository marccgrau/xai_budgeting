import logging

from src.data_fetching.csv_clients.fetch_kdkf import process_excel_file
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    # Define the file path
    file_path = "data/raw/kdkf_2022_raw.xlsx"
    single_digit_df, double_digit_df, year = process_excel_file(file_path)
    single_digit_df.to_csv(f"data/processed/kdkf_{year}_single_digit.csv", index=False)
    double_digit_df.to_csv(f"data/processed/kdkf_{year}_double_digit.csv", index=False)


if __name__ == "__main__":
    main()
