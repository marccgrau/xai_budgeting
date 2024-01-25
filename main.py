import logging
import os

from src.data_fetching.csv_clients.fetch_kdkf import process_excel_file
from src.logging_config import setup_logging
from src.process_data import merge_double_digit_files, merge_single_digit_files

setup_logging()
logger = logging.getLogger(__name__)


def main():
    directory = "data/raw"

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Check if the file is an Excel file
        if filename.endswith(".xlsx"):
            logger.info(f"Processing file: {filename}")
            single_digit_df, double_digit_df, year = process_excel_file(file_path)

            # Save the processed data to CSV files
            single_digit_df.to_csv(
                f"data/processed/kdkf_{year}_single_digit.csv", index=False
            )
            double_digit_df.to_csv(
                f"data/processed/kdkf_{year}_double_digit.csv", index=False
            )
        else:
            logger.info(f"Skipping non-Excel file: {filename}")

    # Merge the processed files
    merge_single_digit_files()
    merge_double_digit_files()


if __name__ == "__main__":
    main()
