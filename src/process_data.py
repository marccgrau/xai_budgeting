import logging
import os

import pandas as pd

from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def merge_files_by_pattern(directory: str, pattern: str, output_file: str) -> None:
    """
    Merges CSV files from a directory that match a given pattern into a single DataFrame,
    and saves it to a specified output file.

    Args:
    directory (str): The directory to search for files.
    pattern (str): The pattern to match files.
    output_file (str): The path to the output file.
    """
    merged_df = pd.DataFrame()

    for filename in os.listdir(directory):
        if pattern in filename:
            file_path = os.path.join(directory, filename)
            logger.info(f"Loading file: {filename}")
            try:
                df = pd.read_csv(file_path)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            except pd.errors.EmptyDataError:
                logger.info(f"Warning: {filename} is empty and will be skipped.")
            except Exception as e:
                logger.info(f"Error reading {filename}: {e}")

    # Save the merged DataFrame
    try:
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Merged file saved as: {output_file}")
    except Exception as e:
        logger.info(f"Error saving merged file: {e}")


def merge_double_digit_files() -> None:
    """
    Merges all double digit CSV files into a single DataFrame and saves it.
    """
    directory = "data/processed"
    output_file = "data/merged/merged_double_digit.csv"
    merge_files_by_pattern(directory, "_double_digit.csv", output_file)


def merge_single_digit_files() -> None:
    """
    Merges all single digit CSV files into a single DataFrame and saves it.
    """
    directory = "data/processed"
    output_file = "data/merged/merged_single_digit.csv"
    merge_files_by_pattern(directory, "_single_digit.csv", output_file)


if __name__ == "__main__":
    merge_double_digit_files()
    merge_single_digit_files()
