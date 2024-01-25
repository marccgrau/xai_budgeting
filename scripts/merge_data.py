import logging

from src.logging_config import setup_logging
from src.process_data import merge_double_digit_files, merge_single_digit_files

setup_logging()
logger = logging.getLogger(__name__)


def main():
    # Merge the processed files
    merge_single_digit_files()
    merge_double_digit_files()


if __name__ == "__main__":
    main()
