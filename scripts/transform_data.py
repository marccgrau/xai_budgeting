import logging
import os

from src.logging_config import setup_logging
from src.transform_data import transform_data

setup_logging()
logger = logging.getLogger(__name__)


def main():
    directory = "data/merged"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            logger.info(f"Processing file: {filename}")
            df = transform_data(file_path)
            df.to_csv(f"data/final/{filename}", index=False)


if __name__ == "__main__":
    main()
