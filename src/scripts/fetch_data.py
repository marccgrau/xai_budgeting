import json
import logging
import os

import hydra
from omegaconf import DictConfig

from src.data_fetching.api_clients.cantons.sg_client import SGAPI
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="api_config", version_base=None)
def main(cfg: DictConfig):
    # Create a data folder if it doesn't exist
    data_folder = "data"
    raw_data_folder = os.path.join(data_folder, "raw")
    sg_folder = os.path.join(raw_data_folder, "sg")
    if not os.path.exists(sg_folder):
        os.makedirs(sg_folder)

    try:
        api = SGAPI(cfg.api_urls.sg)
        logger.info("Fetching catalog datasets for canton SG")

        datasets = api.query_dataset_records(
            dataset_id="budget-2022-kanton-stgallen-institutionelle-gliederung"
        )

        # Store the raw data in a JSON file within the SG subfolder
        raw_data_file = os.path.join(sg_folder, "raw_data.json")
        with open(raw_data_file, "w") as f:
            json.dump(datasets, f)

        logger.info(f"Found {len(datasets)} datasets for canton SG")
    except Exception as e:
        logger.error(f"Error fetching catalog datasets for canton SG: {str(e)}")


if __name__ == "__main__":
    main()
