import logging
import re
from typing import Optional, Tuple

import pandas as pd

from src.data_fetching.csv_clients.sheet_processors import get_hrm_processor
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def extract_hrm_region(sheet_name: str) -> Tuple[str, str]:
    # Regular expression to match various sheet naming conventions
    pattern = re.compile(r"(HRM\d)[_ ]?(KT_)?([A-Z]{2})|([A-Z]{2})[_ ](HRM\d)")
    match = pattern.search(sheet_name)
    if match:
        # Extract HRM type and region from the matched groups
        hrm_type = match.group(1) if match.group(1) else match.group(5)
        region = match.group(3) if match.group(3) else match.group(4)
        return hrm_type, region
    else:
        raise ValueError(f"Invalid sheet name format: {sheet_name}")


def process_excel_file(
    file_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    xls = pd.ExcelFile(file_path)
    single_digit_master = pd.DataFrame()
    double_digit_master = pd.DataFrame()
    current_year = None

    # Step 1: Determine which regions have both HRM1 and HRM2
    region_hrm_presence = {}
    for sheet_name in xls.sheet_names:
        try:
            hrm_type, region = extract_hrm_region(sheet_name)
            if region not in region_hrm_presence:
                region_hrm_presence[region] = [hrm_type]
            else:
                if hrm_type not in region_hrm_presence[region]:
                    region_hrm_presence[region].append(hrm_type)
        except ValueError as e:
            print(e)  # Handle invalid sheet names

    logger.info(f"Region HRM presence: {region_hrm_presence}")
    # Step 2 & 3: Process sheets based on HRM type presence
    for sheet_name in xls.sheet_names:
        if "HRM" not in sheet_name or "alle" in sheet_name.lower():
            logger.info(f"Skipping sheet: {sheet_name}")
            continue
        try:
            logger.info(f"Processing sheet: {sheet_name}")
            hrm_type, region = extract_hrm_region(sheet_name)
            # Process only HRM2 if both HRM1 and HRM2 are present for a region
            if (
                "HRM1" in region_hrm_presence[region]
                and "HRM2" in region_hrm_presence[region]
                and hrm_type == "HRM1"
            ):
                continue  # Skip HRM1 if both types are present
            processor = get_hrm_processor(hrm_type, file_path, sheet_name, region)
            single_digit_df, double_digit_df, current_year = processor.process_sheet()
            single_digit_master = pd.concat(
                [single_digit_master, single_digit_df], ignore_index=True
            )
            double_digit_master = pd.concat(
                [double_digit_master, double_digit_df], ignore_index=True
            )
        except ValueError as e:
            print(e)  # Handle invalid sheet names

    return single_digit_master, double_digit_master, current_year
