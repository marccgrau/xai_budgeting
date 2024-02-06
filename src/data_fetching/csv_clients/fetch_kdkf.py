import logging
import re
from typing import Optional, Tuple

import pandas as pd

from src.data_fetching.csv_clients.sheet_processors import get_hrm_processor
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def determine_year_from_filename(file_path: str) -> str:
    year_matches = re.findall(r"\d{4}", file_path)
    if year_matches:
        return year_matches[0]
    else:
        return "XXXX"


def extract_hrm_region(sheet_name: str) -> Tuple[str, str]:
    # Regular expression to strictly match two-character regions followed by HRM patterns
    pattern = re.compile(r"(HRM\d)[_ ]?KT_([A-Z]{2})|([A-Z]{2})[_ ](HRM\d)")
    match = pattern.search(sheet_name)
    if match:
        # Extract HRM type and region from the matched groups
        hrm_type = match.group(1) if match.group(1) else match.group(4)
        region = match.group(2) if match.group(2) else match.group(3)
        return hrm_type, region
    else:
        raise ValueError(f"Invalid sheet name format: {sheet_name}")


def is_hrm2_adequately_populated(
    df: pd.DataFrame, key_columns: Optional[list] = None
) -> bool:
    if key_columns is None:
        key_columns = ["Realized", "Budget y"]  # Default column to check
    for column in key_columns:
        if df[column].sum() != 0:  # Check if there's any non-zero sum in key columns
            return True
    return False


def process_excel_file(
    file_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    xls = pd.ExcelFile(file_path)
    single_digit_master = pd.DataFrame()
    double_digit_master = pd.DataFrame()

    current_year = determine_year_from_filename(file_path)

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
            ):
                if hrm_type == "HRM2":
                    processor = get_hrm_processor(
                        hrm_type, file_path, sheet_name, region, current_year
                    )
                    (
                        temp_single_digit_df,
                        temp_double_digit_df,
                    ) = processor.process_sheet()
                    if is_hrm2_adequately_populated(
                        temp_single_digit_df
                    ) or is_hrm2_adequately_populated(temp_double_digit_df):
                        single_digit_master = pd.concat(
                            [single_digit_master, temp_single_digit_df],
                            ignore_index=True,
                        )
                        double_digit_master = pd.concat(
                            [double_digit_master, temp_double_digit_df],
                            ignore_index=True,
                        )
                        continue
                    else:
                        # Fallback to HRM1 if HRM2 is not adequately populated
                        logger.info(
                            f"HRM2 sheet for {region} not adequately populated. Falling back to HRM1."
                        )
                        hrm_type = "HRM1"  # Adjust hrm_type to HRM1 for processing
            processor = get_hrm_processor(
                hrm_type, file_path, sheet_name, region, current_year
            )
            single_digit_df, double_digit_df = processor.process_sheet()
            single_digit_master = pd.concat(
                [single_digit_master, single_digit_df], ignore_index=True
            )
            double_digit_master = pd.concat(
                [double_digit_master, double_digit_df], ignore_index=True
            )
        except ValueError as e:
            print(e)  # Handle invalid sheet names

    return single_digit_master, double_digit_master, current_year
