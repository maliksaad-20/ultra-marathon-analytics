import os
import re
import numpy as np
import pandas as pd

import logging
from pathlib import Path
import sys

project_root = Path().resolve().parent
log_file = project_root / "logs" / "app.log"
log_file.parent.mkdir(exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)



class BaseDataProcessor:

    """
    Base class for data loading and processing.

    Provides common functionality such as:
    - Loading CSV files
    - Removing duplicates
    - Logging support
    - Shared preprocessing utilities

    This class is meant to be inherited by domain-specific processors.
    """

    def __init__(self, file_path=None):
        self.logger = logger
        self.data = None
        if file_path:
            self.data = self.load_data(file_path)

    def load_data(self, file_path):
        """
        Load the ultra marathon race data from a CSV file.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            self.data = pd.read_csv(file_path, low_memory=False)
            self.logger.info(f"Data loaded successfully from {file_path}")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def standardize_column_names(self):
        """
        Standardize column names by converting to lowercase and replacing spaces with underscores.
        """

        if self.data is None:
            self.logger.error("No data loaded. Cannot standardize columns.")
            raise ValueError("Data is None. Load data first.")

        self.data.columns = (
            self.data.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )

        self.logger.info("Column names standardized")

    def basic_info(self):
        """ 
        Get basic information about the dataset, including shape, missing values, and memory usage.
        """
        dataset_info = {
            'num_rows': f"{self.data.shape[0]:,}",
            'num_cols': f"{self.data.shape[1]:,}",
            'total_cells': f"{self.data.size:,}",
            'missing_cells': f"{self.data.isna().sum().sum():,}",
            'missing_pct': f"{(self.data.isna().mean().mean() * 100):.2f}%",
            'memory_usage': f"{(self.data.memory_usage(deep=True).sum() / (1024 ** 2)):.2f} MB"
        }

        return dataset_info
    
    def dataset_details(self):
        """
        Perform a data audit to understand the structure and quality of the dataset.
        """
        if self.data is None:
            self.logger.error("No data loaded. Cannot audit.")
            raise ValueError("Data is None. Load data first.")

        audit = pd.DataFrame({
            'dtype'         : self.data.dtypes,
            'non_null'      : self.data.notna().sum(),
            'null_count'    : self.data.isna().sum(),
            'null_pct'      : (self.data.isna().mean() * 100).round(2),
            'unique'        : self.data.nunique(),
            'sample_records': self.data.apply(lambda col: col.dropna().unique()[:5].tolist())
        })

        audit = audit.reset_index().rename(columns={'index': 'column_name'})
        return audit

    def remove_duplicates(self):
        """
        Remove duplicate rows from the DataFrame.
        """
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        final_count = len(self.data)
        self.logger.info(f"Removed {initial_count - final_count} duplicate rows")

    def rename_columns(self, rename_dict):
        """Renames columns in the DataFrame based on a provided dictionary."""
        # Only rename columns that exist
        rename_dict_cleaned = {k: v for k, v in rename_dict.items() if k in self.data.columns}
        if not rename_dict_cleaned:
            self.logger.error("No columns to rename based on the provided dictionary.")
            raise ValueError("No valid columns to rename. Check the keys in the rename dictionary.")
        if len(rename_dict) != len(rename_dict_cleaned):
            missing_cols = set(rename_dict.keys()) - set(rename_dict_cleaned.keys())
            self.logger.warning(f"Some columns in the rename dictionary were not found in the DataFrame and will be skipped: {', '.join(missing_cols)}")
        self.data.rename(columns=rename_dict_cleaned, inplace=True)
        self.logger.info(f"Renamed columns: {', '.join(rename_dict_cleaned.keys())} to {', '.join(rename_dict_cleaned.values())}")

if __name__ == "__main__":
    print("This file is being run directly")