
"""Data loading utilities."""

import pandas as pd


def load_data(file_path):
    """Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded CSV data.
    """
    data = pd.read_csv(file_path)
    return data