"""
Generic CSV data loader.

Dataset-agnostic: no knowledge of band naming, column semantics, or
specific file formats.  Just loads a CSV into a DataFrame.
"""

from __future__ import annotations

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    return pd.read_csv(path)
