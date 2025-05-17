from pathlib import Path
import numpy as np
import pandas as pd

def load_timeseries(ts_file: Path) -> np.ndarray:
    """
    Load unstacked time series from a .npz file.

    Parameters
    ----------
    ts_file : Path
        Path to the .npz file containing 'ts'.

    Returns
    -------
    np.ndarray
        Array of time series data.
    """
    data = np.load(ts_file, allow_pickle=True)
    return data['ts']

def load_cognitive_data(cog_file: Path) -> pd.DataFrame:
    """
    Load cognitive metadata from a CSV file.

    Parameters
    ----------
    cog_file : Path
        Path to the CSV file with cognitive data.

    Returns
    -------
    pd.DataFrame
        Cognitive metadata.
    """
    return pd.read_csv(cog_file)

def validate_alignment(ts_data: np.ndarray, cog_data: pd.DataFrame):
    """
    Ensure time series and cognitive data are aligned.

    Raises
    ------
    AssertionError
        If the lengths do not match.
    """
    assert len(ts_data) == len(cog_data), "Mismatch between time series and cognitive data entries."
