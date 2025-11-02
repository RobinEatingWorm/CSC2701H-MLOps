import chardet
import hashlib
import numpy as np
import pandas as pd


def get_encoding(filepath: str) -> str:
    """
    Detect the encoding of a file.

    Parameters
    ----------
    filepath : str
        The path to the file.

    Returns
    -------
    encoding : str
        The detected file encoding.
    """

    # Open the file and detect its encoding
    with open(filepath, "rb") as file:
        encoding = chardet.detect(file.read())["encoding"]
    return encoding


def hash_str(ser: pd.Series) -> pd.Series:
    """
    Hash a Series of strings.

    Parameters
    ----------
    ser : pd.Series
        A Series of strings.

    Returns
    -------
    ser : pd.Series
        A Series of hashes of the original strings.
    """

    # String hashing
    return ser.apply(lambda s: hashlib.sha512(s.encode()).hexdigest())


def str_to_float(ser: pd.Series) -> pd.Series:
    """
    Convert a Series of strings to floats. Strings containing no numerical values are converted to NaNs instead.

    Parameters
    ----------
    ser : pd.Series
        A Series of strings.

    Returns
    -------
    ser : pd.Series
        The Series with strings converted to floats or NaNs if not possible.
    """

    # String to float conversion
    return ser.str.replace(r"[^0-9.]", "", regex=True).replace("", np.nan).astype(np.float64)
