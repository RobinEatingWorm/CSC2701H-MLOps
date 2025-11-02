import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from src.preprocessing import get_encoding, hash_str, str_to_float


def get_column_mappings() -> dict[str, str]:
    """
    Return a mapping dictionary for standardizing column names of all salary disclosures.

    Returns
    -------
    mappings : dict[str, str]
        A dictionary that maps from possible column names to a standard column name.
    """

    # Mappings from standard names to all variations that appear in data
    reverse_mappings = {
        "Sector": ["Sector"],
        "Last Name": ["Last Name", "Last name", "Surname"],
        "First Name": ["First Name", "First name"],
        "Salary": ["Salary", "Salary Paid", "Salary Paid ", "Salary paid"],
        "Benefits": ["Benefits", "Taxable Benefits", "Taxable benefits"],
        "Employer": ["Employer"],
        "Job Title": ["Job Title", "Job title", "JobTitle", "Position"],
        "Year": ["Calendar Year", "Calendar year", "Year"]
    }

    # Mappings from all variations that appear in data to standard names
    return {variant: default for default, variants in reverse_mappings.items() for variant in variants}


def preprocess_salary_disclosure(year: int) -> pd.DataFrame:
    """
    Perform preprocessing steps on a salary disclosure CSV.

    Parameters
    ----------
    year : int
        The year of the salary disclosure.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing preprocessed data for the specified year's salary disclosure.

    Notes
    -----
    Several years have unique missing or problematic values that are handled by preprocessing.

    - In 1996, line 1716 of the CSV contains an employee whose first name is `NA`.
    - In 1997, line 965 of the CSV contains an employee whose job title is missing.
    - In 1998, line 597 of the CSV contains an employee whose first name is `NA`.
    - In 2013, line 4617 of the CSV contains an employee whose first name is `NA`.
    - In 2015, there are 6980 lines of the CSV where taxable benefits is `$-`.
    - In 2016, line 84272 of the CSV contains an employee whose taxable benefits is missing.
    - Also in 2016, there are 65 lines of the CSV where calendar year is missing.

    First names of `NA` are kept in the data by overriding the default behaviour of `pd.read_csv`. Missing job titles
    and taxable benefits are left as NA/NaN. Taxable benefits of `$-` are considered to be missing and converted to
    NA/NaN by `str_to_float`. Missing calendar years are inserted back into the data.
    """

    # Read the data
    filepath = f"data/raw/{year}.csv"
    df = pd.read_csv(filepath, na_values=[""], keep_default_na=False, encoding=get_encoding(filepath))

    # Standardize column names
    df = df.rename(columns=get_column_mappings())

    # Hash employee names
    df["ID"] = hash_str(df["First Name"] + " " + df["Last Name"])
    df = df.drop(columns=["Last Name", "First Name"])

    # Make salary paid and taxable benefits numeric
    for column in ["Salary", "Benefits"]:
        df[column] = str_to_float(df[column]) if is_object_dtype(df[column]) else df[column]

    # Insert missing years
    df.loc[pd.isna(df["Year"]), "Year"] = year
    df["Year"] = df["Year"].astype(np.int64)

    # Finish preprocessing
    return df


def main() -> None:
    """
    Preprocess salary disclosures from all years.

    Notes
    -----
    Preprocessed data is saved as separate CSVs for each year and can be read using pandas as follows.

    >>> import pandas as pd
    >>> filepath = "data/preprocessed/2024.csv"
    >>> df = pd.read_csv(filepath, na_values=[""], keep_default_na=False)

    This example only reads preprocessed data from 2024, but applies for all years.
    """

    # Cover years with salary disclosures
    for year in range(1996, 2025):
        print(f"Preprocessing year {year}...")

        # Preprocess and save data
        filepath = f"data/preprocessed/{year}.csv"
        preprocess_salary_disclosure(year).to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
