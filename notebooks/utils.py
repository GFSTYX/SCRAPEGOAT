import re
from typing import Literal

import polars as pl


def reorder_within(df: pl.DataFrame, x: str, by: str, within: str) -> pl.DataFrame:
    """
    Reorder a column within groups, similar to reorder_within from R's tidytext.

    Parameters:
    -----------
    df : polars.DataFrame
        The input DataFrame to be sorted and modified.
    x : str
        The name of the column to be reordered and used in concatenation.
    by : str
        The name of the column to group by (primary sorting column).
    within : str
        The name of the column to sort within groups (secondary sorting column).

    Returns:
    --------
    polars.DataFrame
        A new DataFrame sorted by the specified columns and containing a new
        column named '{x}_ordered' that concatenates values from x and by.

    Notes:
    ------
    - The function sorts the DataFrame in ascending order by 'by' and 'x',
      and in descending order by 'within'.
    - The new column '{x}_ordered' is created by concatenating 'x' and 'by'
      values, separated by '___'.
    - This function is inspired by the reorder_within function from R's tidytext
      package, adapted for use with Python's Polars library.

    Example:
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     'deck_type': ['A', 'B', 'C', 'A', 'B'],
    ...     'year': [2020, 2020, 2020, 2021, 2021],
    ...     'len': [10, 15, 5, 12, 8]
    ... })
    >>> reorder_within(df, 'deck_type', 'year', 'len')
    shape: (5, 4)
    ┌───────────┬──────┬─────┬────────────────────┐
    │ deck_type ┆ year ┆ len ┆ deck_type_ordered  │
    │ ---       ┆ ---  ┆ --- ┆ ---                │
    │ str       ┆ i64  ┆ i64 ┆ str                │
    ╞═══════════╪══════╪═════╪════════════════════╡
    │ B         ┆ 2020 ┆ 15  ┆ B___2020           │
    │ A         ┆ 2020 ┆ 10  ┆ A___2020           │
    │ C         ┆ 2020 ┆ 5   ┆ C___2020           │
    │ A         ┆ 2021 ┆ 12  ┆ A___2021           │
    │ B         ┆ 2021 ┆ 8   ┆ B___2021           │
    └───────────┴──────┴─────┴────────────────────┘
    """
    return df.sort([by, within, x], descending=[False, True, False]).with_columns(
        [pl.concat_str([pl.col(x), pl.lit("___"), pl.col(by)]).alias(f"{x}_ordered")]
    )

def remove_suffix(breaks: list[str]) -> list[str]:
    """To be used with reordered_within to remove the suffix from the reordered column in the axis label."""
    return [re.sub(r'^(.*?)_.*$', r'\1', label) for label in breaks]
