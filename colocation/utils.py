"""
A collection of utility functions.
"""

from typing import Tuple, TypeAlias, Union

import numpy as np
import pandas as pd

from colocation.base_colocation import BaseColocation

# Define types
Numeric = Union[int, float]
Coordinate: TypeAlias = Tuple[float, float]


# Functions
def get_distance(location1: Coordinate, location2: Coordinate) -> float:
    """
    A method to get calculate the Euclidean distance between two points.

    :param location1: A tuple containing the x-y coordinates of the first
    point.
    :param location2: A tuple containing the x-y coordinates of the second
    point.
    :return: The distance, d, between the two points.
    """
    x_diff = location1[0] - location2[0]
    y_diff = location1[1] - location2[1]
    return (x_diff**2 + y_diff**2) ** 0.5


def run_dx_dt(
    lowerx: Numeric,
    upperx: Numeric,
    nx: Numeric,
    lowert: Numeric,
    uppert: Numeric,
    nt: Numeric,
    data: pd.DataFrame,
    locations: pd.DataFrame,
    colocation_class: BaseColocation,
) -> list:
    """
    Run a colocator for a range of $x$- and $t$-tolerances.

    :param lowerx: Lower bound for $x$-tolerance
    :param upperx: Upper bound for $x$-tolerance
    :param nx: Number of $x$-tolerances to try
    :param lowert: Lower bound for $t$-tolerance
    :param uppert: Upper bound for $t$-tolerance
    :param nt: Number of $t$-tolerances to try
    :param data: DataFrame of observations of individuals
    :param locations: Discrete locations at which individuals may be found
    :param colocation_class: Colocator to be used
    """
    cols = []

    for dx in np.linspace(lowerx, upperx, nx):
        for dt in np.linspace(lowert, uppert, nt):
            colocator = colocation_class(data, locations, dx, dt)
            col = colocator.get_CoL("x", "y")
            cols.append({"dx": dx, "dt": dt, "colocation_rate": col})

    return cols


def pivot_outputs(
    df: pd.DataFrame, i: str = "dx", c: str = "dt", v: str = "colocation_rate"
):
    """
    Pivot DataFrame of outputs by the deltas in preparation of making a heatmap
    of Co-Location rates.

    :param df: DataFrame of calculation outputs
    :param i: Index variable name
    :param c: Column variable name
    :param v: Variable name for values in pivot table
    """
    pivotted = df.pivot(index=i, columns=c, values=v)
    pivotted = pivotted.iloc[::-1]
    return pivotted


def get_distances(df: pd.DataFrame) -> pd.Series:
    required_cols = {"lat_x", "lat_y", "lng_x", "lng_y"}
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"

    distances = np.sqrt(
        (df["lat_x"] - df["lat_y"]) ** 2 + (df["lng_x"] - df["lng_y"]) ** 2
    )

    return distances


def is_spatially_proximal(df: pd.DataFrame, x_tolerance: float) -> pd.Series:
    required_cols = {"distance"}
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"

    nearby = np.where(df["distance"] <= x_tolerance, True, False)
    return pd.Series(nearby)


def get_time_difference(df: pd.DataFrame) -> pd.Series:
    required_cols = {"datetime_x", "datetime_y"}
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"

    time_difference = df["datetime_x"] - df["datetime_y"]
    return time_difference


def is_temporally_proximal(df: pd.DataFrame, t_tolerance: float) -> pd.Series:
    required_cols = {"time_difference"}
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"

    nearby = np.where(np.abs(df["time_difference"]) < t_tolerance, True, False)
    return pd.Series(nearby)


if __name__ == "__main__":
    l1 = (0, 0)
    l2 = (1, 1)
    print(get_distance(l1, l2))
