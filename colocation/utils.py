"""
A collection of utility functions.
"""

from typing import Tuple, TypeAlias, Union
import numpy as np
import pandas as pd

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


def run_dx_dt(lowerx, upperx, nx,
              lowert, uppert, nt,
              data, locations, colocation_class):
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


if __name__ == "__main__":
    l1 = (0, 0)
    l2 = (1, 1)
    print(get_distance(l1, l2))
