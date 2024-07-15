"""
A collection of utility functions.
"""

from typing import List, Tuple, TypeAlias, Union

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point

from colocation.base_colocation import BaseColocation

# Define types
Numeric = Union[int, float]
Coordinate: TypeAlias = Tuple[float, float]


# Constants
R = 6378.1370


# Functions
def __check_required_columns(df: pd.DataFrame, rc: List[str]):
    required_cols = set(rc)
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"


def __get_haversine_distance(df: pd.DataFrame) -> pd.Series:
    # Convert lat-lng to radians
    lng1 = df["lng_x"] * np.pi / 180.0
    lng2 = df["lng_y"] * np.pi / 180.0
    lat1 = df["lat_x"] * np.pi / 180.0
    lat2 = df["lat_y"] * np.pi / 180.0

    # Calculate differences
    lng_diff = lng2 - lng1
    lat_diff = lat2 - lat1

    a = (np.sin(lat_diff / 2)) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(lng_diff / 2.0)
    ) ** 2
    c = 2.0 * np.atan2(np.sqrt(a), np.sqrt(1.0 - a))
    km = R * c
    return km


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
    __check_required_columns(df, ["lat_x", "lat_y", "lng_x", "lng_y"])

    distances = __get_haversine_distance(df)

    return distances


def get_discrete_proximity(df: pd.DataFrame, x_tolerance: float) -> pd.Series:
    nearby = np.where(df["distance"] <= x_tolerance, True, False)
    return pd.Series(nearby)


def get_triangular_proximity(df: pd.DataFrame, x_tolerance: float) -> pd.Series:
    proximity = 1 - (df["distance"] / x_tolerance)
    return proximity


def get_spatial_proximity(
    df: pd.DataFrame, x_tolerance: float, approach: str = "discrete"
) -> pd.Series:
    __check_required_columns(df, ["distance"])

    approaches = {
        "discrete": get_discrete_proximity,
        "triangular": get_triangular_proximity,
    }

    assert approach in approaches, f"{approach} is not a valid approach"

    proximity = approaches[approach](df, x_tolerance)
    return proximity


def get_time_difference(df: pd.DataFrame) -> pd.Series:
    __check_required_columns(df, ["datetime_x", "datetime_y"])

    time_difference = df["datetime_x"] - df["datetime_y"]
    return time_difference


def is_temporally_proximal(df: pd.DataFrame, t_tolerance: float) -> pd.Series:
    __check_required_columns(df, ["time_difference"])

    nearby = np.where(np.abs(df["time_difference"]) < t_tolerance, True, False)
    return pd.Series(nearby)


def get_all_ids(colocation_points: pd.DataFrame) -> list:
    ids_x = colocation_points["uid_x"].unique()
    ids_y = colocation_points["uid_y"].unique()
    ids = list(set(ids_x) | set(ids_y))
    ids = [int(x) for x in ids]
    return ids


def get_coordinate_centre(colocation_points: pd.DataFrame) -> Point:
    people_x = pd.DataFrame(
        {"lat": colocation_points["lat_x"], "lng": colocation_points["lng_x"]}
    )
    people_y = pd.DataFrame(
        {"lat": colocation_points["lat_y"], "lng": colocation_points["lng_y"]}
    )
    people = pd.concat([people_x, people_y])
    points = [tuple(r) for r in people.to_numpy()]
    points = MultiPoint(points)
    return points.centroid


if __name__ == "__main__":
    l1 = (0, 0)
    l2 = (1, 1)
    print(get_distance(l1, l2))
