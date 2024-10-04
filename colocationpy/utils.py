"""
A collection of utility functions.
"""

from typing import List, Tuple, TypeAlias, Union

import numpy as np
import pandas as pd
from shapely import intersects, distance
from shapely.geometry import LineString, MultiPoint, Point, mapping, Polygon
from shapely.ops import nearest_points

from colocationpy.base_colocation import BaseColocation

# Define types
Numeric = Union[int, float]
Coordinate: TypeAlias = Tuple[float, float]
Barrier = Union[LineString, Polygon]


# Constants
R = 6378.1370


# Functions
def __check_required_columns(df: pd.DataFrame, rc: List[str]):
    """
    Ensure that a dataframe has a collection of required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be checked.
    rc : List[str]
        List of column names required in dataframe.

    """
    required_cols = set(rc)
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"


def __get_haversine_distance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the haversine distance between coordinate observations.

    Parameters
    ----------
    df : pd.DataFrame
        Collection of pairs of coordinates for comparison for co-location.


    Returns
    -------
    pd.Series
        Series of distances.

    """
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
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
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
    """
    Generate a series of distance measurements between individuals x and y,
    described by coordinates (lat_x, lng_x) and (lat_y, lng_y) respectively.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing the pairs of coordinates to be compared.

    Returns
    -------
    pd.Series
        A series of distance measurements generated using a haversine distance
        calculation.

    """
    __check_required_columns(df, ["lat_x", "lat_y", "lng_x", "lng_y"])

    distances = __get_haversine_distance(df)

    return distances


def is_divided_by_barrier(
    location1: Coordinate, location2: Coordinate, wall_geometry: Barrier
) -> bool:
    """
    Check if two locations are divided by a barrier.

    Parameters
    ----------
    location1 : Coordinate
        First location to be considered.
    location2 : Coordinate
        Second location to be considered.
    wall_geometry : Barrier
        Barrier geometry that may be between the two locations.

    Returns
    -------
    bool
        Indication of whether the given barrier divides the two locations.

    """
    connecting_line = LineString([Point(location1), Point(location2)])

    return intersects(connecting_line, wall_geometry)


def get_closest_point(location: Coordinate, barrier: Barrier):
    return barrier.exterior.interpolate(barrier.exterior.project(location))


def get_closest_vertex_index(boundaries, point):
    idx = min(
        range(len(boundaries)),
        key=lambda i: Point(boundaries[i]).distance(point),
    )
    return idx


def get_opposing_line_segments(boundaries, idx1, idx2):
    segment_1 = LineString(boundaries[idx1 : idx2 + 1])
    segment_2 = LineString(boundaries[idx2:] + boundaries[: idx1 + 1])
    return segment_1, segment_2


def get_indoor_distance(
    location1: Coordinate, location2: Coordinate, barrier: Barrier
) -> float:
    if is_divided_by_barrier(location1, location2, barrier):
        return get_distance_around_barrier(location1, location2, barrier)

    return get_distance(location1, location2)


def get_distance_around_barrier(
    location1: Coordinate, location2: Coordinate, barrier: Barrier
) -> float:
    """
    Calculate the distance between two locations, given a barrier between them.

    Parameters
    ----------
    location1 : Coordinate
        First location to consider.
    location2 : Coordinate
        Second location to consider.
    barrier : Barrier
        Barrier geometry between two locations.

    Returns
    -------
    float
        Shortest distance from one location to the other around the barrier.

    """
    location1 = Point(location1)
    location2 = Point(location2)

    # Find the closest points on the barrier to p1 and p2
    closest_point_to_p1 = get_closest_point(location1, barrier)
    closest_point_to_p2 = get_closest_point(location2, barrier)

    # Find the closest vertex indices to these points
    boundary_coords = list(barrier.exterior.coords)
    idx1 = get_closest_vertex_index(boundary_coords, closest_point_to_p1)
    idx2 = get_closest_vertex_index(boundary_coords, closest_point_to_p2)

    # Calculate the distance around the barrier in both directions
    if idx1 < idx2:
        segment_1, segment_2 = get_opposing_line_segments(boundary_coords, idx1, idx2)
    else:
        segment_1, segment_2 = get_opposing_line_segments(boundary_coords, idx2, idx1)

    path_1_distance = (
        location1.distance(Point(boundary_coords[idx1]))
        + segment_1.length
        + Point(boundary_coords[idx2]).distance(location2)
    )
    path_2_distance = (
        location2.distance(Point(boundary_coords[idx1]))
        + segment_2.length
        + Point(boundary_coords[idx2]).distance(location2)
    )

    return min(path_1_distance, path_2_distance)


def get_closest_corner(location, barrier) -> Point:
    points = barrier.exterior.coords
    location = Point(location)
    corner = min(points, key=lambda corner: location.distance(Point(corner)))
    return Point(corner)


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
    """
    Calculate time difference between two different observations.

    Parameters
    ----------
    df : pd.DataFrame
        Collection of pairs of coordinates for comparison for co-location.

    Returns
    -------
    pd.Series
        Collection of time differences.

    """
    __check_required_columns(df, ["datetime_x", "datetime_y"])

    time_difference = df["datetime_x"] - df["datetime_y"]
    return time_difference


def is_temporally_proximal(df: pd.DataFrame, t_tolerance: float) -> pd.Series:
    """
    Check if a pair of observations are temporally proximal, i.e. whether the
    time difference between the observations is less than a given tolerance.

    Parameters
    ----------
    df : pd.DataFrame
        Collection of pairs of coordinates for comparison for co-location.
    t_tolerance : float
        The permissible length of time between observations.

    Returns
    -------
    pd.Series
        Series of booleans indicating whether observations are within the given
        tolerance.

    """
    __check_required_columns(df, ["time_difference"])

    nearby = np.where(np.abs(df["time_difference"]) < t_tolerance, True, False)
    return pd.Series(nearby)


def get_all_ids(colocation_points: pd.DataFrame) -> list:
    """
    Get list of IDs of individuals in a population.

    Parameters
    ----------
    colocation_points : pd.DataFrame
        Collection of pairs of coordinates for comparison for co-location.

    Returns
    -------
    list
        List of unique IDs.

    """
    ids_x = colocation_points["uid_x"].unique()
    ids_y = colocation_points["uid_y"].unique()
    ids = list(set(ids_x) | set(ids_y))
    ids = [int(x) for x in ids]
    return ids


def get_coordinate_centre(colocation_points: pd.DataFrame) -> Point:
    """
    Identify geographical centre of coordinate pairs.

    Parameters
    ----------
    colocation_points : pd.DataFrame
        Collection of pairs of coordinates for comparison for co-location.

    Returns
    -------
    Point
        The centroid of the coordinates.

    """
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


def get_mahalanobis_distance(
    location1: Coordinate,
    location2: Coordinate,
    x_uncertainty1: float,
    x_uncertainty2: float,
    y_uncertainty1: float,
    y_uncertainty2: float,
) -> float:
    """
    Calculate the Mahalanobis distance between two coordinate locations which
    are observed with some degree of uncertainty around them. This works on the
    assumption that uncertainty around the observation is normally distributed,
    with the provided locations representing the means of the two 2-d gaussian
    distributions. Furthermore, we assume that the uncertainty is uncorrelated,
    i.e. there is no x-y correlation, resulting in covariance matrices that are
    diagonal.

    Parameters
    ----------
    location1 : Coordinate
        (x, y) location of the first observation.
    location2 : Coordinate
        (x, y) location of the second observation.
    x_uncertainty1 : float
        x-uncertainty of the first observation.
    x_uncertainty2 : float
        x-uncertainty of the second observation.
    y_uncertainty1 : float
        y-uncertainty of the first observation.
    y_uncertainty2 : float
        y-uncertainty of the second observation.

    Returns
    -------
    float
        The Mahalanobis distance between the two locations.

    """
    x_measure = (location1[0] - location2[0]) ** 2 / (x_uncertainty1 + x_uncertainty2)
    y_measure = (location1[0] - location2[1]) ** 2 / (y_uncertainty1 + y_uncertainty2)
    return np.sqrt(x_measure + y_measure)


if __name__ == "__main__":
    l1 = (0, 0)
    l2 = (1, 1)
    print(get_distance(l1, l2))
