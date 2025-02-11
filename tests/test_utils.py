"""
Test functions in `colocation.utils`
"""

# Imports
import pytest
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, MultiPolygon

from colocationpy.utils import (
    get_closest_corner,
    get_distance_around_barrier,
    get_mahalanobis_distance,
    get_discrete_proximity,
    is_divided_by_barrier,
)

diagonal_barrier = Polygon([(0, 9), (1, 10), (10, 1), (9, 0), (0, 9)])
vertical_barrier = Polygon([(4, 2), (5, 2), (5, 9), (4, 9), (4, 2)])
corner_barrier = Polygon([(0, 4), (0, 5), (5, 5), (5, 0), (4, 0), (4, 4), (0, 4)])
barrier1 = Polygon(
    [
        (35, 35),
        (35, 705),
        (850, 705),
        (850, 700),
        (750, 700),
        (750, 500),
        (400, 500),
        (400, 200),
        (200, 200),
        (200, 40),
        (335, 40),
        (335, 35),
        (35, 35),
    ]
)

barrier2 = Polygon(
    [
        (350, 35),
        (350, 40),
        (400, 40),
        (400, 500),
        (1050, 500),
        (1050, 700),
        (865, 700),
        (865, 705),
        (1179, 705),
        (1179, 35),
        (350, 35),
    ]
)

barrier_geom = MultiPolygon([barrier1, barrier2])


# Define test data
barrier_divide_data = [
    ((0, 0), (1, 1), LineString([(1, 0), (0, 1)]), True),
    ((0, 0), (1, 1), LineString([(1, 0), (2, 1)]), False),
    ((0, 0), (10, 10), diagonal_barrier, True),
    ((0, 10), (10, 0), diagonal_barrier, True),
    ((0, 0), (3, 3), diagonal_barrier, False),
    (
        (0, 0),
        (10, 10),
        corner_barrier,
        True,
    ),
    (
        (0, 10),
        (10, 10),
        corner_barrier,
        False,
    ),
    (
        (300, 0),
        (300, 100),
        barrier_geom,
        True,
    ),
    (
        (0, 100),
        (350, 100),
        barrier_geom,
        True,
    ),
    (
        (250, 100),
        (350, 100),
        barrier_geom,
        False,
    ),
]

corner_data = [((0, 5), vertical_barrier, Point((4, 2)))]

barrier_distance_data = [
    ((0, 5), (9, 5), vertical_barrier, 11),
    ((0, 5), (10, 5), Polygon([(4, 2), (6, 2), (6, 9), (4, 9), (4, 2)]), 12),
]

mahalanobis_data = [
    ((0, 0), (1, 0), 1, 1, 1, 1, 0.5**0.5),
    ((0, 0), (1, 0), 0.5, 0.5, 0.5, 0.5, 1),
]

discrete_proximity_data = [
    (pd.DataFrame({"distance": [1, 2, 3]}), 2, pd.Series([True, True, False])),
    (pd.DataFrame({"distance": [1, 2, 3]}), 2.5, pd.Series([True, True, False])),
    (pd.DataFrame({"distance": [1, 2, 3]}), 0.5, pd.Series([False, False, False])),
    (
        pd.DataFrame({"distance": [0.002863, 0.004824, 0.0012]}),
        0.002,
        pd.Series([False, False, True]),
    ),
]


# Tests


@pytest.mark.parametrize("df, tolerance, expected", discrete_proximity_data)
def test_discrete_proximity(df, tolerance, expected):
    result = get_discrete_proximity(df, tolerance)
    print(expected)
    print(result)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("location1, location2, barrier, expected", barrier_divide_data)
def test_divided_by_barrier(location1, location2, barrier, expected):
    result = is_divided_by_barrier(location1, location2, barrier)
    assert result == expected


@pytest.mark.parametrize("location, barrier, expected", corner_data)
def test_closest_corner(location, barrier, expected):
    result = get_closest_corner(location, barrier)
    assert result == expected


@pytest.mark.parametrize(
    "location1, location2, barrier, expected", barrier_distance_data
)
def test_barrier_distance(location1, location2, barrier, expected):
    result = get_distance_around_barrier(location1, location2, barrier)
    assert result == expected


@pytest.mark.parametrize(
    "loc1, loc2, x_unc1, x_unc2, y_unc1, y_unc2, expected", mahalanobis_data
)
def test_get_mahalanobis_distance(loc1, loc2, x_unc1, x_unc2, y_unc1, y_unc2, expected):
    distance = get_mahalanobis_distance(loc1, loc2, x_unc1, x_unc2, y_unc1, y_unc2)
    assert distance == expected
