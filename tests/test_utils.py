"""
Test functions in `colocation.utils`
"""

# Imports
import math

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from colocationpy.utils import (
    get_closest_corner,
    get_discrete_proximity,
    get_distance_around_barrier,
    get_mahalanobis_distance,
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

P = {
    "O": (0.0, 0.0),
    "A": (1.0, 2.0),
    "B": (3.0, 4.0),
    "C": (2.0, 1.0),
    "D": (2.0, -1.0),
    "E": (-5.0, 3.0),
}

V = {
    "ones": (1.0, 1.0),
    "zeros": (0.0, 0.0),
    "x23": (2.0, 3.0),
    "y45": (4.0, 5.0),
    "x15": (1.5, 0.5),
    "y2535": (2.5, 3.5),
    "twos": (2.0, 2.0),
}

NEG_VARS = [
    ((-1.0, 1.0), (1.0, 1.0)),
    ((1.0, 1.0), (-0.1, 0.0)),
    ((-0.5, -0.1), (0.0, 0.0)),
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


def expected_mahalanobis(a, b, vx, vy):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx / (vx[0] + vx[1]) + dy * dy / (vy[0] + vy[1]))


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
    "a,b,vx,vy,expected",
    [
        (
            P["A"],
            P["O"],
            V["ones"],
            V["ones"],
            math.sqrt(2.5),
        ),  # regression: y-index mix-up
        (
            P["O"],
            P["B"],
            V["ones"],
            V["ones"],
            5 / math.sqrt(2),
        ),  # 3-4-5 with unit variances
        (P["O"], P["C"], V["x23"], V["y45"], None),  # numeric non-symmetric
    ],
)
def test_mahalanobis_expected_values(a, b, vx, vy, expected):
    d = get_mahalanobis_distance(a, b, vx[0], vx[1], vy[0], vy[1])
    if expected is None:
        assert np.isclose(d, expected_mahalanobis(a, b, vx, vy))
    else:
        assert np.isclose(d, expected)


@pytest.mark.parametrize(
    "a,b,vx,vy",
    [
        (
            P["A"],
            P["O"],
            V["zeros"],
            V["ones"],
        ),  # zero combined x variance, non-zero dx
        (
            P["O"],
            P["A"],
            V["ones"],
            V["zeros"],
        ),  # zero combined y variance, non-zero dy
    ],
)
def test_mahalanobis_infinite_on_zero_variance_with_offset(a, b, vx, vy):
    assert get_mahalanobis_distance(a, b, vx[0], vx[1], vy[0], vy[1]) == float("inf")


def test_mahalanobis_zero_distance_with_zero_variance():
    assert get_mahalanobis_distance(P["O"], P["O"], 0.0, 0.0, 0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "a,b,vx,vy",
    [
        (P["D"], P["E"], V["x23"], V["y45"]),
        (P["B"], P["A"], V["x15"], V["y2535"]),
    ],
)
def test_mahalanobis_symmetry(a, b, vx, vy):
    d1 = get_mahalanobis_distance(a, b, vx[0], vx[1], vy[0], vy[1])
    d2 = get_mahalanobis_distance(b, a, vx[1], vx[0], vy[1], vy[0])
    assert np.isclose(d1, d2)


@pytest.mark.parametrize(
    "b_small,b_large", [(P["A"], (2.0, 2.0)), ((-1.0, 0.5), (-2.0, 1.0))]
)
def test_mahalanobis_monotonic_in_offset(b_small, b_large):
    a = P["O"]
    d_small = get_mahalanobis_distance(a, b_small, *V["twos"], *V["twos"])
    d_large = get_mahalanobis_distance(a, b_large, *V["twos"], *V["twos"])
    assert d_large > d_small


@pytest.mark.parametrize("vx,vy", NEG_VARS)
def test_mahalanobis_rejects_negative_variance(vx, vy):
    with pytest.raises(ValueError):
        get_mahalanobis_distance(P["O"], P["A"], vx[0], vx[1], vy[0], vy[1])
