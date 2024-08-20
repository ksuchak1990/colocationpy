"""
Test functions in `colocation.utils`
"""

# Imports
import pytest
from shapely.geometry import Polygon, LineString, Point
from colocation.utils import (
    is_divided_by_barrier,
    get_distance_around_barrier,
    get_closest_corner,
)


# Define test data
barrier_divide_data = [
    ((0, 0), (1, 1), LineString([(1, 0), (0, 1)]), True),
    ((0, 0), (1, 1), LineString([(1, 0), (2, 1)]), False),
    ((0, 0), (10, 10), Polygon([(0, 9), (1, 10), (10, 1), (9, 0), (0, 9)]), True),
    ((0, 10), (10, 0), Polygon([(0, 9), (1, 10), (10, 1), (9, 0), (0, 9)]), True),
    ((0, 0), (3, 3), Polygon([(0, 9), (1, 10), (10, 1), (9, 0), (0, 9)]), False),
    (
        (0, 0),
        (10, 10),
        Polygon([(0, 4), (0, 5), (5, 5), (5, 0), (4, 0), (4, 4), (0, 4)]),
        True,
    ),
]

corner_data = [
    ((0, 5), Polygon([(4, 2), (5, 2), (5, 9), (4, 9), (4, 2)]), Point((4, 2)))
]

barrier_distance_data = [
    ((0, 5), (9, 5), Polygon([(4, 2), (5, 2), (5, 9), (4, 9), (4, 2)]), 11),
    ((0, 5), (10, 5), Polygon([(4, 2), (6, 2), (6, 9), (4, 9), (4, 2)]), 12),
]


# Tests
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
