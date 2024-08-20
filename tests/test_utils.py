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

@pytest.mark.parametrize("location1, location2, barrier, expected", barrier_divide_data)
def test_divided_by_barrier(location1, location2, barrier, expected):
    result = is_divided_by_barrier(location1, location2, barrier)
    assert result == expected


@pytest.mark.parametrize("location, barrier, expected", corner_data)
def test_closest_corner(location, barrier, expected):
    result = get_closest_corner(location, barrier)
    assert result == expected
