from shapely.geometry import Polygon, LineString
from colocation.utils import is_divided_by_barrier
import pytest


barrier_divide_data = [
    ((0, 0), (1, 1), LineString([(1, 0), (0, 1)]), True),
    ((0, 0), (1, 1), LineString([(1, 0), (2, 1)]), False),
    ((0, 0), (10, 10), Polygon([(0, 9), (1, 10), (10, 1), (9, 0)]), True),
    ((0, 10), (10, 0), Polygon([(0, 9), (1, 10), (10, 1), (9, 0)]), True),
    ((0, 0), (3, 3), Polygon([(0, 9), (1, 10), (10, 1), (9, 0)]), False),
]


@pytest.mark.parametrize("location1, location2, barrier, expected", barrier_divide_data)
def test_divided_by_barrier(location1, location2, barrier, expected):
    result = is_divided_by_barrier(location1, location2, barrier)
    assert result == expected
