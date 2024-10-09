"""
Set of tests for the metrics calculated in this package.
"""

import pandas as pd
import pytest

from colocationpy.metrics import get_average_entropy, get_entropies

entropy_data = [
    (
        pd.DataFrame({"species_x": [1, 1, 2, 2], "species_y": [1, 2, 1, 2]}),
        pd.Series([0.0, 1.0, 1.0, 0.0]),
    ),
    (
        pd.DataFrame(
            {"species_x": ["a", "a", "b", "b"], "species_y": ["a", "b", "a", "b"]}
        ),
        pd.Series([0.0, 1.0, 1.0, 0.0]),
    ),
]

average_entropy_data = [
    (pd.DataFrame({"species_x": [1, 1, 2, 2], "species_y": [1, 2, 1, 2]}), 0.5),
    (
        pd.DataFrame(
            {"species_x": ["a", "a", "b", "b"], "species_y": ["a", "b", "a", "b"]}
        ),
        1 / 2,
    ),
    (
        pd.DataFrame(
            {
                "species_x": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "species_y": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        ),
        2 / 3,
    ),
    (
        pd.DataFrame(
            {
                "species_x": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                "species_y": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            }
        ),
        3 / 4,
    ),
]


@pytest.mark.parametrize("data, expected", entropy_data)
def test_get_entropies(data: pd.DataFrame, expected: pd.Series):
    """
    Test the `get_entropies()` method.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances.
    expected : pd.Series
        A Series of co-location instance entropies.

    """
    entropies = get_entropies(data)
    pd.testing.assert_series_equal(entropies, expected)


@pytest.mark.parametrize("data, expected", average_entropy_data)
def test_get_average_entropy(data: pd.DataFrame, expected: float):
    """
    Test the `get_average_entropy()` method.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances.
    expected : float
        The expected average entropy.

    """
    actual = get_average_entropy(data)
    assert actual == expected
