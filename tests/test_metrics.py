"""
Set of tests for the metrics calculated in this package.
"""

import pandas as pd
import pytest

from colocationpy.metrics import get_entropies

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
