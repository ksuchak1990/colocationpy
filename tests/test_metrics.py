"""
Set of tests for the metrics calculated in this package.
"""

import math

import numpy as np
import pandas as pd
import pytest

from colocationpy.metrics import (  # get_average_entropy,; get_entropies,
    get_individual_entropies,
    get_mutual_information,
    get_average_entropy,
)

CASES = {
    # Three unordered pairs with equal frequency: H = log2(3)
    "balanced_three": [("cat", "dog"), ("cat", "cat"), ("dog", "dog")],
    # Frequencies: {("cat","dog"):2, ("cat","cat"):1, ("dog","dog"):1} ⇒ [0.5, 0.25, 0.25]
    "skewed_three": [("cat", "dog"), ("cat", "dog"), ("cat", "cat"), ("dog", "dog")],
    # Empty → 0.0 by definition
    "empty": [],
}
# entropy_data = [
#     (
#         pd.DataFrame({"species_x": [1, 1, 2, 2], "species_y": [1, 2, 1, 2]}),
#         pd.Series([0.0, 1.0, 1.0, 0.0]),
#     ),
#     (
#         pd.DataFrame(
#             {"species_x": ["a", "a", "b", "b"], "species_y": ["a", "b", "a", "b"]}
#         ),
#         pd.Series([0.0, 1.0, 1.0, 0.0]),
#     ),
# ]

# average_entropy_data = [
#     (pd.DataFrame({"species_x": [1, 1, 2, 2], "species_y": [1, 2, 1, 2]}), 0.5),
#     (
#         pd.DataFrame(
#             {"species_x": ["a", "a", "b", "b"], "species_y": ["a", "b", "a", "b"]}
#         ),
#         1 / 2,
#     ),
#     (
#         pd.DataFrame(
#             {
#                 "species_x": [1, 1, 1, 2, 2, 2, 3, 3, 3],
#                 "species_y": [1, 2, 3, 1, 2, 3, 1, 2, 3],
#             }
#         ),
#         2 / 3,
#     ),
#     (
#         pd.DataFrame(
#             {
#                 "species_x": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
#                 "species_y": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
#             }
#         ),
#         3 / 4,
#     ),
# ]

mutual_information_data = [
    (pd.DataFrame({"species_x": [1, 1, 2, 2], "species_y": [1, 2, 1, 2]}), 0.0),
    (
        pd.DataFrame(
            {
                "species_x": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "species_y": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        ),
        0.0,
    ),
    (
        pd.DataFrame(
            {
                "species_x": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                "species_y": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            }
        ),
        0.0,
    ),
]

individual_entropies_data = [
    (
        pd.DataFrame(
            {
                "uid_x": [1, 1, 1, 1, 2, 1],
                "uid_y": [1, 2, 3, 4, 3, 5],
                "species_x": [1, 1, 1, 1, 2, 1],
                "species_y": [1, 2, 3, 1, 3, 2],
                "coloc_prob": [1, 1, 1, 1, 1, 1],
            }
        ),
        pd.DataFrame({"uid": [1, 2, 3, 4, 5], "species": [1, 2, 3, 1, 2]}),
        pd.DataFrame({"uid": [1, 2, 3, 4, 5], "entropy": [1.5, 1.0, 1.0, -0.0, -0.0]}),
    )
]


def make_df(pairs):
    return pd.DataFrame(pairs, columns=["species_x", "species_y"])


def expected_entropy_from_pairs(pairs) -> float:
    if not pairs:
        return 0.0
    # Unordered species pairs (sorted tuple), matching the intended metric
    ser = pd.Series([tuple(sorted(p)) for p in pairs])
    counts = ser.value_counts()
    total = int(counts.sum())
    p = counts.values / total
    # Shannon entropy base 2
    return float(-(p * np.log2(p)).sum())


@pytest.mark.parametrize("data, species_map, expected", individual_entropies_data)
def test_get_individual_entropies(data, species_map, expected):
    entropies = get_individual_entropies(data, species_map)

    pd.testing.assert_frame_equal(entropies, expected)


# def test_get_location_entropies():
#     data = pd.read_csv("data/trajectories.csv")
#     species_map = pd.read_csv("data/species.csv")
#     expected = pd.read_csv("data/entropies_by_location.csv")
#     entropies = get_location_entropies(data, species_map)

#     pd.testing.assert_frame_equal(entropies, expected)


# @pytest.mark.parametrize("data, expected", entropy_data)
# def test_get_entropies(data: pd.DataFrame, expected: pd.Series):
#     """
#     Test the `get_entropies()` method.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         A DataFrame of co-location instances.
#     expected : pd.Series
#         A Series of co-location instance entropies.

#     """
#     entropies = get_entropies(data)
#     pd.testing.assert_series_equal(entropies, expected)


@pytest.mark.parametrize(
    "case_name, expected",
    [
        ("balanced_three", math.log2(3)),
        ("skewed_three", 1.5),
        ("empty", 0.0),
    ],
)
def test_get_average_entropy_expected_values(case_name, expected):
    df = make_df(CASES[case_name])
    got = get_average_entropy(df)
    assert np.isclose(got, expected)


@pytest.mark.parametrize("case_name", list(CASES.keys()))
def test_get_average_entropy_matches_reference(case_name):
    pairs = CASES[case_name]
    df = make_df(pairs)
    ref = expected_entropy_from_pairs(pairs)
    got = get_average_entropy(df)
    assert np.isclose(got, ref)


@pytest.mark.parametrize("case_name", ["balanced_three", "skewed_three"])
def test_get_average_entropy_no_mutation(case_name):
    df = make_df(CASES[case_name])
    df_before = df.copy(deep=True)
    _ = get_average_entropy(df)
    # No extra columns, same data
    assert list(df.columns) == list(df_before.columns)
    assert df.equals(df_before)


@pytest.mark.parametrize("data, expected", mutual_information_data)
def test_get_mutual_information(data: pd.DataFrame, expected: float):
    """
    Test the `get_mutual_information()` method.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances.
    expected : float
        The expected average entropy.

    """
    actual = get_mutual_information(data)
    assert actual == expected
