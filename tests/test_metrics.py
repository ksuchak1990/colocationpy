# tests/test_metrics.py
import math

import numpy as np
import pandas as pd
import pytest

from colocationpy.metrics import (
    __get_marginal_distribution,
    get_average_entropy,
    get_entropies,
    get_individual_entropies,
    get_interaction_network,
    get_mutual_information,
    get_species_interaction_network,
)

CASES = {
    "balanced_three": [("cat", "dog"), ("cat", "cat"), ("dog", "dog")],
    "skewed_three": [("cat", "dog"), ("cat", "dog"), ("cat", "cat"), ("dog", "dog")],
    "empty": [],
}

SPECIES_MAP = pd.DataFrame(
    {
        "uid": ["u1", "u2", "u3", "u4", "u5"],
        "species": ["cat", "cat", "dog", "fox", "dog"],
    }
)

CONTACTS = pd.DataFrame(
    {
        "uid_x": ["u1", "u1", "u1", "u1", "u2", "u1"],
        "uid_y": ["u1", "u2", "u3", "u4", "u3", "u5"],
        "species_x": ["cat", "cat", "cat", "cat", "dog", "cat"],
        "species_y": ["cat", "dog", "fox", "cat", "dog", "dog"],
        "coloc_prob": [1, 1, 1, 1, 1, 1],
    }
)

LOCATIONS = pd.DataFrame(
    {
        "locationID": ["L1", "L1", "L2", "L2", "L3"],
        "uid": ["u1", "u2", "u3", "u4", "u1"],
    }
)

SPECIES_ONLY = pd.DataFrame(
    {
        "species_x": ["cat", "cat", "dog", "dog", "dog"],
        "species_y": ["dog", "cat", "cat", "fox", "dog"],
    }
)

SPECIES_NET_INPUT = pd.DataFrame(
    {
        "species_x": ["cat", "cat", "dog", "dog", "cat"],
        "species_y": ["dog", "cat", "cat", "fox", "cat"],
    }
)

MUTUAL_INFORMATION_DATA = [
    (
        pd.DataFrame(
            {"species_x": ["1", "1", "2", "2"], "species_y": ["1", "2", "1", "2"]}
        ),
        0.0,
    ),
    (
        pd.DataFrame(
            {
                "species_x": ["1", "1", "1", "2", "2", "2", "3", "3", "3"],
                "species_y": ["1", "2", "3", "1", "2", "3", "1", "2", "3"],
            }
        ),
        0.0,
    ),
    (
        pd.DataFrame(
            {
                "species_x": [
                    "1",
                    "1",
                    "1",
                    "1",
                    "2",
                    "2",
                    "2",
                    "2",
                    "3",
                    "3",
                    "3",
                    "3",
                    "4",
                    "4",
                    "4",
                    "4",
                ],
                "species_y": [
                    "1",
                    "2",
                    "3",
                    "4",
                    "1",
                    "2",
                    "3",
                    "4",
                    "1",
                    "2",
                    "3",
                    "4",
                    "1",
                    "2",
                    "3",
                    "4",
                ],
            }
        ),
        0.0,
    ),
]


INDIVIDUAL_ENTROPIES_DATA = [
    (
        CONTACTS,
        SPECIES_MAP.loc[SPECIES_MAP["uid"].isin(["u1", "u2", "u3", "u4", "u5"])],
        pd.DataFrame(
            {
                "uid": ["u1", "u2", "u3", "u4", "u5"],
                "entropy": [1.5, 1.0, 1.0, -0.0, -0.0],
            }
        ),
    )
]


def make_df(pairs):
    return pd.DataFrame(pairs, columns=["species_x", "species_y"])


def expected_entropy_from_pairs(pairs) -> float:
    if not pairs:
        return 0.0
    ser = pd.Series([tuple(sorted(p)) for p in pairs])
    counts = ser.value_counts()
    total = int(counts.sum())
    p = counts.values / total
    return float(-(p * np.log2(p)).sum())


def _reference_individual_entropies(
    contacts: pd.DataFrame, species_map: pd.DataFrame
) -> pd.DataFrame:
    """Mirror the implementation: unique partners per direction, exclude self."""
    species_of = dict(zip(species_map["uid"], species_map["species"]))
    # Sorted union of all uids, as in the implementation
    uids = sorted(set(contacts["uid_x"]).union(set(contacts["uid_y"])))
    rows = []
    for uid in uids:
        # Partners seen as uid_x (look at uid_y), unique and not self
        partners_x = pd.unique(contacts.loc[contacts["uid_x"] == uid, "uid_y"])
        partners_x = [p for p in partners_x if p != uid]
        sx = pd.Series(partners_x, dtype=object).map(species_of).dropna()
        counts_x = sx.value_counts()

        # Partners seen as uid_y (look at uid_x), unique and not self
        partners_y = pd.unique(contacts.loc[contacts["uid_y"] == uid, "uid_x"])
        partners_y = [p for p in partners_y if p != uid]
        sy = pd.Series(partners_y, dtype=object).map(species_of).dropna()
        counts_y = sy.value_counts()

        all_counts = counts_x.add(counts_y, fill_value=0)
        total = int(all_counts.sum())
        if total == 0:
            entropy = 0.0
        else:
            p = all_counts / total
            entropy = float(-(p * np.log2(p)).sum())
        rows.append((uid, entropy))
    return (
        pd.DataFrame(rows, columns=["uid", "entropy"])
        .sort_values("uid")
        .reset_index(drop=True)
    )


def test_get_individual_entropies_matches_reference():
    got = (
        get_individual_entropies(CONTACTS, SPECIES_MAP)
        .sort_values("uid")
        .reset_index(drop=True)
    )
    ref = _reference_individual_entropies(CONTACTS, SPECIES_MAP)
    pd.testing.assert_frame_equal(got, ref)


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
    assert list(df.columns) == list(df_before.columns)
    assert df.equals(df_before)


@pytest.mark.parametrize("data, expected", MUTUAL_INFORMATION_DATA)
def test_get_mutual_information(data: pd.DataFrame, expected: float):
    actual = get_mutual_information(data)
    assert actual == expected


def test_entropies_location_default_location_col():
    out = get_entropies(LOCATIONS, species_map=SPECIES_MAP, how="location")
    assert set(out.columns) == {"locationID", "entropy"}
    assert np.isfinite(out["entropy"]).all()
    assert (out["entropy"] >= 0).all()


def test_entropies_location_custom_location_col():
    df = LOCATIONS.rename(columns={"locationID": "LSOA"})
    out = get_entropies(
        df, species_map=SPECIES_MAP, how="location", location_col="LSOA"
    )
    assert set(out.columns) == {"LSOA", "entropy"}
    assert np.isfinite(out["entropy"]).all()


def test_entropies_individual_path():
    out = get_entropies(
        CONTACTS[["uid_x", "uid_y"]], species_map=SPECIES_MAP, how="individual"
    )
    assert set(out.columns) == {"uid", "entropy"}
    assert out["uid"].is_unique


def test_entropies_missing_species_map_raises():
    with pytest.raises(Exception):
        # type: ignore[arg-type]
        _ = get_entropies(
            CONTACTS[["uid_x", "uid_y"]], species_map=None, how="individual"
        )


def test_entropies_missing_required_columns_raises():
    with pytest.raises(Exception):
        _ = get_entropies(
            LOCATIONS.drop(columns=["uid"]), species_map=SPECIES_MAP, how="location"
        )


def test_marginal_distributions_sum_to_one():
    px, py = __get_marginal_distribution(SPECIES_ONLY)
    assert np.isclose(px.sum(), 1.0)
    assert np.isclose(py.sum(), 1.0)
    assert px.dtype == float and py.dtype == float


def test_marginal_empty_input_returns_empty_series():
    px, py = __get_marginal_distribution(SPECIES_ONLY.iloc[0:0])
    assert px.empty and py.empty
    assert px.dtype == float and py.dtype == float


VALID_CONTACTS = pd.DataFrame(
    {
        "uid_x": ["u1", "u2"],
        "uid_y": ["u2", "u3"],
        "species_x": ["cat", "dog"],
        "species_y": ["dog", "fox"],
    }
)

MISSING_CONTACT_CASES = {
    "no_uid_x": VALID_CONTACTS.drop(columns=["uid_x"]),
    "no_uid_y": VALID_CONTACTS.drop(columns=["uid_y"]),
    "no_species_x": VALID_CONTACTS.drop(columns=["species_x"]),
    "no_species_y": VALID_CONTACTS.drop(columns=["species_y"]),
}


@pytest.mark.parametrize(
    "bad", list(MISSING_CONTACT_CASES.values()), ids=list(MISSING_CONTACT_CASES.keys())
)
def test_interaction_network_requires_uid_and_species(bad):
    with pytest.raises(Exception):
        get_interaction_network(bad)


def test_interaction_network_builds_from_valid():
    g = get_interaction_network(VALID_CONTACTS)
    assert g.number_of_nodes() >= 1
    assert g.number_of_edges() >= 1


def test_species_network_symmetry_and_no_self_loops():
    g = get_species_interaction_network(SPECIES_NET_INPUT)
    w_cd = g.get_edge_data("cat", "dog")["weight"]
    assert w_cd == 2
    assert not any(u == v for u, v in g.edges())
