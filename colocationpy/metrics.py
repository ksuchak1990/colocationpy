"""
A collection of measurements that we may wish to take when considering
    instances of co-location.
"""

from collections import Counter

import numpy as np
import pandas as pd


def __get_shannon_entropy(proportions: list[float]) -> float:
    proportion_entropies = [p * np.log2(p) for p in proportions if p > 0]
    return -sum(proportion_entropies)


def __get_record_entropy(record: pd.Series) -> float:
    all_species = [record["species_x"], record["species_y"]]

    species_counts = Counter(all_species)
    pop_size = sum(species_counts.values())

    proportions = [count / pop_size for count in species_counts.values()]

    record_entropy = __get_shannon_entropy(proportions)
    return record_entropy


def get_entropies(data: pd.DataFrame) -> pd.Series:
    """
    Calculate entropies for a DataFrame of co-location instances.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances, with columns "species_x" and
        "species_y" indicating the types/species of the two individuals involved
        in the co-location.

    Returns
    -------
    pd.Series
        A Series of entropy measures for each of the co-location instances,
        where 1.0 means that the two types are not the same, and 0.0 means that
        they are the same.
    """
    entropies = data.apply(__get_record_entropy, axis=1)
    return entropies
