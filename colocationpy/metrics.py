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


def get_average_entropy(data: pd.DataFrame) -> float:
    """
    Calculate the average entropy for a DataFrame of co-location instances.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances, with columns "species_x" and
        "species_y" indicating the types/species of the two individuals involved
        in the co-location.

    Returns
    -------
    float
        The mean entropy for the interactions identified.
    """
    return np.mean(get_entropies(data))


def __get_joint_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the joint probability distribution of species_x and species_y.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances, with columns "species_x" and "species_y".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the joint probabilities of species_x and species_y.
    """
    # Calculate joint frequencies (counts)
    joint_freq = pd.crosstab(data["species_x"], data["species_y"])

    # Convert frequencies to joint probabilities by dividing by the total number of instances
    total_instances = len(data)
    joint_prob = joint_freq / total_instances

    return joint_prob


def __get_marginal_distribution(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the marginal probability distribution for species_x and species_y.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances, with columns "species_x" and "species_y".

    Returns
    -------
    pd.Series
        A Series containing marginal probabilities for each species.
    """
    # Marginal probabilities for species_x
    species_x_prob = data["species_x"].value_counts(normalize=True)

    # Marginal probabilities for species_y
    species_y_prob = data["species_y"].value_counts(normalize=True)

    return species_x_prob, species_y_prob


def get_mutual_information(data: pd.DataFrame) -> float:
    """
    Calculate the mutual information between species_x and species_y in a vectorized way.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances, with columns "species_x" and "species_y".

    Returns
    -------
    float
        The mutual information between species_x and species_y.
    """
    # Get the joint and marginal probabilities
    joint_prob = __get_joint_distribution(data)
    species_x_prob, species_y_prob = __get_marginal_distribution(data)

    # Align the marginal probabilities with the joint distribution's index and columns
    p_x = species_x_prob.reindex(joint_prob.index).values[
        :, None
    ]  # Shape (n_species_x, 1)
    p_y = species_y_prob.reindex(joint_prob.columns).values[
        None, :
    ]  # Shape (1, n_species_y)

    # Convert the joint probability DataFrame into a NumPy array
    p_xy = joint_prob.values  # Shape (n_species_x, n_species_y)

    # Calculate the mutual information using vectorized operations
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_xy > 0, p_xy / (p_x * p_y), 0)  # Avoid divide by zero
        mutual_info_matrix = np.where(p_xy > 0, p_xy * np.log2(ratio), 0)

    mutual_info = np.sum(mutual_info_matrix)

    return mutual_info
