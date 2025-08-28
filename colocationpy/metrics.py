"""
A collection of measurements that we may wish to take when considering
instances of co-location.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Core schema for contact-level rows used across metrics.
CONTACTS_SCHEMA = DataFrameSchema(
    {
        # or pa.String if you’ve normalised to str
        "uid_x": Column(object),
        "uid_y": Column(object),
        "species_x": Column(object),
        "species_y": Column(object),
    },
    strict=False,  # allow extra columns; we only assert what we actually need
)

# When functions rely on a precomputed weight or count per (uid_x, uid_y) edge:
EDGE_WEIGHTS_SCHEMA = DataFrameSchema(
    {
        "uid_x": Column(object),
        "uid_y": Column(object),
        "weight": Column(float, checks=Check.ge(0.0)),
    },
    strict=False,
)


def __get_shannon_entropy(proportions: list[float]) -> float:
    proportion_entropies = [p * np.log2(p) for p in proportions if p > 0]
    return -sum(proportion_entropies)


def __get_record_entropy(record: pd.Series, species_proportions: pd.Series) -> float:
    all_species = [record["species_x"], record["species_y"]]

    proportions = [species_proportions[species] for species in all_species]
    record_entropy = __get_shannon_entropy(proportions)
    # species_counts = Counter(all_species)
    # pop_size = sum(species_counts.values())

    # proportions = [count / pop_size for count in species_counts.values()]

    # record_entropy = __get_shannon_entropy(proportions)
    return record_entropy


def __get_proportions(data: pd.DataFrame) -> pd.Series:
    schema = pa.DataFrameSchema({"species_x": pa.Column(), "species_y": pa.Column()})
    schema.validate(data)

    x_counts = data["species_x"].value_counts()
    y_counts = data["species_y"].value_counts()
    total_counts = x_counts.add(y_counts, fill_value=0)

    total_proportions = total_counts / total_counts.sum()
    return total_proportions


def __get_counts(
    df: pd.DataFrame,
    species_map: pd.DataFrame,
    primary_column: str,
    secondary_column: str,
    primary_id: int | str,
) -> pd.Series:
    tdf = df.loc[df[primary_column] == primary_id, :]
    tdf = tdf.loc[tdf[secondary_column] != primary_id, :]
    individuals_in_contact = pd.DataFrame({"uid": tdf[secondary_column].unique()})

    individuals_in_contact = pd.merge(
        left=individuals_in_contact, right=species_map, on="uid", how="left"
    )

    counts = individuals_in_contact["species"].value_counts()

    return counts


def get_individual_entropies(
    data: pd.DataFrame, species_map: pd.DataFrame
) -> pd.DataFrame:
    # required_columns = ["uid_x", "uid_y", "species_x", "species_y", "coloc_prob"]
    # check_for_required_columns(data, required_columns)
    DataFrameSchema(
        {"uid_x": Column(object), "uid_y": Column(object)}, strict=False
    ).validate(data, lazy=False)
    DataFrameSchema(
        {"uid": Column(object), "species": Column(object)}, strict=False
    ).validate(species_map, lazy=False)

    ids_x = set(data["uid_x"].unique())
    ids_y = set(data["uid_y"].unique())
    all_ids = sorted(list(ids_x.union(ids_y)))

    # if not np.isclose(data["coloc_prob"], 1.0).all():
    #     raise ValueError("You have used the probabilistic approach!")

    # print("You have used the deterministic approach")

    entropies = []

    for i in all_ids:
        # uid_x
        counts_x = __get_counts(data, species_map, "uid_x", "uid_y", i)

        # uid_y
        counts_y = __get_counts(data, species_map, "uid_y", "uid_x", i)

        all_counts = counts_x.add(counts_y, fill_value=0)
        total = int(all_counts.sum())
        if total == 0:
            entropy = 0.0
        else:
            proportions = all_counts / total
            entropy = __get_shannon_entropy(proportions)

        d = {"uid": i, "entropy": entropy}

        entropies.append(d)

    entropies = pd.DataFrame(entropies)
    return entropies


def get_location_entropies(
    data: pd.DataFrame, species_map: pd.DataFrame, location_col: str = "locationID"
) -> pd.DataFrame:
    # required_columns = [location_col, "uid", "species"]
    # check_for_required_columns(data, required_columns)

    locations = data[location_col].unique()

    entropies = []

    for location in locations:
        tdf = data.loc[data[location_col] == location, :]
        individuals_at_location = pd.DataFrame({"uid": tdf["uid"].unique()})
        individuals_at_location = pd.merge(
            left=individuals_at_location, right=species_map, on="uid", how="left"
        )
        counts = individuals_at_location["species"].value_counts()

        total = int(counts.sum())
        entropy = 0.0 if total == 0 else __get_shannon_entropy(counts / total)

        d = {location_col: location, "entropy": entropy}
        entropies.append(d)

    entropies = pd.DataFrame(entropies)

    return entropies


def get_entropies(
    data: pd.DataFrame,
    *,
    species_map: pd.DataFrame,
    how: str = "location",
    location_col: str = "locationID",
) -> pd.DataFrame:
    """
    Compute Shannon entropies using a chosen aggregation strategy.

    Parameters
    ----------
    data : pandas.DataFrame
        Input table.
        When ``how == "individual"``, requires columns ``"uid_x"``, ``"uid_y"``.
        When ``how == "location"``, requires columns ``location_col`` and ``"uid"``.
    species_map : pandas.DataFrame
        Mapping of individuals to species with columns ``"uid"`` and ``"species"``.
    how : {"location", "individual"}, default "location"
        Aggregation strategy:
        - ``"location"``: entropy of species mix per location.
        - ``"individual"``: entropy of species mix across each individual's contacts.
    location_col : str, default "locationID"
        Column in ``data`` naming the location identifier (used when ``how == "location"``).

    Returns
    -------
    pandas.DataFrame
        If ``how == "location"``, columns ``[location_col, "entropy"]``.
        If ``how == "individual"``, columns ``["uid", "entropy"]``.

    Raises
    ------
    ValueError
        If ``how`` is not one of the supported options.
    pandera.errors.SchemaError
        If required columns are missing from ``data`` or ``species_map``.
    """
    # Validate species_map once
    DataFrameSchema(
        {"uid": Column(object), "species": Column(object)}, strict=False
    ).validate(species_map, lazy=False)

    if how == "individual":
        DataFrameSchema(
            {"uid_x": Column(object), "uid_y": Column(object)}, strict=False
        ).validate(data, lazy=False)
        return get_individual_entropies(data, species_map)

    if how == "location":
        DataFrameSchema(
            {location_col: Column(object), "uid": Column(object)}, strict=False
        ).validate(data, lazy=False)
        return get_location_entropies(data, species_map, location_col=location_col)

    raise ValueError("Unsupported value for 'how'. Use 'location' or 'individual'.")


# def get_entropies(data: pd.DataFrame) -> pd.Series:
#     """
#     Calculate entropies for a DataFrame of co-location instances.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         A DataFrame of co-location instances, with columns "species_x" and
#         "species_y" indicating the types/species of the two individuals involved
#         in the co-location.

#     Returns
#     -------
#     pd.Series
#         A Series of entropy measures for each of the co-location instances,
#         where 1.0 means that the two types are not the same, and 0.0 means that
#         they are the same.
#     """
#     species_proportions = __get_proportions(data)

#     entropies = data.apply(
#         __get_record_entropy, axis=1, species_proportions=species_proportions
#     )
#     return entropies


def __get_species_pair(row: pd.Series):
    return tuple(sorted([row["species_x"], row["species_y"]]))


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
    pairs = data.apply(__get_species_pair, axis=1)

    pair_counts = pairs.value_counts()

    total_interactions = int(pair_counts.sum())
    if total_interactions == 0:
        return 0.0
    probabilities = pair_counts / total_interactions

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy
    # return np.mean(get_entropies(data))


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
    SPECIES_ONLY_SCHEMA = DataFrameSchema(
        {"species_x": Column(object), "species_y": Column(object)}, strict=False
    )
    SPECIES_ONLY_SCHEMA.validate(data, lazy=False)

    # Calculate joint frequencies (counts)
    joint_freq = pd.crosstab(data["species_x"], data["species_y"])

    # Convert frequencies to joint probabilities by dividing by the total number of instances
    total_instances = len(data)
    joint_prob = joint_freq / total_instances

    return joint_prob


def __get_marginal_distribution(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Calculate the marginal probability distributions for ``species_x`` and ``species_y``.

    Parameters
    ----------
    data : pandas.DataFrame
        Co-location records with required columns ``"species_x"`` and ``"species_y"``.
        Extra columns are permitted and ignored.

    Returns
    -------
    tuple of pandas.Series
        ``(p_x, p_y)``, where:
        - ``p_x`` is the marginal distribution over ``species_x`` (index: species label).
        - ``p_y`` is the marginal distribution over ``species_y`` (index: species label).
        Probabilities sum to 1.0 along each Series. Empty input returns empty Series.

    Raises
    ------
    pandera.errors.SchemaError
        If required columns are missing.
    """
    DataFrameSchema(
        {"species_x": Column(object), "species_y": Column(object)},
        strict=False,
    ).validate(data, lazy=False)

    n = len(data)
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    p_x = data["species_x"].value_counts().sort_index() / n
    p_y = data["species_y"].value_counts().sort_index() / n

    # Ensure float dtype and stable names (optional but tidy)
    p_x = p_x.astype(float)
    p_y = p_y.astype(float)
    p_x.name = "P(species_x)"
    p_y.name = "P(species_y)"

    return p_x, p_y


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


def get_species_interaction_network(data: pd.DataFrame) -> nx.Graph:
    DataFrameSchema(
        {"species_x": Column(object), "species_y": Column(object)}, strict=False
    ).validate(data, lazy=False)

    ctab = pd.crosstab(data["species_x"], data["species_y"])
    adjacency = (ctab + ctab.T).fillna(0)
    np.fill_diagonal(adjacency.values, 0)
    graph = nx.from_pandas_adjacency(adjacency)

    # Ensure no self-loops remain
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph


def get_interaction_network(data: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected individual–individual interaction network.

    Parameters
    ----------
    data : pandas.DataFrame
        Contact rows with required columns:
        ``"uid_x"``, ``"uid_y"``, ``"species_x"``, ``"species_y"``.

    Returns
    -------
    networkx.Graph
        Undirected graph with individuals as nodes. Each node has a
        ``"species"`` attribute.
    """
    CONTACTS_SCHEMA.validate(data, lazy=False)

    # Initialize an undirected graph
    graph = nx.Graph()

    # Create edges between individuals (id_x and id_y) using vectorized operations
    edges = list(zip(data["uid_x"], data["uid_y"]))

    # Add edges to the graph
    graph.add_edges_from(edges)

    # Add species as node attributes using vectorized operations
    # Concatenate the 'id_x' and 'id_y' columns along with their respective species
    nodes_data = pd.concat(
        [
            data[["uid_x", "species_x"]].rename(
                columns={"uid_x": "id", "species_x": "species"}
            ),
            data[["uid_y", "species_y"]].rename(
                columns={"uid_y": "id", "species_y": "species"}
            ),
        ]
    )

    # Drop duplicates so each individual is processed only once
    nodes_data = nodes_data.drop_duplicates(subset="id")

    # Add species information as node attributes
    species_dict = nodes_data.set_index("id")["species"].to_dict()
    nx.set_node_attributes(graph, species_dict, "species")

    return graph


def get_network_modularity(data: pd.DataFrame) -> float:
    """
    Calculate the network modularity for a DataFrame of co-location instances.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances, with columns "species_x" and "species_y".

    Returns
    -------
    float
        The network modularity of the co-location network.

    """
    graph = get_interaction_network(data)

    species_communities = nx.get_node_attributes(graph, "species")

    # Create communities based on species
    communities = {}
    for node, species in species_communities.items():
        if species not in communities:
            communities[species] = []
        communities[species].append(node)

    # Convert communities into the list format expected by networkx's
    # modularity function
    community_list = list(communities.values())

    # Calculate the modularity of the graph based on species communities
    modularity = nx.algorithms.community.modularity(graph, community_list)
    return modularity


def get_clustering_coefficient(data: pd.DataFrame) -> float:
    """
    Calculate the clustering coefficient for a DataFrame of co-location instances.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame of co-location instances, with columns "species_x" and "species_y".

    Returns
    -------
    float
        The average clustering coefficient of the co-location network.

    """
    graph = get_interaction_network(data)
    clustering = nx.average_clustering(graph)
    return clustering


def get_assortativity_coefficient(data: pd.DataFrame) -> float:
    graph = get_interaction_network(data)
    assortativity = nx.attribute_assortativity_coefficient(graph, "species")
    return assortativity
