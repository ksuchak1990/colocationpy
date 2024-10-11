"""
A collection of measurements that we may wish to take when considering
instances of co-location.
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
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


def get_species_interaction_network(data: pd.DataFrame) -> nx.Graph:
    adjacency_matrix = pd.crosstab(data["species_x"], data["species_y"])
    graph = nx.from_pandas_adjacency(adjacency_matrix)
    return graph


def get_interaction_network(data: pd.DataFrame) -> nx.Graph:
    """
    Creates an undirected graph from a pandas DataFrame containing individual
    interaction data. Each individual is represented as a node, and interactions
    between individuals are represented as edges. Species information is added
    as node attributes.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing individual interaction data with columns 'id_x',
        'id_y', 'species_x', and 'species_y'.

    Returns
    -------
    G : nx.Graph
        A NetworkX graph representing the interaction network between
        individuals, with species as node attributes.
    """
    # Initialize an undirected graph
    graph = nx.Graph()

    # Create edges between individuals (id_x and id_y) using vectorized operations
    edges = list(zip(data["id_x"], data["id_y"]))

    # Add edges to the graph
    graph.add_edges_from(edges)

    # Add species as node attributes using vectorized operations
    # Concatenate the 'id_x' and 'id_y' columns along with their respective species
    nodes_data = pd.concat(
        [
            data[["id_x", "species_x"]].rename(
                columns={"id_x": "id", "species_x": "species"}
            ),
            data[["id_y", "species_y"]].rename(
                columns={"id_y": "id", "species_y": "species"}
            ),
        ]
    )

    # Drop duplicates so each individual is processed only once
    nodes_data = nodes_data.drop_duplicates(subset="id")

    # Add species information as node attributes
    species_dict = nodes_data.set_index("id")["species"].to_dict()
    nx.set_node_attributes(graph, species_dict, "species")

    return graph


def draw_interaction_network(graph: nx.Graph) -> None:
    # Get the species attribute for each node
    species = nx.get_node_attributes(graph, "species")

    # Generate a colour map for the species
    species_values = list(set(species.values()))  # Unique species values
    # Assign an index for each species
    colour_map = {species: idx for idx, species in enumerate(species_values)}
    # Map node colours based on species
    node_colours = [colour_map[species[node]] for node in graph.nodes()]

    plt.figure()
    nx.draw(graph, with_labels=True, node_color=node_colours)
    plt.show()


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
