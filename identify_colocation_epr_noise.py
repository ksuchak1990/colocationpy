"""
Identify instances of co-location between individuals in a set of mobility
trajectories.
"""

# TO-DO
# Create projections
# Create projections for indoor mobility
# Flatten combo loop?
# Introduce altitude to distance measures?

# Imports
import logging
from argparse import ArgumentParser
from itertools import combinations

import folium
import numpy as np
import pandas as pd
import skmob
from tqdm import tqdm

from colocation.utils import (
    get_all_ids,
    get_coordinate_centre,
    get_distances,
    get_spatial_proximity,
    get_time_difference,
    is_temporally_proximal,
)

# Constants
T_TOLERANCE = np.timedelta64(2, "h")
X_TOLERANCE = 1.0
SHOW_TRAJ = True
SAVE_COLOCS = True
N = 100
SHOW_LOCATIONS = False
distance = 1000


# Functions
def add_random_noise_x_lon(row, x_lon=100):
    r = np.pi * row["lat"] / 180
    x_lon_dist = x_lon / (111320 * np.cos(r))

    x = row["lng"] + np.random.uniform(-x_lon_dist, x_lon_dist)
    return x


def add_random_noise_y_lat(row, y_lat=100):
    y_lat_dist = y_lat / 111320

    y = row["lat"] + np.random.uniform(-y_lat_dist, y_lat_dist)
    return y


# Basic setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Read data
logging.info("Reading data")
tdf = skmob.TrajDataFrame.from_file("data/traj.csv")

# Add noise to observations
stdf = tdf.copy()
tdf["lat"] = stdf.apply(add_random_noise_y_lat, axis=1, y_lat=distance)
tdf["lng"] = stdf.apply(add_random_noise_x_lon, axis=1, x_lon=distance)

# Subset data
pop_size = tdf["uid"].unique().shape[0]
logging.info("Considering %s individuals (%s%%)", N, (N / pop_size) * 100)
stdf = tdf.loc[tdf["uid"] <= N, :]

# Iterate over combinations of individuals
logging.info("Generating combinations")
individuals = stdf["uid"].unique()
combos = list(combinations(individuals, 2))

# Create dict of subsets of data for each individual
individual_trajectories = {uid: stdf.loc[stdf["uid"] == uid, :] for uid in individuals}

logging.info("Identifying co-locations within (%s km, %s)", X_TOLERANCE, T_TOLERANCE)
all_observation_combinations = []
for combo in tqdm(combos, desc="Comparing trajectories"):
    # Get relevant trajectories
    person1 = individual_trajectories[combo[0]]
    person2 = individual_trajectories[combo[1]]

    # Create cross product for comparisons
    cross = person1.merge(person2, how="cross")

    # Calculate spatial and temporal displacements
    cross["distance"] = get_distances(cross)
    cross["time_difference"] = get_time_difference(cross)

    # Identify points which are near to each other depending on tolerances
    cross["is_sloc"] = get_spatial_proximity(cross, X_TOLERANCE, "triangular")
    cross["is_tloc"] = is_temporally_proximal(cross, T_TOLERANCE)

    # Define points that are co-located
    cross["is_coloc"] = cross["is_sloc"] * cross["is_tloc"]
    coloc_instances = cross.loc[cross["is_coloc"] > 0.8, :]

    # Collate results
    all_observation_combinations.append(coloc_instances)

logging.info("Collecting results")
all_observation_combinations = pd.concat(all_observation_combinations)

logging.info("Number of instances found: %s", len(all_observation_combinations))

if SAVE_COLOCS:
    logging.info("Writing co-location combinations to `.csv`")
    all_observation_combinations.to_csv(f"data/epr_coloc_{distance}.csv")

# if SHOW_LOCATIONS:
#     logging.info("Creating map of co-locations")

#     # Create base map
#     if SHOW_TRAJ:
#         m = tdf.plot_trajectory(zoom=11, max_users=10, opacity=0.5)
#     else:
#         c = get_coordinate_centre(all_observation_combinations)
#         m = folium.Map(location=[c.x, c.y], zoom_start=11)

#     # Create feature groups for each individual
#     # Get complete list of ids
#     ids = get_all_ids(all_observation_combinations)

#     # Create feature groups
#     fgs = {i: folium.FeatureGroup(name=i, show=False).add_to(m) for i in ids}
#     # fgs["trajectories"] = folium.FeatureGroup(
#     #     name="trajectories", show=True).add_to(m)

#     # Add trajectories
#     # tdf.plot_trajectory(zoom=12).add_to(fgs["trajectories"])

#     # Add markers
#     for i in range(len(all_observation_combinations)):
#         record = all_observation_combinations.iloc[i]
#         folium.Marker(
#             location=[record["lat_y"], record["lng_y"]], popup=record["uid_y"]
#         ).add_to(fgs[record["uid_y"]])
#         folium.Marker(
#             location=[record["lat_x"], record["lng_x"]], popup=record["uid_x"]
#         ).add_to(fgs[record["uid_x"]])

#     folium.LayerControl().add_to(m)
#     m.save("figures/colocations.html")
