"""
Identify instances of co-location between individuals in a set of mobility
trajectories.
"""

# Imports
import logging
from argparse import ArgumentParser
from itertools import combinations

import numpy as np
import pandas as pd
import skmob
from tqdm import tqdm

from colocation.utils import (get_distances, get_time_difference,
                              is_spatially_proximal, is_temporally_proximal)

# Constants
T_TOLERANCE = np.timedelta64(4, "h")
X_TOLERANCE = 0.0

# Basic setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = ArgumentParser(
    prog="Co-location identification", description="Identify instances of co-location"
)
parser.add_argument("N", type=int, default=50)
args = parser.parse_args()

# Read data
logging.info("Reading data")
tdf = skmob.TrajDataFrame.from_file("data/traj.csv")

# Subset data
N = args.N
logging.info(f"Considering {N} individuals")
stdf = tdf.loc[tdf["uid"] <= N, :]


# Iterate over combinations of individuals
logging.info("Generating combinations")
individuals = stdf["uid"].unique()
combos = list(combinations(individuals, 2))


# Create dict of subsets of data for each individual
individual_trajectories = {uid: stdf.loc[stdf["uid"] == uid, :] for uid in individuals}

all_observation_combinations = []
logging.info("Comparing trajectories")
for combo in tqdm(combos):
    person1 = individual_trajectories[combo[0]]
    person2 = individual_trajectories[combo[1]]

    cross = person1.merge(person2, how="cross")
    # cross["is_coloc"] = np.where(
    #     is_spatially_proximal(
    #         cross["lat_x"], cross["lat_y"], cross["lng_x"], cross["lng_y"]
    #     )
    #     & is_temporally_proximal(cross["datetime_x"], cross["datetime_y"]),
    #     1,
    #     0,
    # )
    cross["distance"] = get_distances(cross)
    cross["time_difference"] = get_time_difference(cross)
    cross["is_sloc"] = is_spatially_proximal(cross, X_TOLERANCE)
    cross["is_tloc"] = is_temporally_proximal(cross, T_TOLERANCE)
    cross["is_coloc"] = cross["is_sloc"] & cross["is_tloc"]

    coloc_instances = cross.loc[cross["is_coloc"] == 1, :]
    all_observation_combinations.append(coloc_instances)

logging.info("Collecting results")
all_observation_combinations = pd.concat(all_observation_combinations)

logging.info(f"Number of instances found: {(len(all_observation_combinations))}")
