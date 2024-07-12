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

# Constants
T_TOLERANCE = np.timedelta64(4, "h")
X_TOLERANCE = 0.0

# Basic setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s"
)

parser = ArgumentParser(
    prog="Co-location identification", description="Identify instances of co-location"
)
parser.add_argument("N", type=int, default=50)
args = parser.parse_args()

# Functions


def is_spatially_proximal(lat1: float, lat2: float, lng1: float, lng2: float) -> bool:
    return (lat1 == lat2) & (lng1 == lng2)


def is_temporally_proximal(t1: np.timedelta64, t2: np.timedelta64) -> bool:
    return ((t1 - t_tolerace) <= t2) & (t2 <= (t1 + t_tolerace))


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

all_observation_combinations = []

logging.info("Comparing trajectories")
for combo in tqdm(combos):
    person1 = stdf.loc[stdf["uid"] == combo[0]]
    person2 = stdf.loc[stdf["uid"] == combo[1]]

    cross = person1.merge(person2, how="cross")
    cross["is_coloc"] = np.where(
        is_spatially_proximal(
            cross["lat_x"], cross["lat_y"], cross["lng_x"], cross["lng_y"]
        )
        & is_temporally_proximal(cross["datetime_x"], cross["datetime_y"]),
        1,
        0,
    )
    coloc_instances = cross.loc[cross["is_coloc"] == 1, :]
    all_observation_combinations.append(coloc_instances)

logging.info("Collecting results")
all_observation_combinations = pd.concat(all_observation_combinations)

logging.info(
    f"Number of instances found: {(len(all_observation_combinations))}")
