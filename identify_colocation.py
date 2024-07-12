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
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = ArgumentParser(
    prog="Co-location identification", description="Identify instances of co-location"
)
parser.add_argument("N", type=int, default=50)
args = parser.parse_args()

# Functions
def get_distances(df: pd.DataFrame) -> pd.Series:
    required_cols = {"lat_x", "lat_y", "lng_x", "lng_y"}
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"

    distances = np.sqrt(
        (cross["lat_x"] - cross["lat_y"]) ** 2 + (cross["lng_x"] - cross["lng_y"]) ** 2
    )

    return distances


def is_spatially_proximal(df: pd.DataFrame, x_tolerance: float) -> pd.Series:
    required_cols = {"distance"}
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"

    nearby = np.where(df["distance"] <= x_tolerance, True, False)
    return pd.Series(nearby)


def get_time_difference(df: pd.DataFrame) -> pd.Series:
    required_cols = {"datetime_x", "datetime_y"}
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"

    time_difference = df["datetime_x"] - df["datetime_y"]
    return time_difference


def is_temporally_proximal(df: pd.DataFrame, t_tolerance: float) -> pd.Series:
    required_cols = {"time_difference"}
    provided_columns = set(df.columns)
    assert required_cols.issubset(provided_columns), "Missing required columns"

    nearby = np.where(np.abs(df["time_difference"]) < t_tolerance, True, False)
    return pd.Series(nearby)


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
