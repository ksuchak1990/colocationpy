"""
Identify instances of co-location between individuals in a set of indoor
mobility trajectories sourced from Nick's ABM.
"""

# Imports
import logging
from argparse import ArgumentParser
from itertools import combinations

import folium
import numpy as np
import pandas as pd
import skmob
from tqdm import tqdm
import pyproj
import geopandas as gpd

from colocation.utils import (
    get_all_ids,
    get_coordinate_centre,
    get_distances,
    get_spatial_proximity,
    get_time_difference,
    is_temporally_proximal,
)

# Constants
T_TOLERANCE = np.timedelta64(2, "m")
X_TOLERANCE = 1.0
SHOW_TRAJ = False

# Basic setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# parser = ArgumentParser(
#     prog="Co-location identification", description="Identify instances of co-location"
# )
# parser.add_argument("N", type=int, default=50)
# parser.add_argument("show_locations", type=bool, default=False)
# args = parser.parse_args()

# Setting up local crs
# Reference location for UCLH
reference_lat = 51.524468
reference_lon = -0.137571

# Define a custom local CRS based on the known reference point
local_crs = pyproj.CRS.from_proj4(
    f"+proj=tmerc +lat_0={reference_lat} +lon_0={
        reference_lon} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

# Read data
logging.info("Reading data")
df = pd.read_csv("data/agent_traj_CINCHserverparams_sq_20240619_1_1723552143.csv")

# 10 distance units in model equivalent to 1 metre

logging.info("Creating GeoDataFrame")
df["x"] = df["x"] / 10
df["y"] = df["y"] / 10

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))
gdf.set_crs(local_crs, inplace=True)
gdf = gdf.to_crs(epsg=4326)
gdf["latitude"] = gdf.geometry.y
gdf["longitude"] = gdf.geometry.x

logging.info("Creating TrajDataFrame")
tdf = skmob.TrajDataFrame(
    gdf,
    latitude="latitude",
    longitude="longitude",
    user_id="id",
    datetime="timestep",
    crs=local_crs.to_dict(),
)

logging.info("Select active agent data")
tdf = tdf.loc[tdf["status"] == "active", :]

logging.info("Saving trajectory plot")
m = tdf.plot_trajectory(max_users=5, max_points=None)
m.save("figures/colocation_abm.html")

first_step = tdf["datetime"].min()
first_step_agents = tdf.loc[tdf["datetime"] == first_step, :]
logging.info("Number of agents active in first step: %s", len(first_step_agents))
