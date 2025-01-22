# Imports
import pandas as pd
import skmob
import logging
from colocationpy.transformations import construct_transformation, transform_dataframe

logging.basicConfig(level=logging.INFO)


# Trial run
x0, y0 = 0, 0
lat0, lon0 = 51.525191, -0.136427
x1, y1 = 0, 780
lat1, lon1 = 51.525368, -0.135999
x2, y2 = 1254, 780
lat2, lon2 = 51.524977, -0.135629
x3, y3 = 1254, 0
lat3, lon3 = 51.524827, -0.136008

reference_data = {
    "origin": {"x": x0, "y": y0, "lat": lat0, "lon": lon0},
    "p1": {"x": x1, "y": y1, "lat": lat1, "lon": lon1},
    "p2": {"x": x2, "y": y2, "lat": lat2, "lon": lon2},
    "p3": {"x": x3, "y": y3, "lat": lat3, "lon": lon3},
}


# Initial guess for the affine transformation parameters
initial_guess = [1, 0, 0, 0, 1, 0]

logging.info("Setting up transformation")
affine_transform = construct_transformation(reference_data, initial_guess=initial_guess)

# Apply to indoor mobility trajectories
logging.info("Reading in data")
data_path = "../colocation_experiments/data/agent_traj_CINCHserverparams_sq_20240619_1_1723552143.csv"
data = pd.read_csv(data_path)
data = data.dropna(how="any")

logging.info("Transforming data")
transformed_data = transform_dataframe(data, affine_transform)

tdf = skmob.TrajDataFrame(
    transformed_data,
    latitude="lat",
    longitude="lon",
    datetime="timestep",
    user_id="id",
    timestamp=True,
)

logging.info("Plotting trajectories")
m = tdf.plot_trajectory(max_users=5)
m.save("outputs/indoor_trajectories.html")
