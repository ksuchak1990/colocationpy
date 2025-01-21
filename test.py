import numpy as np
import pandas as pd
import skmob
from scipy.optimize import minimize

x1, y1 = 0, 0
lat1, lon1 = 51.525191, -0.136427
x2, y2 = 0, 780
lat2, lon2 = 51.525368, -0.135999
x3, y3 = 1254, 780
lat3, lon3 = 51.524977, -0.135629
x4, y4 = 1254, 0
lat4, lon4 = 51.524827, -0.136008

reference_data = [
    (x1, y1, lat1, lon1),
    (x2, y2, lat2, lon2),
    (x3, y3, lat3, lon3),
    (x4, y4, lat4, lon4),
]

# Extract x, y, lat, lon from the reference data
local_coords = np.array([(x, y) for x, y, _, _ in reference_data])
geo_coords = np.array([(lon, lat) for _, _, lat, lon in reference_data])

# Function to compute residuals for the transformation


def residuals(params):
    a, b, c, d, e, f = params
    # Apply affine transformation
    transformed = np.dot(local_coords, [[a, b], [d, e]]) + [c, f]
    # Compute residuals between transformed and geographic coordinates
    return np.sum((geo_coords - transformed) ** 2)


# Initial guess for the affine transformation parameters
initial_guess = [1, 0, 0, 0, 1, 0]  # Identity transformation

# Optimise to find the best-fit affine transformation
result = minimize(residuals, initial_guess)
a, b, c, d, e, f = result.x

# Function to apply the affine transformation


def apply_affine_transform(xy_coords, params):
    a, b, c, d, e, f = params
    transformed = np.dot(xy_coords, [[a, b], [d, e]]) + [c, f]
    return transformed


# Apply the transformation to the DataFrame
def transform_dataframe(df, params):
    xy_coords = df[["x", "y"]].values  # Extract x and y as numpy array
    transformed_coords = apply_affine_transform(
        xy_coords, params)  # Apply transform
    df["lon"], df["lat"] = transformed_coords[:, 0], transformed_coords[:, 1]
    df = df.drop(columns=["x", "y"])
    return df


# Transform local x-y coordinates to geographic lon-lat
transformed_coords = apply_affine_transform(local_coords, [a, b, c, d, e, f])

# Print the results
print("Affine transformation parameters:")
print(f"a: {a}, b: {b}, c: {c}, d: {d}, e: {e}, f: {f}")
print("\nTransformed geographic coordinates (lon, lat):")
for original, transformed in zip(local_coords, transformed_coords):
    print(f"Local: {original}, Geographic: {transformed}")

data_path = "~/Documents/dev/colocation_experiments/data/agent_traj_CINCHserverparams_sq_20240619_1_1723552143.csv"
data = pd.read_csv(data_path)
data = data.dropna(how="any")

print(data.head())

transformed_data = transform_dataframe(data, [a, b, c, d, e, f])
print(transformed_data.head())

tdf = skmob.TrajDataFrame(transformed_data, latitude="lat", longitude="lon",
                          datetime="timestep", user_id="id", timestamp=True)

print(tdf.head())

m = tdf.plot_trajectory()
m.show_in_browser()
