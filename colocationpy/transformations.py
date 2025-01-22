# Imports
import numpy as np
import pandas as pd
import pandera as pa
from scipy.optimize import minimize


# Functions
def extract_local_coords(reference_data: dict[str, dict[str, float]]) -> np.ndarray:
    return np.array([(v["x"], v["y"]) for v in reference_data.values()])


def extract_geo_coords(reference_data: dict[str, dict[str, float]]) -> np.ndarray:
    return np.array([(v["lon"], v["lat"]) for v in reference_data.values()])


def calculate_residuals(transform_params, local_coords, geo_coords):
    # Function to compute residuals for the transformation
    a, b, c, d, e, f = transform_params
    transformed = np.dot(local_coords, [[a, b], [d, e]]) + [c, f]
    # Compute residuals between transformed and geographic coordinates
    return np.sum((geo_coords - transformed) ** 2)


def apply_affine_transform(xy_coords, transform_params):
    # Function to apply the affine transformation
    a, b, c, d, e, f = transform_params
    transformed = np.dot(xy_coords, [[a, b], [d, e]]) + [c, f]
    return transformed


def transform_dataframe(
    df: pd.DataFrame, transform_params: list[float], replace: bool = True
) -> pd.DataFrame:
    # Ensure that data contains x-y columns
    schema = pa.DataFrameSchema({"x": pa.Column(), "y": pa.Column()})
    schema.validate(df)

    # Apply the transformation to the DataFrame
    xy_coords = df[["x", "y"]].values
    transformed_coords = apply_affine_transform(xy_coords, transform_params)
    df["lon"], df["lat"] = transformed_coords[:, 0], transformed_coords[:, 1]

    # Get rid of original x-y columns if necessary
    df = df.drop(columns=["x", "y"]) if replace else df
    return df


def construct_transformation(
    reference_data: dict[str, dict[str, float]],
    initial_guess: list[float] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
) -> np.ndarray:
    # Extract x, y, lat, lon from the reference data
    local_coords = extract_local_coords(reference_data)
    geo_coords = extract_geo_coords(reference_data)

    # Optimise to find the best-fit affine transformation
    result = minimize(
        calculate_residuals,
        initial_guess,
        args=(local_coords, geo_coords),
        options={"disp": False},
        method="Powell",
    )

    return result.x
