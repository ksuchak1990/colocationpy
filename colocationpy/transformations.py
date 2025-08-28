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
        # WARNING: Not sure which method is most appropriate here
        method="Powell",
    )

    return result.x


def apply_time_transform(
    start_time: pd.Timestamp, time_step: int, interval_duration: pd.Timedelta
) -> pd.Timestamp:
    return start_time + (time_step * interval_duration)


def apply_time_transform_df(
    df: pd.DataFrame,
    *,
    timestep_col: str = "timestep",
    start_time: pd.Timestamp | str = "1970-01-01T00:00:00Z",
    interval_seconds: int | float = 60,
    replace: bool = False,
    out_col: str = "datetime",
) -> pd.DataFrame:
    """
    Map integer timesteps to wall-clock datetimes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing a timestep column.
    timestep_col : str, default "timestep"
        Name of the integer timestep column to convert.
    start_time : pandas.Timestamp or str, default "1970-01-01T00:00:00Z"
        Start reference time (inclusive). Strings are parsed with UTC.
    interval_seconds : int or float, default 60
        Duration of one timestep in seconds. Must be > 0.
    replace : bool, default False
        If True, overwrite ``timestep_col`` with datetimes; otherwise write to ``out_col``.
    out_col : str, default "datetime"
        Destination column when ``replace`` is False.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with datetimes added (or replacing ``timestep_col``).

    Raises
    ------
    KeyError
        If ``timestep_col`` is missing.
    ValueError
        If ``interval_seconds`` <= 0.
    """
    if timestep_col not in df.columns:
        raise KeyError(f"Missing required column: {timestep_col!r}")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")

    out = df.copy()
    origin = pd.to_datetime(start_time, utc=True)

    steps = pd.to_numeric(out[timestep_col], errors="coerce")
    # NaNs in steps become NaT in the output (expected, and preferable to crashing)
    delta = pd.to_timedelta(steps * float(interval_seconds), unit="s")
    datetimes = origin + delta

    if replace:
        out[timestep_col] = datetimes
    else:
        out[out_col] = datetimes
    return out
