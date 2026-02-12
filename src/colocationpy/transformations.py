# Imports
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera as pa
from scipy.optimize import minimize

GEO_COORD_ORDER = ("lat", "lon")


@dataclass(frozen=True)
class AffineFit:
    """
    Result of an affine fit y ≈ A x + b (2D).
    """

    A: np.ndarray  # (2, 2)
    b: np.ndarray  # (2,)
    rms_error: float
    success: bool
    message: str


# Functions
def extract_local_coords(reference_data: dict[str, dict[str, float]]) -> np.ndarray:
    return np.array([(v["x"], v["y"]) for v in reference_data.values()])


def extract_geo_coords(reference_data: dict[str, dict[str, float]]) -> np.ndarray:
    k0, k1 = GEO_COORD_ORDER  # ("lat", "lon")
    return np.array([(v[k0], v[k1]) for v in reference_data.values()], dtype=float)


def calculate_residuals(transform_params, local_coords, geo_coords):
    # Function to compute residuals for the transformation
    a, b, c, d, e, f = transform_params
    transformed = np.dot(local_coords, [[a, b], [d, e]]) + [c, f]
    # Compute residuals between transformed and geographic coordinates
    return np.sum((geo_coords - transformed) ** 2)


def apply_affine_transform(
    points: np.ndarray, A: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Apply y = A x + b to 2D points.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 2).
    A : numpy.ndarray
        Affine matrix of shape (2, 2).
    b : numpy.ndarray
        Translation vector of shape (2,).

    Returns
    -------
    numpy.ndarray
        Transformed points, shape (N, 2).

    Raises
    ------
    ValueError
        If input shapes are incompatible.
    """
    P = np.asarray(points, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(b, dtype=float)

    if A.shape != (2, 2):
        raise ValueError("A must have shape (2, 2).")
    if B.shape != (2,):
        raise ValueError("b must have shape (2,).")

    if P.ndim == 1:
        if P.shape != (2,):
            raise ValueError("points must have shape (2,) or (N, 2).")
        return (A @ P) + B  # shape (2,)
    if P.ndim == 2 and P.shape[1] == 2:
        return P @ A.T + B  # shape (N, 2)
    raise ValueError("points must have shape (2,) or (N, 2).")


def transform_dataframe(
    df: pd.DataFrame, transform_params: list[float], replace: bool = True
) -> pd.DataFrame:
    # Ensure that data contains x-y columns
    schema = pa.DataFrameSchema({"x": pa.Column(), "y": pa.Column()})
    schema.validate(df)

    # Apply the transformation to the DataFrame
    xy_coords = df[["x", "y"]].values

    a, b_, c, d, e, f = transform_params
    A = np.array([[a, b_], [d, e]])
    b = np.array([c, f])

    transformed_coords = apply_affine_transform(xy_coords, A, b)

    # transformed_coords = apply_affine_transform(xy_coords, transform_params)
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
        If True, overwrite ``timestep_col`` with datetimes; otherwise write to
        ``out_col``.
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
    delta = pd.to_timedelta(steps * float(interval_seconds), unit="s")
    datetimes = origin + delta

    out = df.copy()
    if replace:
        # Overwrite the timestep column in place
        out[timestep_col] = datetimes
    else:
        out[out_col] = datetimes
        if timestep_col in out.columns:
            out = out.drop(columns=[timestep_col])
    return out


def _ols_affine(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Closed-form least-squares initialiser for y ≈ A x + b.
    Returns (A, b, rms_error).
    """
    n = src.shape[0]
    X = np.hstack([src, np.ones((n, 1))])  # [x, y, 1]
    theta, *_ = np.linalg.lstsq(X, dst, rcond=None)  # (3, 2)
    A = theta[:2, :].T  # (2, 2)
    b = theta[2, :]  # (2,)
    resid = dst - (src @ A.T + b)
    rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    return A, b, rms


def _unpack_affine(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map flat parameter vector -> (A, b).
    """
    A = theta[:4].reshape(2, 2)
    b = theta[4:]
    return A, b


def _affine_objective(theta: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """
    Mean squared error objective for y ≈ A x + b.
    """
    A, b = _unpack_affine(theta)
    pred = src @ A.T + b
    err = pred - dst
    return float(np.mean(np.sum(err * err, axis=1)))


def fit_affine_transform(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> AffineFit:
    """
    Fit an affine transform y ≈ A x + b mapping 2D points src→dst, using SciPy.

    Parameters
    ----------
    src, dst : numpy.ndarray
        Arrays of shape (N, 2), N ≥ 2, same shape, no NaNs.
    maxiter : int, default 1000
        Maximum iterations for the optimiser.
    tol : float, default 1e-8
        Convergence tolerance for the optimiser.

    Returns
    -------
    AffineFit
        Dataclass containing A, b, RMS error, success flag, and message.

    Raises
    ------
    ValueError
        If shapes are invalid or inputs contain NaNs.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError("src and dst must be the same shape (N, 2).")
    if np.isnan(src).any() or np.isnan(dst).any():
        raise ValueError("NaNs in src/dst are not supported.")
    if src.shape[0] < 2:
        raise ValueError("At least two point pairs are required.")

    A, b, rms = _ols_affine(src, dst)
    return AffineFit(
        A=A,
        b=b,
        rms_error=rms,
        success=True,
        message="Closed-form OLS affine fit",
    )
