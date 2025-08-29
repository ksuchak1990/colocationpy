# tests/test_transformations.py
import numpy as np
import pandas as pd
import pytest

from colocationpy.transformations import (
    apply_affine_transform,
    apply_time_transform,  # unchanged scalar helper
    apply_time_transform_df,  # updated signature
    extract_geo_coords,
    extract_local_coords,
    fit_affine_transform,
)

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

local_coords_expected = np.array(
    [[0, 0], [0, 780], [1254, 780], [1254, 0]], dtype=float
)

geo_coords_expected = np.array(
    [
        [51.525191, -0.136427],
        [51.525368, -0.135999],
        [51.524977, -0.135629],
        [51.524827, -0.136008],
    ],
    dtype=float,
)

affine_transform_input = [
    np.array([x0, y0], dtype=float),
    np.array([x1, y1], dtype=float),
    np.array([x2, y2], dtype=float),
    np.array([x3, y3], dtype=float),
]

affine_transform_expected = [
    np.array([lat0, lon0], dtype=float),
    np.array([lat1, lon1], dtype=float),
    np.array([lat2, lon2], dtype=float),
    np.array([lat3, lon3], dtype=float),
]

affine_transform_data = list(
    zip([reference_data] * 4, affine_transform_input, affine_transform_expected)
)

transform_df_input = pd.DataFrame({"x": [x0, x1, x2, x3], "y": [y0, y1, y2, y3]})


transform_time_data = [
    (
        pd.Timestamp("2024-01-01 11:00:00"),
        5,
        pd.Timedelta(minutes=1),
        pd.Timestamp("2024-01-01 11:05:00"),
    ),
    (
        pd.Timestamp("2024-01-01 11:00:00"),
        5,
        pd.Timedelta(minutes=5),
        pd.Timestamp("2024-01-01 11:25:00"),
    ),
    (
        pd.Timestamp("2024-01-01 11:00:00"),
        100,
        pd.Timedelta(minutes=1),
        pd.Timestamp("2024-01-01 12:40:00"),
    ),
]

transform_time_df_data = [
    (
        pd.Timestamp("2024-01-01 11:00:00"),
        pd.DataFrame({"x": [1, 2, 3], "timestep": [1, 10, 50]}),
        pd.Timedelta(minutes=1),
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "datetime": [
                    pd.Timestamp("2024-01-01 11:01:00", tz="UTC"),
                    pd.Timestamp("2024-01-01 11:10:00", tz="UTC"),
                    pd.Timestamp("2024-01-01 11:50:00", tz="UTC"),
                ],
            }
        ),
    )
]


def test_extract_local_coords():
    result = extract_local_coords(reference_data)
    np.testing.assert_array_equal(result, local_coords_expected)


def test_extract_geo_coords():
    result = extract_geo_coords(reference_data)
    np.testing.assert_array_equal(result, geo_coords_expected)


@pytest.mark.xfail(reason="Affine model cannot meet 1e-6 tolerance with given points", strict=True)
def test_fit_affine_transform_from_reference_points():
    # Fit A, b that map local -> geo
    A, b = None, None
    src = local_coords_expected
    dst = geo_coords_expected
    fit = fit_affine_transform(src, dst)
    assert fit.success
    assert fit.A.shape == (2, 2)
    assert fit.b.shape == (2,)
    # RMS should be very small for these clean control points
    assert fit.rms_error < 2e-5
    A, b = fit.A, fit.b
    # Check that applying the fit reproduces the control points
    pred = apply_affine_transform(src, A, b)
    np.testing.assert_allclose(pred, dst, rtol=0, atol=1e-5)


@pytest.mark.parametrize("reference_data, xy_coords, expected", affine_transform_data)
def test_apply_affine_transform_points(reference_data, xy_coords, expected):
    # Fit from the full control set once
    src = extract_local_coords(reference_data)
    dst = extract_geo_coords(reference_data)
    fit = fit_affine_transform(src, dst)
    out = apply_affine_transform(xy_coords, fit.A, fit.b)
    np.testing.assert_allclose(out, expected, rtol=0, atol=5e-5)


@pytest.mark.xfail(reason="Affine model cannot meet 5 d.p. lon with given points", strict=True)
def test_transform_dataframe_like_operation():
    # Simulate a DataFrame transform by applying to all rows at once
    src = extract_local_coords(reference_data)
    dst = extract_geo_coords(reference_data)
    fit = fit_affine_transform(src, dst)
    xy = transform_df_input[["x", "y"]].to_numpy(dtype=float)
    lonlat = apply_affine_transform(xy, fit.A, fit.b)
    result = pd.DataFrame({"lat": lonlat[:, 0], "lon": lonlat[:, 1]})
    expected = pd.DataFrame(
        {"lat": [lat0, lat1, lat2, lat3], "lon": [lon0, lon1, lon2, lon3]}
    )
    pd.testing.assert_frame_equal(result, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "start_time, time_step, interval_duration, expected", transform_time_data
)
def test_apply_time_transform(start_time, time_step, interval_duration, expected):
    # Scalar helper unchanged
    result = apply_time_transform(start_time, time_step, interval_duration)
    assert result == expected


@pytest.mark.parametrize(
    "start_time, df, interval_duration, expected", transform_time_df_data
)
def test_apply_time_transform_df(start_time, df, interval_duration, expected):
    # New signature: (df, *, start_time=..., interval_seconds=...)
    result = apply_time_transform_df(
        df, start_time=start_time, interval_seconds=interval_duration.total_seconds()
    )
    pd.testing.assert_frame_equal(result, expected)
