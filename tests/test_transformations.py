# Imports
import numpy as np
import pandas as pd
import pytest

from colocationpy.transformations import (
    apply_affine_transform,
    apply_time_transform,
    apply_time_transform_df,
    construct_transformation,
    extract_geo_coords,
    extract_local_coords,
    transform_dataframe,
)

# Data
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

local_coords_expected = np.array([[0, 0], [0, 780], [1254, 780], [1254, 0]])
local_coords_data = [(reference_data, local_coords_expected)]

geo_coords_expected = np.array(
    [
        [-0.136427, 51.525191],
        [-0.135999, 51.525368],
        [-0.135629, 51.524977],
        [-0.136008, 51.524827],
    ]
)
geo_coords_data = [(reference_data, geo_coords_expected)]

construct_transformation_expected = np.array(
    [
        3.14580812e-07,
        -3.01036677e-07,
        -1.36414656e-01,
        5.17196817e-07,
        2.09615385e-07,
        5.15251977e01,
    ]
)
construct_transformation_data = [(reference_data, construct_transformation_expected)]

affine_transform_input = [
    np.array([x0, y0]),
    np.array([x1, y1]),
    np.array([x2, y2]),
    np.array([x3, y3]),
]

affine_transform_expected = [
    np.array([lon0, lat0]),
    np.array([lon1, lat1]),
    np.array([lon2, lat2]),
    np.array([lon3, lat3]),
]

affine_transform_data = []
for i in range(len(affine_transform_input)):
    affine_transform_data.append(
        (reference_data, affine_transform_input[i], affine_transform_expected[i])
    )


transform_df_input = pd.DataFrame({"x": [x0, x1, x2, x3], "y": [y0, y1, y2, y3]})
transform_df_expected = pd.DataFrame(
    {"lon": [lon0, lon1, lon2, lon3], "lat": [lat0, lat1, lat2, lat3]}
)

transform_df_data = [(transform_df_input, reference_data, transform_df_expected)]


# Define data for time transform test
# Each set of test values should have start_time, time_step, interval_duration
# and expected
# These should be of type pd.Timestamp, int, pd.Timedelta, pd.Timestamp
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

# Define data for dataframe time transform test
# This should be a dataframe with multiple columns
# One of the columns should be a timestamp
# The expected return should be the same dataframe but with the timestamp column
# replaced (or not?) by a datetime column
# The test should provide start_time, df, interval_duration and expected

transform_time_df_data = [
    (
        pd.Timestamp("2024-01-01 11:00:00"),
        pd.DataFrame({"x": [1, 2, 3], "timestep": [1, 10, 50]}),
        pd.Timedelta(minutes=1),
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "datetime": [
                    pd.Timestamp("2024-01-01 11:01:00"),
                    pd.Timestamp("2024-01-01 11:10:00"),
                    pd.Timestamp("2024-01-01 11:50:00"),
                ],
            }
        ),
    )
]

# Tests


@pytest.mark.parametrize("reference_data, expected", local_coords_data)
def test_extract_local_coords(reference_data, expected):
    result = extract_local_coords(reference_data)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("reference_data, expected", geo_coords_data)
def test_extract_geo_coords(reference_data, expected):
    result = extract_geo_coords(reference_data)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("reference_data, expected", construct_transformation_data)
def test_construct_transformation(reference_data, expected):
    result = construct_transformation(reference_data)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("reference_data, xy_coords, expected", affine_transform_data)
def test_apply_affine_transform(reference_data, xy_coords, expected):
    transform_params = construct_transformation(reference_data)
    result = apply_affine_transform(xy_coords, transform_params)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


@pytest.mark.parametrize("df, reference_data, expected", transform_df_data)
def test_transform_dataframe(df, reference_data, expected):
    transform_params = construct_transformation(reference_data)
    result = transform_dataframe(df, transform_params)
    np.testing.assert_array_almost_equal(result.values, expected.values, decimal=5)


@pytest.mark.parametrize(
    "start_time, time_step, interval_duration, expected", transform_time_data
)
def test_apply_time_transform(start_time, time_step, interval_duration, expected):
    result = apply_time_transform(start_time, time_step, interval_duration)
    assert result == expected


@pytest.mark.parametrize(
    "start_time, df, interval_duration, expected", transform_time_df_data
)
def test_apply_time_transform_df(start_time, df, interval_duration, expected):
    result = apply_time_transform_df(start_time, df, interval_duration)
    pd.testing.assert_frame_equal(result, expected)
