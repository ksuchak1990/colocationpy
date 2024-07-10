"""
Use a Density-EPR model to generate mobility trajectories for individuals based
around Leeds.
https://www.nature.com/articles/ncomms9166.pdf
"""

# Imports
import logging

import geopandas as gpd
import pandas as pd
import skmob
from skmob.measures.individual import (distance_straight_line,
                                       radius_of_gyration)
from skmob.models.epr import DensityEPR, SpatialEPR

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s"
)


# Functions
def make_populated_leeds_tessellation(
    pop: pd.DataFrame, tess: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # Filter tessellation down to Leeds
    leeds_tessellation = tess.loc[tess["LSOA21NM"].str.contains("Leeds"), :]
    populated_leeds_tessellation = leeds_tessellation.merge(
        right=pop, left_on="LSOA21CD", right_on="LSOA 2021 Code", how="left"
    )
    return populated_leeds_tessellation


def make_trajectories(
    start: str, end: str, tess: gpd.GeoDataFrame, pop_size: int, model_type: str
) -> skmob.TrajDataFrame:

    # Set data collection period
    start_time = pd.to_datetime(start)
    end_time = pd.to_datetime(end)

    # Set up model
    logging.info(f"Generating from {model_type} model")
    if model_type == "density":
        model = DensityEPR()
        trajectories = model.generate(
            start_time,
            end_time,
            tess,
            relevance_column="Total",
            n_agents=pop_size,
            random_state=28,
            show_progress=True,
        )
    elif model_type == "spatial":
        model = SpatialEPR()
        trajectories = model.generate(
            start_time,
            end_time,
            tess,
            n_agents=pop_size,
            random_state=28,
            show_progress=True,
        )
    else:
        raise ValueError(f"EPR model type not recognised: {model_type}")

    return trajectories


# Main
if __name__ == "__main__":
    clean_tessellation: bool = False
    verbose: bool = True
    to_vis: bool = False
    to_read: bool = False
    to_write: bool = True
    to_stats: bool = False
    # Takes values either "spatial" or "density"
    epr_type: str = "spatial"

    if to_read:
        logging.info("Reading trajectories")
        tdf = skmob.TrajDataFrame.from_file("data/traj.csv")
    else:
        if clean_tessellation:
            logging.info("Building clean tessellation")
            tessellation = gpd.read_file("data/lsoa_uk.geojson")
            population = pd.read_csv("data/populations_uk.csv")
            populated_leeds_tessellation = make_populated_leeds_tessellation(
                population, tessellation
            )
        else:
            populated_leeds_tessellation = gpd.read_file("data/populated_leeds.geojson")

        logging.info("Building trajectories")
        # Set data collection period
        START_TIME = "2024/01/01 08:00:00"
        END_TIME = "2024/01/14 08:00:00"

        tdf = make_trajectories(
            START_TIME, END_TIME, populated_leeds_tessellation, 100, epr_type
        )

    if verbose:
        logging.info("Sample of trajectories")
        print(tdf.head())

    if to_write:
        logging.info("Saving trajectories to csv")
        tdf.to_csv("data/traj.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    if to_vis:
        m = tdf.plot_trajectory()
        m.save("traj.html")

    if to_stats:
        SAMPLE_SIZE = 3
        first_people = tdf.loc[tdf["uid"] <= SAMPLE_SIZE, :]

        if verbose:
            logging.info("Sample of straight line distances")
            print(distance_straight_line(first_people))
            logging.info("Sample of radii of gyration")
            print(radius_of_gyration(first_people))
