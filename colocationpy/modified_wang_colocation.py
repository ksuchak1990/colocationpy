"""
A class to perform colocation in a manner similar to Wang (2011); this modifies
the method to work with continuous space.
"""

# Imports
import pandas as pd
from colocationpy.wang_colocation import WangColocation
from colocationpy.utils import get_distance

# Class


class ModifiedWangColocation(WangColocation):
    """
    A class to perform colocation in a manner similar to Wang (2011); this
    modifies the method to work with continuous space.
    """

    def __init__(
        self,
        population_data: pd.DataFrame,
        location_data: pd.DataFrame,
        x_tolerance: float,
        t_tolerance: float,
    ) -> None:
        super().__init__(population_data, location_data)
        self.x_tolerance = x_tolerance
        self.t_tolerance = t_tolerance

    def get_delta_x(self, a: int, b: int) -> int:
        coords1 = self.get_coords_of_location(a)
        coords2 = self.get_coords_of_location(b)

        dist = get_distance(coords1, coords2)

        return 1 if dist <= self.x_tolerance else 0
