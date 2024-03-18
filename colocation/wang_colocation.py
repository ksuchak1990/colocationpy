"""
A class to calculate the colocation rate based on Wang et al. (2011)
"""

# Imports
from typing import Tuple, Union
import numpy as np
import pandas as pd
from colocation.base_colocation import BaseColocation


# Define numeric type
Numeric = Union[int, float]


class WangColocation(BaseColocation):
    """
    A class to perform colocation in the manner laid out by Wang et al. (2011).
    """

    def __init__(
        self, population_data: pd.DataFrame, location_data: pd.DataFrame
    ) -> None:
        super().__init__(population_data)
        self.locations = location_data
        self.t_tolerance = 1

    def get_delta_t(self) -> Numeric:
        """
        Get the time difference that we permit between observations in order
        for them to count towards spatio-temporal co-location.

        :return: Numeric value of time tolerance.
        """
        return self.t_tolerance

    @staticmethod
    def get_delta_x(a: int, b: int) -> int:
        """
        Get the locational difference between two locations. This method checks
        that two locations are the same, i.e.
        "location a is the same as location b"

        :param a: Location ID A
        :param b: Location ID B
        :return: 1 if the locations are the same, and 0 otherwise.

        """
        return 1 if a == b else 0

    def get_most_likely_location(self, user_id: int) -> int:
        """
        Identify the most likely location for a user to be.

        :param user_id: The ID of the user
        :return: The ID of the location where we are mostly likely to find the
        user.
        """
        locs = self.data["locationID"].unique()
        probabilities = []

        for loc in locs:
            prob = self.get_probability_of_visit(user_id, loc)
            probabilities.append(loc, prob)

        probabilities = np.array(probabilities)
        return np.argmax(probabilities, axis=0)[1]

    def get_probability_of_visit(self, user_id: int, location_id: int) -> float:
        """
        Calculate the probability of a user visiting a particular location.

        :param user_id: The ID of the user
        :param location_id: The ID of the location
        :return: The probability of the user visiting the location
        """
        tdf = self.data.loc[self.data["userID"] == user_id, :]
        n_user = self.get_number_of_observations(user_id)
        locations = list(tdf["locationID"])

        total = 0
        for i in range(n_user):
            a = self.get_delta_x(location_id, locations[i])
            b = n_user
            total += a / b

        return total

    def get_number_of_observations(self, user_id: int) -> int:
        tdf = self.data.loc[self.data["userID"] == user_id, :]
        return len(tdf)

    def get_individual_by_id(self, user_id: int) -> pd.DataFrame:
        tdf = self.data.loc[self.data["userID"] == user_id, :]
        return tdf

    def get_ith_location_id(self, user_id: int, i: int) -> int:
        tdf = self.get_individual_by_id(user_id)
        return tdf.iloc[i]["locationID"]

    def get_ith_time(self, user_id: int, i: int) -> int:
        tdf = self.get_individual_by_id(user_id)
        return tdf.iloc[i]["time"]

    def get_coords_of_location(self, location_id: int) -> Tuple[float, float]:
        mask = self.locations["locationID"] == location_id
        c = self.locations.loc[mask, ["x", "y"]]
        coords = (c.iloc[0]["x"], c.iloc[0]["y"])
        return coords

    def get_SCoL(self, individual1: int, individual2: int) -> float:
        total = 0

        for loc in self.data["locationID"].unique():
            prob1 = self.get_probability_of_visit(individual1, loc)
            prob2 = self.get_probability_of_visit(individual2, loc)
            prob = prob1 * prob2
            total += prob

        return total

    def get_heaviside_input(self, individual1: int, individual2: int,
                            index1: int, index2: int) -> int:
        delta_t = self.get_delta_t()
        time1 = self.get_ith_time(individual1, index1)
        time2 = self.get_ith_time(individual2, index2)
        time_diff = time1 - time2
        return delta_t - abs(time_diff)

    def get_CoL(self, individual1: int, individual2: int) -> float:
        number1 = self.get_number_of_observations(individual1)
        number2 = self.get_number_of_observations(individual2)

        top_total = 0
        bottom_total = 0

        for i in range(number1):
            for j in range(number2):
                heaviside_input = self.get_heaviside_input(individual1,
                                                           individual2,
                                                           i, j)
                common = np.heaviside(heaviside_input, 1)

                bottom_total += common
                loc1 = self.get_ith_location_id(individual1, i)
                loc2 = self.get_ith_location_id(individual2, j)
                location_diff = self.get_delta_x(loc1, loc2)
                top_total += common * location_diff

        return top_total / bottom_total
