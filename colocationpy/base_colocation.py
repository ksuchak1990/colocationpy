"""
A base class on which to base colocation methods.
"""

# Imports
import pandas as pd


# Class
class BaseColocation:
    def __init__(self, data: pd.DataFrame) -> None:
        self.required_columns = ["userID", "time"]
        self.data = data
        self.validate_data(data)

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Ensure that the data provided has the appropriate columns.

        :param data: DataFrame containing observations of individuals
        :raises ValueError: Alerts of missing columns
        """
        missing_cols = []
        for col in self.required_columns:
            if col not in list(data):
                missing_cols.append(col)

        if len(missing_cols) > 0:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def __len__(self) -> int:
        """
        Method to define the `len` of colocation classes.

        :return: The number of observations in our data.
        """
        return len(self.data)
