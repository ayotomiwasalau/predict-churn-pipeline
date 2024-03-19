from util.logger import Logger
import pandas as pd
from constants.ref_cols import keep_cols


class FeatureEngineering:
    """
    This class contains methods for feature engineering.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the class.

        Args:
            df (pd.DataFrame): The dataframe to work on.
        """
        self.df = df
        self.logger = Logger()

    def return_df(self) -> pd.DataFrame:
        """
        Return the dataframe.

        Returns:
            pd.DataFrame: The dataframe.
        """
        return self.df

    def encode_data_col(self, col: str, new_col: str, predictor: str) -> None:
        """
        Encode a data column using the mean of the predictor column.

        Args:
            col (str): The name of the column to encode.
            new_col (str): The name of the new encoded column.
            predictor (str): The name of the column to use for encoding.
        """
        self.df[new_col] = self.df[col].map(
            self.df.groupby(col)[predictor].mean()
        )
