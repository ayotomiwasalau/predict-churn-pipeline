import pandas as pd

from util.logger import Logger


class Datasource:
    """class for reading csv file from sources

    Args:
        fileurl (str): file path of the csv file

    Attributes:
        fileurl (str): file path of the csv file
        logger (Logger): instance of the Logger class for logging messages
    """

    def __init__(self, fileurl):
        self.fileurl = fileurl
        self.logger = Logger()

    def return_df(self):
        """method for reading the csv file and returning a pandas dataframe

        Returns:
            pd.DataFrame: a pandas dataframe containing the data from the csv file
        """
        try:
            df = pd.read_csv(self.fileurl)
            self.logger.logging.info("File import complete")
            return df
        except FileNotFoundError as e:
            self.logger.logging.error(e)
            raise FileNotFoundError
