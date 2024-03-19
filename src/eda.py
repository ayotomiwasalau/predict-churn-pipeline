from util.logger import Logger
from src.datasource import Datasource
from util.plotter import Plotter
from constants.ref_cols import quant_columns
import pandas as pd


class EDA:
    """
    exploratory data analysis
    """

    def __init__(self, data):
        """
        Args:
            data (pandas.DataFrame): The data to perform EDA on.
        """
        self.data = data
        self.logger = Logger()
        self.plotter = Plotter()
        self.data['Churn'] = self.data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

    def display(self):
        """
        Returns:
            pandas.DataFrame: The original dataframe with the first 5 rows.
        """
        return self.data.head()

    def shape(self):
        """
        Returns:
            tuple: The number of rows and columns in the dataframe.
        """
        return self.data.shape

    def check_null(self):
        """
        Returns:
            pandas.DataFrame: A dataframe showing the number of null values in each column.
        """
        return self.data.isnull().sum()

    def show_summary(self):
        """
        Returns:
            pandas.DataFrame: A dataframe showing summary statistics for each column.
        """
        return self.data.describe()

    def return_df(self):
        """
        Returns:
            pandas.DataFrame: The original dataframe.
        """
        return self.data

    def general_stats(self):
        """
        Generate data statistics and log them to the console.
        """
        try:
            self.logger.logging.info("Generating data statistics...")
            self.logger.logging.info(
                f"The dataframe has {self.shape()[0]} cols and {self.shape()[1]} rows"
            )
            self.logger.logging.info("Check the nulls of the dataframe...")
            self.logger.logging.info(self.check_null())
            self.logger.logging.info("Show summary statistics for data...")
            self.logger.logging.info(self.show_summary())
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def plot_churn_hist_distribution(self):
        """
        Plots a histogram of the churn column.

        Returns:
            True if the plot is generated successfully, False otherwise.

        Raises:
            Exception: if an error occurs while generating the plot.
        """
        try:
            self.plotter.plot_and_save_histogram(
                self.data['Churn'], 'churn_hist_distribution')
            self.logger.logging.info(
                'Churn distirbution chart complete and saved')
            return True
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def plot_customer_hist_distribution(self):
        """
        Plots a histogram of the customer age distribution.

        Returns:
            True if the plot is generated successfully, False otherwise.

        Raises:
            Exception: if an error occurs while generating the plot.
        """
        try:
            self.plotter.plot_and_save_histogram(
                self.data['Customer_Age'], 'customer_hist_distribution')
            self.logger.logging.info(
                'Customer distirbution chart complete and saved')
            return True
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def plot_marital_status_dist(self):
        """
        Plots a bar chart of the marital status distribution.

        Returns:
            True if the plot is generated successfully, False otherwise.

        Raises:
            Exception: if an error occurs while generating the plot.
        """
        try:
            data = self.data.Marital_Status.value_counts('normalize')
            self.plotter.plot_and_save_barchart(data, 'marital_status_dist')
            self.logger.logging.info(
                'Marital status bar chart complete and saved')
            return True
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def plot_total_trans_density_hist_distribution(self):
        """
        Plots a histogram of the total transaction count density.

        Returns:
            True if the plot is generated successfully, False otherwise.

        Raises:
            Exception: if an error occurs while generating the plot.
        """
        try:
            data = self.data['Total_Trans_Ct']
            self.plotter.plot_and_save_densityhist(
                data, 'total_trans_density_hist_distribution')
            self.logger.logging.info(
                'Total transaction density chart complete and saved')
            return True
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def plot_heatmap_to_check_correllation(self):
        """
        Plots a heatmap of the dataframe to check for feature correlations.

        Returns:
            True if the plot is generated successfully, False otherwise.

        Raises:
            Exception: if an error occurs while generating the plot.
        """
        try:
            self.plotter.plot_and_save_heatmap(
                self.data, 'heatmap_to_feat_correllation')
            self.logger.logging.info('Heatmap corr chart complete and saved')
            return True
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))
