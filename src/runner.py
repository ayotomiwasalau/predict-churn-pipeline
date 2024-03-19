from src.model import Model
from src.feature_engineering import FeatureEngineering
from src.datasource import Datasource
from src.eda import EDA
from util.logger import Logger
import pandas as pd
from constants.ref_cols import keep_cols
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from util.plotter import Plotter


class Runner:
    """
    This class is responsible for running the entire project. It initializes objects for the different components of the project, such as the model, the data source, the EDA, and the logger. It also provides methods for importing data, performing EDA, splitting the data into training and testing sets, training and testing different models, and outputting the results.
    """

    def __init__(self) -> None:
        """
        Initialize objects for the different components of the project.
        """
        self.logger = Logger()
        self.model = Model()
        self.plotter = Plotter()
        self.y = pd.DataFrame()
        self.X = pd.DataFrame()

    def import_data(self, pth):
        """
        Import data from a CSV file.

        Args:
            pth (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: Data from the CSV file.

        Raises:
            Exception: If there is an error importing the data.
        """
        try:
            data = Datasource(pth)
            return data.return_df()
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def perform_eda(self, df):
        """
        Perform EDA on the data and plot some basic statistics and histograms.

        Args:
            df (pandas.DataFrame): The data to perform EDA on.

        Returns:
            pandas.DataFrame: The input data with additional columns added for EDA.

        Raises:
            Exception: If there is an error performing EDA.
        """
        eda = EDA(df)

        # log general stats
        try:
            eda.general_stats()
            eda.plot_churn_hist_distribution()
            eda.plot_customer_hist_distribution()
            eda.plot_marital_status_dist()
            eda.plot_total_trans_density_hist_distribution()
            eda.plot_heatmap_to_check_correllation()
            return eda.return_df()
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def encoder_helper(self, df, category_lst, new_category_lst):
        """
        Helper function to encode categorical variables.

        Args:
            df (pandas.DataFrame): The data to encode.
            category_lst (list): List of columns that contain categorical features.
            new_category_lst (list): List of new columns to create for each categorical feature.

        Returns:
            pandas.DataFrame: The input data with new columns for encoded categorical features.

        Raises:
            Exception: If there is an error encoding the data.
        """
        try:
            self.logger.logging.info("Encoding of categorical variables...")
            fe_helper = FeatureEngineering(df)
            for col, new_col in zip(category_lst, new_category_lst):
                fe_helper.encode_data_col(col, new_col, 'Churn')
            self.logger.logging.info("Encoding complete.")
            return fe_helper.return_df()
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def splitting_the_feature(self, df, response=None):
        """
        Split the data into training and testing sets.

        Args:
            df (pandas.DataFrame): The data to split.
            response (str, optional): The name of the response variable. Defaults to None.

        Returns:
            tuple: A tuple containing the X and y training and testing data.

        Raises:
            Exception: If there is an error splitting the data.
        """
        try:
            self.y = df['Churn']
            self.X[keep_cols] = df[keep_cols]
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.3, random_state=42)
            self.logger.logging.info("Train and test set ready!")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def train_predict_models(self, X_train, X_test, y_train, y_test):
        """
        Train and test different models.

        Args:
            X_train (pandas.DataFrame): The X training data.
            X_test (pandas.DataFrame): The X testing data.
            y_train (pandas.Series): The y training data.
            y_test (pandas.Series): The y testing data.

        Returns:
            tuple: A tuple containing the trained models and their predictions on the testing data.

        Raises:
            Exception: If there is an error training or testing the models.
        """
        return self.model.classification(X_train, X_test, y_train, y_test)

    def output_pred_result(self, X_train, X_test, y_train, y_test):
        """
        Output the prediction results for the training and testing data.

        Args:
            X_train (pandas.DataFrame): The X training data.
            X_test (pandas.DataFrame): The X testing data.
            y_train (pandas.Series): The y training data.
            y_test (pandas.Series): The y testing data.

        Returns:
            None

        Raises:
            Exception: If there is an error outputting the prediction results.
        """
        try:
            y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = self.train_predict_models(
                X_train, X_test, y_train, y_test)
        except Exception as e:
            self.logger.logging.error(repr(e))
            raise Exception(repr(e))

        try:

            self.logger.logging.info('random forest results')
            self.logger.logging.info('test results')
            self.logger.logging.info(
                classification_report(
                    y_test, y_test_preds_rf))
            self.logger.logging.info('train results')
            self.logger.logging.info(
                classification_report(
                    y_train, y_train_preds_rf))

            self.logger.logging.info('logistic regression results')
            self.logger.logging.info('test results')
            self.logger.logging.info(
                classification_report(
                    y_test, y_test_preds_lr))
            self.logger.logging.info('train results')
            self.logger.logging.info(
                classification_report(
                    y_train, y_train_preds_lr))
            self.logger.logging.info("Results displayed successfully")

            self.plotter.mode_classification_plots(
                y_train,
                y_test,
                y_train_preds_lr,
                y_train_preds_rf,
                y_test_preds_lr,
                y_test_preds_rf,
                'matrix_plot')
            self.logger.logging.info("Matrix plot stored successfully")

        except Exception as e:
            self.logger.logging.error(repr(e))
            raise Exception(repr(e))

        try:
            self.model.save_model()
            self.logger.logging.info("Models saved!")
        except Exception as e:
            self.logger.logging.error(repr(e))
            raise Exception(repr(e))

    def model_evaluation_plot(self, X_test, y_test, chartname):
        """
        Plot the ROC curve and other model evaluation metrics.

        Args:
            X_test (pandas.DataFrame): The X testing data.
            y_test (pandas.Series): The y testing data.
            chartname (str): The name of the chart to plot.

        Returns:
            bool: True if the plotting is successful, False otherwise.

        Raises:
            Exception: If there is an error plotting the evaluation metrics.
        """
        try:
            rfc_model, lr_model = self.model.load_model()
            self.plotter.roc_plot(
                X_test, y_test, lr_model, rfc_model, chartname)
            self.logger.logging.info(
                "Models roc plot complete. Check images folder for result!")
            self.plotter.model_explainer(
                X_test, rfc_model, 'model_explainer.png')
            self.logger.logging.info(
                "Model explainer plot complete. Check images folder for result!")
            self.plotter.featureImportance(
                rfc_model, self.X, 'feat_importance')
            self.logger.logging.info(
                "Feature importance plot complete. Check images folder for result!")
            return True
        except Exception as e:
            self.logger.logging.error(repr(e))
            raise Exception(repr(e))
