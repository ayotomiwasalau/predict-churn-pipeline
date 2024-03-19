from src.model import Model
from src.datasource import Datasource
from src.eda import EDA
from src.runner import Runner
from util.logger import Logger
from util.plotter import Plotter
from constants.ref_cols import cat_columns, newcat_columns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class UnitTest:

    def __init__(self):
        """
        Initialize the test class.
        """
        self.logger = Logger(logname='logs/test.log')
        self.plotter = Plotter("tests/images/eda/", "tests/images/results/")
        self.model = Model(
            'tests/models/logistic_model.pkl',
            'tests/models/rfc_model.pkl')
        self.app = Runner()
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.passed = 0
        self.failed = 0

    def test_should_import_csv_file(self):
        """
        Test if the CSV file can be imported.
        """
        try:
            self.logger.logging.info('Running test_should_import_csv_file...')
            pth = 'data/bank_test_data.csv'
            df = self.app.import_data(pth)
            assert df.empty == False
            self.logger.logging.info('test_should_import_csv_file passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_should_import_csv_file failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_should_plot_churn_distribution(self):
        """
        Test if the churn distribution can be plotted.
        """
        try:
            self.logger.logging.info(
                'Running test_should_plot_churn_distribution...')
            pth = 'data/bank_test_data.csv'
            eda = EDA(self.app.import_data(pth))
            eda.plotter = self.plotter
            res = eda.plot_churn_hist_distribution()
            assert res
            self.logger.logging.info(
                'test_should_plot_churn_distribution passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_should_plot_churn_distribution failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_should_plot_customer_hist_distribution(self):
        """
        Test if the customer distribution can be plotted.
        """
        try:
            self.logger.logging.info(
                'Running test_should_plot_customer_hist_distribution...')
            pth = 'data/bank_test_data.csv'
            eda = EDA(self.app.import_data(pth))
            eda.plotter = self.plotter
            res = eda.plot_customer_hist_distribution()
            assert res
            self.logger.logging.info(
                'test_should_plot_customer_hist_distribution passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_should_plot_customer_hist_distribution failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_should_plot_marital_status_dist(self):
        """
        Test if the marital status distribution can be plotted.
        """
        try:
            self.logger.logging.info(
                'Running test_should_plot_marital_status_dist...')
            pth = 'data/bank_test_data.csv'
            eda = EDA(self.app.import_data(pth))
            eda.plotter = self.plotter
            res = eda.plot_marital_status_dist()
            assert res
            self.logger.logging.info(
                'test_should_plot_marital_status_dist passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_should_plot_marital_status_dist failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_should_plot_total_trans_density_hist_distribution(self):
        """
        Test if the total transaction density histogram can be plotted.
        """
        try:
            self.logger.logging.info(
                'Running test_should_plot_total_trans_density_hist_distribution...')
            pth = 'data/bank_test_data.csv'
            eda = EDA(self.app.import_data(pth))
            eda.plotter = self.plotter
            res = eda.plot_total_trans_density_hist_distribution()
            assert res
            self.logger.logging.info(
                'test_should_plot_total_trans_density_hist_distribution passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_should_plot_total_trans_density_hist_distribution failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_should_plot_heatmap_to_check_correllation(self):
        """
        Test if the heatmap to check correlations can be plotted.
        """
        try:
            self.logger.logging.info(
                'Running test_should_plot_heatmap_to_check_correllation...')
            pth = 'data/bank_test_data.csv'
            eda = EDA(self.app.import_data(pth))
            eda.plotter = self.plotter
            res = eda.plot_heatmap_to_check_correllation()
            assert res
            self.logger.logging.info(
                'test_should_plot_heatmap_to_check_correllation passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_plot_heatmap_to_check_correllation failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_should_encode_cat_features(self):
        """
        Test if the categorical features can be encoded.
        """
        try:
            self.logger.logging.info(
                'Running test_should_encode_cat_features...')
            pth = 'data/bank_test_data.csv'
            data = self.app.import_data(pth)
            eda = EDA(data)
            eda.plotter = self.plotter
            df = self.app.encoder_helper(
                eda.return_df(), cat_columns, newcat_columns)
            assert df.empty == False
            self.logger.logging.info('test_should_encode_cat_features passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_should_encode_cat_features failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_should_split_features(self):
        """
        Test if the features can be splitted.
        """
        try:
            self.logger.logging.info('Running test_should_split_features...')
            pth = 'data/bank_test_data.csv'
            data = self.app.import_data(pth)
            eda = EDA(data)
            eda.plotter = self.plotter
            df = self.app.encoder_helper(
                eda.return_df(), cat_columns, newcat_columns)
            X_train, X_test, y_train, y_test = self.app.splitting_the_feature(
                df)
            assert X_train.empty == False
            assert X_test.empty == False
            assert y_train.empty == False
            assert y_test.empty == False
            self.logger.logging.info('test_should_split_features passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_should_split_features failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_classification_fitting_and_predicting(self):
        try:
            self.logger.logging.info(
                'Running test_classification_fitting_and_predicting...')
            pth = 'data/bank_test_data.csv'
            data = self.app.import_data(pth)
            eda = EDA(data)
            eda.plotter = self.plotter
            df = self.app.encoder_helper(
                eda.return_df(), cat_columns, newcat_columns)
            X_train, X_test, y_train, y_test = self.app.splitting_the_feature(
                df)
            y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = self.model.classification(
                X_train, X_test, y_train, y_test)
            self.X_test = X_test
            self.y_test = y_test
            assert y_train_preds_lr.size != 0
            assert y_train_preds_rf.size != 0
            assert y_test_preds_lr.size != 0
            assert y_test_preds_rf.size != 0
            self.logger.logging.info(
                'test_classification_fitting_and_predicting passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(
                f'test_classification_fitting_and_predicting failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_saving_model(self):
        try:
            self.logger.logging.info('Running test_saving_model...')
            res = self.model.save_model()
            assert res
            self.logger.logging.info('test_saving_model passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(f'test_saving_model failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_loading_model(self):
        try:
            self.logger.logging.info('Running test_loading_model...')
            rfc_model, lr_model = self.model.load_model()
            assert isinstance(rfc_model, RandomForestClassifier)
            assert isinstance(lr_model, LogisticRegression)
            self.logger.logging.info('test_loading_model passed')
            self.passed += 1
            self.logger.logging.info('')
        except Exception as e:
            self.logger.logging.error(f'test_loading_model failed {repr(e)}')
            self.failed += 1
            self.logger.logging.info('')

    def test_model_evaluation_plot(self):
        try:
            self.logger.logging.info('Running test_model_evaluation_plot...')
            self.app.plotter = self.plotter
            res = self.app.model_evaluation_plot(
                self.X_test, self.y_test, 'unit_test_roc')
            assert res
            self.logger.logging.info('test_model_evaluation_plot passed')
            self.passed += 1
        except Exception as e:
            self.logger.logging.error(
                f'test_model_evaluation_plot failed {repr(e)}')
            self.failed += 1
