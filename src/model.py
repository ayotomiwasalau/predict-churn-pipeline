from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from util.logger import Logger


class Model:
    """
    This class contains the methods for training and using machine learning models.

    Args:
        lrc_pth (str, optional): The path to the logistic regression model file. Defaults to 'models/logistic_model.pkl'.
        rfc_pth (str, optional): The path to the random forest classifier model file. Defaults to 'models/rfc_model.pkl'.

    Attributes:
        logger (Logger): The logger object used for logging messages.
        lrc (object): The trained logistic regression model.
        cv_rfc (object): The trained random forest classifier model.
        lrc_pth (str): The path to the logistic regression model file.
        rfc_pth (str): The path to the random forest classifier model file.
    """

    def __init__(
            self,
            lrc_pth='models/logistic_model.pkl',
            rfc_pth='models/rfc_model.pkl'):
        self.logger = Logger()
        self.lrc = None
        self.cv_rfc = None
        self.lrc_pth = lrc_pth
        self.rfc_pth = rfc_pth

    def init_classifers(self):
        """
        Initialize the logistic regression and random forest classifiers.

        Returns:
            lrc (object): The trained logistic regression model.
            cv_rfc (object): The trained random forest classifier model.
        """
        try:

            rfc = RandomForestClassifier(random_state=42)
            # Use a different solver if the default 'lbfgs' fails to converge
            # Reference:
            # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
            lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
            param_grid = {
                'n_estimators': [200, 500],
                'max_features': ['log2', 'sqrt'],
                'max_depth': [4, 5, 100],
                'criterion': ['gini', 'entropy']
            }
            cv_rfc = GridSearchCV(
                estimator=rfc,
                param_grid=param_grid,
                cv=5,
                error_score='raise')
            self.logger.logging.info("Parameters initialized!")
            return lrc, cv_rfc
        except Exception as e:
            self.logger.logging.error(
                "Error initializing model parameters " + repr(e))
            raise Exception(repr(e))

    def classification(self, X_train, X_test, y_train, y_test):
        """
        Perform classification using the trained logistic regression and random forest classifiers.

        Args:
            X_train (array-like): The training features.
            X_test (array-like): The testing features.
            y_train (array-like): The training labels.
            y_test (array-like): The testing labels.

        Returns:
            y_train (array-like): The predicted training labels.
            y_test (array-like): The predicted testing labels.
            y_train_preds_lr (array-like): The predicted training labels for the logistic regression model.
            y_train_preds_rf (array-like): The predicted training labels for the random forest classifier model.
            y_test_preds_lr (array-like): The predicted testing labels for the logistic regression model.
            y_test_preds_rf (array-like): The predicted testing labels for the random forest classifier model.
        """
        # grid search

        try:
            self.logger.logging.info("Fitting of model ongoing...")
            lrc, cv_rfc = self.init_classifers()

            cv_rfc.fit(X_train, y_train)
            lrc.fit(X_train, y_train)
            self.logger.logging.info("Fitting completed!")
        except Exception as e:
            self.logger.logging.error(
                "Error fitting model parameters " + repr(e))
            raise Exception(repr(e))

        try:
            y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
            y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
            y_train_preds_lr = lrc.predict(X_train)
            y_test_preds_lr = lrc.predict(X_test)
            self.logger.logging.info("Prediction complete")
            self.cv_rfc = cv_rfc
            self.lrc = lrc
            return y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
        except Exception as e:
            self.logger.logging.error("Error predicting values " + e)
            raise Exception(repr(e))

    def save_model(self):
        """
        Save the trained logistic regression and random forest classifiers to files.

        Returns:
            bool: True if the models were saved successfully, False otherwise.
        """
        try:
            joblib.dump(self.cv_rfc.best_estimator_, self.rfc_pth)
            joblib.dump(self.lrc, self.lrc_pth)
            return True
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))

    def load_model(self):
        """
        Load the trained logistic regression and random forest classifiers from files.

        Returns:
            rfc_model (object): The loaded random forest classifier model.
            lr_model (object): The loaded logistic regression model.
        """
        try:
            rfc_model = joblib.load(self.rfc_pth)
            lr_model = joblib.load(self.lrc_pth)
            return rfc_model, lr_model
        except Exception as e:
            self.logger.logging.error(e)
            raise Exception(repr(e))
