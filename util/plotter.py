from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Plotter:
    """
    This class is used to create visualizations and save them to disk.

    Args:
        save_eda (str, optional): The directory where EDA charts will be saved. Defaults to "images/eda/".
        save_result (str, optional): The directory where result charts will be saved. Defaults to "images/results/".

    Attributes:
        data (DataFrame): The data used for plotting.
        save_eda (str): The directory where EDA charts will be saved.
        save_result (str): The directory where result charts will be saved.
    """

    def __init__(self, save_eda="images/eda/", save_result="images/results/"):
        self.data = None
        self.save_eda = save_eda
        self.save_result = save_result

    def plot_and_save_histogram(self, data, chartname):
        """
        Plots a histogram of the given data and saves it to disk.

        Args:
            data (DataFrame): The data to plot.
            chartname (str): The name of the chart file.
        """
        plt.figure(figsize=(20, 10))
        data.hist()
        plt.savefig(self.save_eda + chartname)

    def plot_and_save_barchart(self, data, chartname):
        """
        Plots a bar chart of the given data and saves it to disk.

        Args:
            data (DataFrame): The data to plot.
            chartname (str): The name of the chart file.
        """
        plt.figure(figsize=(20, 10))
        data.plot(kind='bar')
        plt.savefig(self.save_eda + chartname)

    def plot_and_save_densityhist(self, data, chartname):
        """
        Plots a density histogram of the given data and saves it to disk.

        Args:
            data (DataFrame): The data to plot.
            chartname (str): The name of the chart file.
        """
        plt.figure(figsize=(20, 10))
        sns.histplot(data, stat='density', kde=True)
        plt.savefig(self.save_eda + chartname)

    def plot_and_save_heatmap(self, data, chartname):
        """
        Plots a heatmap of the correlation between the columns of the given data and saves it to disk.

        Args:
            data (DataFrame): The data to plot.
            chartname (str): The name of the chart file.
        """
        plt.figure(figsize=(20, 15))
        sns.heatmap(
            data.corr(
                numeric_only=True),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig(self.save_eda + chartname)

    def roc_plot(
            self,
            X_test,
            y_test,
            classifier_model1,
            classifier_model2,
            chartname):
        """
        Plots an ROC curve and saves it to disk.

        Args:
            X_test (DataFrame): The features used for testing.
            y_test (Series): The labels used for testing.
            classifier_model1 (sklearn.base.BaseEstimator): The first classifier model.
            classifier_model2 (sklearn.base.BaseEstimator): The second classifier model.
            chartname (str): The name of the chart file.
        """
        lrc_plot = RocCurveDisplay.from_estimator(
            classifier_model1, X_test, y_test)

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = RocCurveDisplay.from_estimator(
            classifier_model2, X_test, y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(self.save_result + chartname + ' lrc_and_rfc')

    def model_explainer(self, X_test, classifer, name):
        """
        Creates a model explainer and saves the SHAP values and summary plot to disk.

        Args:
            X_test (DataFrame): The features used for testing.
            classifer (sklearn.base.BaseEstimator): The classifier model.
            name (str): The name of the chart file.
        """
        plt.figure(figsize=(15, 8))
        explainer = shap.TreeExplainer(classifer)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(self.save_result + name, dpi=700)

    def featureImportance(self, classifier, X, chartname):
        """
        Calculates and plots the feature importances of the given classifier and saves the plot to disk.

        Args:
            classifier (sklearn.base.BaseEstimator): The classifier model.
            X (DataFrame): The features.
            chartname (str): The name of the chart file.
        """
        # Calculate feature importances
        importances = classifier.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X.shape[1]), names, rotation=90)
        plt.savefig(self.save_result + chartname)

    def mode_classification_plots(
            self,
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
            chartname):
        """
        Creates classification plots for the given training and testing data and saves them to disk.

        Args:
            y_train (Series): The training labels.
            y_test (Series): The testing labels.
            y_train_preds_lr (Series): The training predictions for the logistic regression model.
            y_train_preds_rf (Series): The training predictions for the random forest model.
            y_test_preds_lr (Series): The testing predictions for the logistic regression model.
            y_test_preds_rf (Series): The testing predictions for the random forest model.
            chartname (str): The name of the chart file.
        """
        # plt.rc('figure', figsize=(5, 5))
        plt.figure(figsize=(8, 8))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 0.03, str('Random Forest Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.1, str(classification_report(y_test, y_test_preds_rf)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig(self.save_result + chartname + ' rfc')

        # plt.rc('figure', figsize=(5, 5))
        plt.figure(figsize=(8, 8))
        plt.text(0.01, 0.03, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.1, str(classification_report(y_train, y_train_preds_lr)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig(self.save_result + chartname + ' lrc')
