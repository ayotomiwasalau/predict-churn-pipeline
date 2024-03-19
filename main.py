from src.runner import Runner
from constants.ref_cols import cat_columns
from constants.ref_cols import newcat_columns
from util.logger import Logger
import argparse


def main():
    """
    Main function of the pipeline.
    """
    parser = argparse.ArgumentParser(description='ml pipeline')
    parser.add_argument(
        '--filepath',
        type=str,
        help='Input file',
        required=True)
    args = parser.parse_args()
    pth = args.filepath

    app = Runner()
    logger = Logger()

    try:
        """
        Import the data from the given filepath.
        """
        print("Starting ml pipeline================================")
        print("Importing csv file...")
        df = app.import_data(pth)

        """
        Perform exploratory analysis on the data.
        """
        print("Running expolatory analysis on data...")
        df = app.perform_eda(df)
        print("EDA complete! Check images folder for data charts...")

        """
        Encode the categorical features in the data.
        """
        print("Encoding categorical features...")
        df = app.encoder_helper(df, cat_columns, newcat_columns)
        print("Catgorical columns encoded.")

        """
        Split the features into train and test sets.
        """
        print("Spllting features into train and test set...")
        X_train, X_test, y_train, y_test = app.splitting_the_feature(df)
        print("Train and test set ready!")

        """
        Train the model on the training data.
        """
        print("Training of model begins...")
        app.output_pred_result(X_train, X_test, y_train, y_test)
        print("Prediction results on train and test sets for models out. Check logs for details ")

        """
        Plot the evaluation of the model's results.
        """
        print("Plot evaluation of model result...")
        app.model_evaluation_plot(X_test, y_test, 'roc_plot')
        print("Evaluation done. Check images folder for model evaluation plot.")
        print("Ending ml pipeline================================")
    except Exception as e:
        """
        Log any errors that occur during the pipeline.
        """
        logger.logging.error(repr(e))
        print("Exiting...")
        exit("Check logs for errors")

if __name__ == '__main__':
    main()
