from src.runner import Runner
from constants.ref_cols import cat_columns
from constants.ref_cols import newcat_columns
from util.logger import Logger
import argparse


def main():
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
        print("Importing csv file...")
        df = app.import_data(pth)

        print("Running expolatory analysis on data...")
        df = app.perform_eda(df)
        print("EDA complete! Check images folder for data charts...")

        print("Encoding categorical features...")
        df = app.encoder_helper(df, cat_columns, newcat_columns)
        print("Catgorical columns encoded.")

        print("Spllting features into train and test set...")
        X_train, X_test, y_train, y_test = app.splitting_the_feature(df)
        print("Train and test set ready!")

        print("Training of model begins...")
        app.output_pred_result(X_train, X_test, y_train, y_test)
        print("Prediction results on train and test sets for models out. Check logs for details ")

        print("Plot evaluation of model result...")
        app.model_evaluation_plot(X_test, y_test, 'roc_plot')
        print("Evaluation done. Check images folder for model evaluation plot.")
    except Exception as e:
        logger.logging.error(repr(e))
        print("Exiting...")
        exit("Check logs for errors")


if __name__ == '__main__':
    main()
