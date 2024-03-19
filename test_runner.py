from tests.unit_tests import UnitTest


if __name__ == "__main__":
    print("Running tests...")

    test = UnitTest()

    test.test_should_import_csv_file()
    test.test_should_plot_churn_distribution()
    test.test_should_plot_customer_hist_distribution()
    test.test_should_plot_marital_status_dist()
    test.test_should_plot_total_trans_density_hist_distribution()
    test.test_should_plot_heatmap_to_check_correllation()
    test.test_should_encode_cat_features()
    test.test_should_split_features()
    test.test_classification_fitting_and_predicting()
    test.test_saving_model()
    test.test_loading_model()
    test.test_model_evaluation_plot()

    print("Complete.")
    print(f"{test.passed} passed, {test.failed} failed")
    print(f"Total - {test.passed + test.failed}")
    print("Check logs for details.")
