# Predict Customer Churn

- Project **Predict Customer Churn** - ML DevOps Engineering

## Project Description
This project automates the data science workflow for predicting customer churn given a dataset

Create a virtual python environment - [link](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/#)

`install` all the packages in the requirement.txt file (python 3.8 and above)

## Files and data description
`logs/` contain the logs for both the main and test
`models/` contain the models generated from training on the data
`images/` contains both plots for the exploratory data analysis and the plot for the result evaluation
`tests/` contains the unit tests as well well as an `images` and `models` directory for the test runner

## Running Files

To initiate the script with the filepath flag for the path to the csv datafile, run the following command
```
python main.py --filepath 'data/bank_data.csv'

```

Run the test with this command
```
python test_runner.py 

```

Navigate to `logs`, `models`, and  `images` to view the outputs.



