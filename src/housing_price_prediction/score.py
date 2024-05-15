"""
A script (score.py) to score the model(s).
The script accepts arguments for model folder, and dataset folder.
"""
import argparse
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mle_lib.api import (
    logger,
    read,
    write,
)
from train import process_data
from ta_lib.core.api import (
    DEFAULT_DATA_BASE_PATH,
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_RESULTS_PATH,
)


def score_models(models_path=DEFAULT_ARTIFACTS_PATH, data_path=DEFAULT_DATA_BASE_PATH):
    """
    Read models and generate score

    Parameters:
    -----------
    models_path:
        Path for trained models
    data_path:
        path for train and test data
    """
    # get train and test data
    housing_prepared, housing_labels, X_test, y_test = process_data(
        data_path, return_train=True
    )

    metrics = []

    logger.info("loading linear regression and compuite metrics")
    lin_metr = {}
    lin_metr["model"] = "Linear Regression"
    lin_reg = read(os.path.join(models_path, "Linear_Regression.pkl"))
    housing_predictions = lin_reg.predict(housing_prepared)
    housing_predictions_test = lin_reg.predict(X_test)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_metr["train_mse"] = lin_mse
    lin_mse_test = mean_squared_error(y_test, housing_predictions_test)
    lin_metr["test_mse"] = lin_mse_test
    lin_rmse = np.sqrt(lin_mse)
    lin_metr["train_rmse"] = lin_rmse
    lin_rmse_test = np.sqrt(lin_mse_test)
    lin_metr["test_rmse"] = lin_rmse_test
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    lin_metr["train_mae"] = lin_mae
    lin_mae_test = mean_absolute_error(y_test, housing_predictions_test)
    lin_metr["test_mae"] = lin_mae_test
    metrics.append(pd.DataFrame(lin_metr, index=[0]))
    del lin_metr["model"]
    mlflow.log_metrics(lin_metr, step=1)

    logger.info("loading Decision tree regressor and compute metrics")
    tree_metr = {}
    tree_metr["model"] = "Decision Tree Regressor"
    tree_reg = read(os.path.join(models_path, "Decision_Tree_Regressor.pkl"))
    housing_predictions = tree_reg.predict(housing_prepared)
    housing_predictions_test = tree_reg.predict(X_test)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_metr["train_mse"] = tree_mse
    tree_mse_test = mean_squared_error(y_test, housing_predictions_test)
    tree_metr["test_mse"] = tree_mse_test
    tree_rmse = np.sqrt(tree_mse)
    tree_metr["train_rmse"] = tree_rmse
    tree_rmse_test = np.sqrt(tree_mse_test)
    tree_metr["test_rmse"] = tree_rmse_test
    tree_mae = mean_absolute_error(housing_labels, housing_predictions)
    tree_metr["train_mae"] = tree_mae
    tree_mae_test = mean_absolute_error(y_test, housing_predictions_test)
    tree_metr["test_mae"] = tree_mae_test
    metrics.append(pd.DataFrame(tree_metr, index=[1]))
    del tree_metr["model"]
    mlflow.log_metrics(tree_metr, step=2)

    logger.info("loading Random Search best Random Forest and compuite metrics")
    rand_search_metr = {}
    rand_search_metr["model"] = "Random Search Random Forest"
    rand_search_reg = read(
        os.path.join(models_path, "random_search_best_estimator_forest.pkl")
    )
    housing_predictions = rand_search_reg.predict(housing_prepared)
    housing_predictions_test = rand_search_reg.predict(X_test)
    rand_search_mse = mean_squared_error(housing_labels, housing_predictions)
    rand_search_metr["train_mse"] = rand_search_mse
    rand_search_mse_test = mean_squared_error(y_test, housing_predictions_test)
    rand_search_metr["test_mse"] = rand_search_mse_test
    rand_search_rmse = np.sqrt(rand_search_mse)
    rand_search_metr["train_rmse"] = rand_search_rmse
    rand_search_rmse_test = np.sqrt(rand_search_mse_test)
    rand_search_metr["test_rmse"] = rand_search_rmse_test
    rand_search_mae = mean_absolute_error(housing_labels, housing_predictions)
    rand_search_metr["train_mae"] = rand_search_mae
    rand_search_mae_test = mean_absolute_error(y_test, housing_predictions_test)
    rand_search_metr["test_mae"] = rand_search_mae_test
    metrics.append(pd.DataFrame(rand_search_metr, index=[2]))
    del rand_search_metr["model"]
    mlflow.log_metrics(rand_search_metr, step=3)

    logger.info("loading Grid Search best Random Forest and compuite metrics")
    grid_search_metr = {}
    grid_search_metr["model"] = "Grid Search Random Forest"
    grid_search_reg = read(
        os.path.join(models_path, "grid_search_best_estimator_forest.pkl")
    )
    housing_predictions = grid_search_reg.predict(housing_prepared)
    housing_predictions_test = grid_search_reg.predict(X_test)
    grid_search_mse = mean_squared_error(housing_labels, housing_predictions)
    grid_search_metr["train_mse"] = grid_search_mse
    grid_search_mse_test = mean_squared_error(y_test, housing_predictions_test)
    grid_search_metr["test_mse"] = grid_search_mse_test
    grid_search_rmse = np.sqrt(grid_search_mse)
    grid_search_metr["train_rmse"] = grid_search_rmse
    grid_search_rmse_test = np.sqrt(grid_search_mse_test)
    grid_search_metr["test_rmse"] = grid_search_rmse_test
    grid_search_mae = mean_absolute_error(housing_labels, housing_predictions)
    grid_search_metr["train_mae"] = grid_search_mae
    grid_search_mae_test = mean_absolute_error(y_test, housing_predictions_test)
    grid_search_metr["test_mae"] = grid_search_mae_test
    metrics.append(pd.DataFrame(grid_search_metr, index=[3]))
    del grid_search_metr["model"]
    mlflow.log_metrics(grid_search_metr, step=4)

    metrics = pd.concat(metrics)

    write(metrics, os.path.join(DEFAULT_RESULTS_PATH, "metrics.csv"))
    mlflow.log_artifact(os.path.join(DEFAULT_RESULTS_PATH, "metrics.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", help="Models Path")
    parser.add_argument("--data_path", help="Data Path")
    args = parser.parse_args()

    if (args.models_path is not None) and (args.data_path is not None):
        score_models(args.models_path, args.data_path)
    else:
        score_models()
