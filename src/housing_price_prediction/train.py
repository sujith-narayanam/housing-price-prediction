"""
A script (train.py) to train the model(s).
The script accepts arguments for input (dataset) and output folders (model pickles)
"""
import argparse
import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from data_transformer import data_transformer
from mle_lib.api import (
    logger,
    read,
    write,
)
from ta_lib.core.api import (
    DEFAULT_DATA_BASE_PATH,
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_RESULTS_PATH,
)


def income_cat_proportions(data):
    """
    Computes the proportion of all income categories in the data.

    Parameters:
    -----------
    data:
        pd.DataFrame, Data with income_categories
    """
    return data["income_cat"].value_counts() / len(data)


def load_data(input_path=DEFAULT_DATA_BASE_PATH):
    """
    Loads the data from given path

    Parameters:
    -----------
    input_path:
        Path where data is stored
    """
    # declaring global variables
    global housing_raw, train_set, test_set, strat_train_set, strat_test_set
    raw_path = os.path.join(input_path, "raw", "housing.csv")
    train_set_path = os.path.join(input_path, "train", "housing_train.csv")
    test_set_path = os.path.join(input_path, "test", "housing_test.csv")
    strat_train_set_path = os.path.join(
        input_path, "train", "housing_train_income_stratified.csv"
    )
    strat_test_set_path = os.path.join(
        input_path, "test", "housing_test_income_stratified.csv"
    )

    # reading data
    logger.info("Reading Data files")
    housing_raw = read(raw_path)
    train_set = read(train_set_path)
    test_set = read(test_set_path)
    strat_train_set = read(strat_train_set_path)
    strat_test_set = read(strat_test_set_path)


def process_data(input_path=DEFAULT_DATA_BASE_PATH, return_train=False):
    """
    Process and ready the data for training and testing

    Parameters:
    -----------
    input_path:
        Path where data is stored

    Returns:
    --------
    Features and target data of train and test sets

    """
    global housing_labels, housing_prepared
    load_data(input_path)
    logger.info("Data Loaded")
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing_raw),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    logger.info("Processing train data")

    os.makedirs(DEFAULT_RESULTS_PATH, exist_ok=True)

    housing.plot(kind="scatter", x="longitude", y="latitude", backend="matplotlib")
    plt.savefig(os.path.join(DEFAULT_RESULTS_PATH, "long_vs_lat.png"))
    plt.close()

    housing.plot(
        kind="scatter", x="longitude", y="latitude", alpha=0.1, backend="matplotlib"
    )
    plt.savefig(os.path.join(DEFAULT_RESULTS_PATH, "long_vs_lat_transparent.png"))
    plt.close()

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    logger.info("Train data Imputing")

    cat_columns = ["ocean_proximity"]
    num_columns = sorted(list(set(housing.columns).difference(cat_columns)))

    transformer = data_transformer(num_columns, cat_columns)
    transformer.fit(housing)
    housing_prepared = transformer.transform(housing)

    logger.info("Processing test data")
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = transformer.transform(X_test)

    if not return_train:
        return X_test_prepared, y_test
    else:
        return housing_prepared, housing_labels, X_test_prepared, y_test


def train_models(output_path=DEFAULT_ARTIFACTS_PATH):
    """
    Trains models and saves the objects

    Parameters:
    ----------
    output_path:
        Path to save the model objects
    """
    # training linear regression
    logger.info("Training Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    filename_lin = os.path.join(output_path, "Linear_Regression.pkl")
    write(lin_reg, filename_lin)
    mlflow.log_artifact(filename_lin)

    # training decision tree
    logger.info("Training Decision Tree Regressor")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    filename_tree = os.path.join(output_path, "Decision_Tree_Regressor.pkl")
    write(tree_reg, filename_tree)
    mlflow.log_artifact(filename_tree)

    # random search CV
    logger.info("Training Random forest and tuning parameters using random search")
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    filename_random = os.path.join(
        output_path, "random_search_best_estimator_forest.pkl"
    )
    write(rnd_search.best_estimator_, filename_random)
    mlflow.log_artifact(filename_random)

    feature_importances = rnd_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    write(
        pd.DataFrame(feature_importances),
        os.path.join(DEFAULT_RESULTS_PATH, "random_search_feature_importances.csv"),
    )
    mlflow.log_artifact(
        os.path.join(DEFAULT_RESULTS_PATH, "random_search_feature_importances.csv")
    )

    logger.info("Training Random forest and tuning parameters using grid search")
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)
    filename_grid = os.path.join(output_path, "grid_search_best_estimator_forest.pkl")
    write(grid_search.best_estimator_, filename_grid)
    mlflow.log_artifact(filename_grid)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    write(
        pd.DataFrame(feature_importances),
        os.path.join(DEFAULT_RESULTS_PATH, "grid_search_feature_importances.csv"),
    )
    mlflow.log_artifact(
        os.path.join(DEFAULT_RESULTS_PATH, "grid_search_feature_importances.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Data Path")
    parser.add_argument("--output_path", help="Path to save model objects")
    args = parser.parse_args()

    if args.input_path is not None:
        X_test, y_test = process_data(args.input_path)
    else:
        X_test, y_test = process_data()

    if args.output_path is not None:
        train_models(args.output_path)
    else:
        train_models()
