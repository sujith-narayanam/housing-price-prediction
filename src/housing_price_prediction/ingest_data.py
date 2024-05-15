"""
A script (ingest_data.py)to download and create training and validation datasets.
The script should accept the output folder/file path as an user argument.
"""
import argparse
import os
import tarfile

import mlflow
import numpy as np
import pandas as pd
from six.moves import urllib  # pyright: ignore
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from mle_lib import api
from ta_lib.core.api import DEFAULT_DATA_BASE_PATH
from mle_lib.api import config


def fetch_housing_data(housing_url, housing_path):
    """
    Downloads Housing data from the specified URL and saves it in the path provided.
    Extracts the csv file from .tgz compression.

    Parameters:
    -----------
    housing_url
        String, URL to fetch the data from
    housing_path
        string, Path to save the fetched data
    """
    # create directory
    os.makedirs(housing_path, exist_ok=True)
    # construct target path
    api.logger.info("Starting Data Downloading")
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # pull data
    api.logger.info("Extracting Data from tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    # extract from .tgz
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    # close connection
    housing_tgz.close()


def load_housing_data(housing_url, housing_path):
    """
    Returns the data from specified URl, and saves it in given path

    Parameters:
    -----------
    housing_url
        String, URL to fetch the data from
    housing_path
        string, Path to save the fetched data

    Returns:
    --------
    DataFrame
        Data from the specified URL
    """
    # fetch data
    fetch_housing_data(housing_url, housing_path)
    # read and return
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def pull_housing_data(output_path=DEFAULT_DATA_BASE_PATH, context=None):
    """
    Driver code to pull the data

    Parameters:
    -----------
    output_path:
        String, Path to save the downloaded file
    """
    # get credentials
    HOUSING_PATH = os.path.join(output_path, "raw")
    HOUSING_URL = config.HOUSING_URL
    df = load_housing_data(
        housing_url=HOUSING_URL,
        housing_path=HOUSING_PATH,
    )
    mlflow.log_param("Housing Data Size", df.shape)
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    api.write(df, os.path.join(HOUSING_PATH, "housing.csv"))

    api.logger.info("Train test splitting")
    # train test split
    train_set, test_set = train_test_split(
        df, test_size=config.test_size, random_state=42
    )
    mlflow.log_param("Housing Train set Size", train_set.shape)
    mlflow.log_param("Housing Test set Size", test_set.shape)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    mlflow.log_param("Stratified Train set Size", strat_train_set.shape)
    mlflow.log_param("Stratified Test set Size", strat_test_set.shape)

    # save train and test data
    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    api.write(train_set, os.path.join(output_path, "train", "housing_train.csv"))
    api.write(
        strat_train_set,
        os.path.join(output_path, "train", "housing_train_income_stratified.csv"),
    )

    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    api.write(test_set, os.path.join(output_path, "test", "housing_test.csv"))
    api.write(
        strat_test_set,
        os.path.join(output_path, "test", "housing_test_income_stratified.csv"),
    )
    api.logger.info("Data Downloading and splitting is done")
    api.logger.info(f"Data saved at: {output_path}")


if __name__ == "__main__":
    # get output path from command line args

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", help="Path where data should be saved")
    args = parser.parse_args()
    api.logger.info("Data Downloading and splitting is Starting")
    if args.output_path is not None:
        pull_housing_data(args.output_path)
    else:
        pull_housing_data()
