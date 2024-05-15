"""
Common utilities function that can be used across project to maintain uniformity and enhance re-usability.
"""
import os
import pickle

import mlflow
import pandas as pd


def read(path):
    """
    Function to read files of various formats

    Parameters:
    -----------
    path:
        String, Path of file to read
    """
    _type = path.split(".")[-1]

    if _type == "csv":
        return pd.read_csv(path)
    elif _type == "pkl":
        return pickle.load(open(path, "rb"))
    else:
        pass


def write(data, path):
    """
    Function to write files of various formats

    Parameters:
    -----------
    data:
        data to write
    path:
        String, Path where file should be written
    """
    _type = path.split(".")[-1]

    if _type == "csv":
        data.to_csv(path, index=False)
    elif _type == "pkl":
        pickle.dump(data, open(path, "wb"))
    else:
        pass


def get_new_exp_num():
    """
    Generates new experiment id for mlflow tracker automatically
    """
    try:
        all_exps = [exp.experiment_id for exp in MlflowClient().list_experiments()]
        ids = [int(i.split("_")[-1]) for i in all_exps]
        ids.sort()
        return ids[-1] + 1
    except:
        return 1
