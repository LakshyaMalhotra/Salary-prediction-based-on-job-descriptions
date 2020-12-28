__author__ = "Lakshya Malhotra"
__copyright__ = "Copyright (c) 2020 Lakshya Malhotra"

# Library imports
import os
from typing import Union, Callable
from argparse import ArgumentParser

import numpy as np
import pandas as pd


# create command line args parser
def build_argparser():
    """
    Parse command line arguments
    :return: ArgumentParser
    """
    parser = ArgumentParser(
        description="Select the model and various hyperparameters."
    )
    parser.add_argument('--model', required=True, help='Model to be used.')

    return parser


# define the baseline model
def baseline_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str = "salary",
    index: str = "jobId",
    key: str = None,
) -> pd.DataFrame:
    """
    Baseline model for predicting the salaries. It is based on taking the
    average value of the salaries for a given feature.

    :param train_df: training dataset
    :param valid_df: validation dataset
    :param target: target name
    :param index: unique identifier for each row
    :param key: feature on which average salaries are to be calculated
    :return: dataframe containing predictions
    """
    if key is None:
        raise TypeError("key cannot be NoneType")

    target_mean_df = pd.DataFrame(train_df.groupby([key])[target].mean())
    target_mean_df.columns = ["predictions"]
    pred_df = valid_df[[index, key, target]].set_index(key)
    pred_df = pred_df.join(target_mean_df).set_index(index)

    return pred_df


# define the loss function
def MSE(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Calculate the mean square error.

    :param y_true: target values
    :param y_pred: predictions from the model
    :return: Root mean square error
    """
    avg_diff = np.mean((y_true - y_pred) ** 2)
    return avg_diff


# display results
def show_results(
    fold: int, loss: Union[np.ndarray, float], model: Union[str, Callable]
) -> None:
    print(f"Model: {model}, fold: {fold}")
    print(f"Average MSE: {loss}")


# run the baseline model
def run_baseline(
    df: pd.DataFrame,
    fold: int,
    model: Union[str, Callable] = "baseline",
    key: str = "jobType",
) -> None:
    """
    Runs the baseline model on the data for a given cross validation fold and make
    predictions.

    :param key: optional argument to be used only when baseline model is used
    :param df: original dataframe containing the data
    :param fold: fold of cross validation
    :param model: model name, could be a string or a function
    :return: None
    """
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    pred_df = baseline_model(train_df, valid_df, key=key)
    print(f"Making predictions with baseline model with {key}")
    y_true, y_pred = pred_df.salary.values, pred_df.predictions.values
    assert len(y_true) == len(y_pred)

    loss = MSE(y_true, y_pred)
    show_results(fold, loss, model)


if __name__ == "__main__":
    path = "data/"
    data = pd.read_csv(os.path.join(path, "train_folds.csv"))

    for i in range(10):
        run_baseline(data, fold=i)
