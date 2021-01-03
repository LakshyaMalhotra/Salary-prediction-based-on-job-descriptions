__author__ = "Lakshya Malhotra"
__copyright__ = "Copyright (c) 2020 Lakshya Malhotra"

# Library imports
import os
from typing import Union, Callable
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import pipeline
import lightgbm as lgb
from sklearn import decomposition
import category_encoders as ce


# create command line args parser
def build_argparser():
    """
    Parse command line arguments
    :return: ArgumentParser
    """
    parser = ArgumentParser(
        description="Select the model and various " "hyperparameters."
    )
    parser.add_argument("--model", required=True, help="Model to be used.")

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
    loss: Union[np.ndarray, float], model: Union[str, Callable]
) -> None:
    print(f"Model: {model}")
    print(f"Average MSE: {np.mean(loss)}")
    print(f"Average std: {np.std(loss)}")


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
    show_results(loss, model)


# utility function to return the score
def scorer(score_fn: Callable) -> Callable:
    """
    Returns the estimated value of the scoring function.

    :param score_fn: scoring function (can be a loss or accuracy or any other
    function in `sklearn.metrics`) :return: function giving the score
    """
    # `greater_is_better=False` for MSE, RMSE, etc.
    return metrics.make_scorer(score_fn, greater_is_better=False)


# perform some feature engineering
def target_encode(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, cat_vars: list, target: str,
) -> tuple(pd.DataFrame, pd.DataFrame):
    te = ce.TargetEncoder(verbose=10, cols=cat_vars, smoothing=0.5)
    train_df_cat_vars = te.fit_transform(train_df[cat_vars], train_df[target])
    valid_df_cat_vars = te.transform(valid_df[cat_vars])
    train_df_cat_vars.columns = ["meanSalary_per_" + col for col in cat_vars]
    valid_df_cat_vars.columns = ["meanSalary_per_" + col for col in cat_vars]

    return train_df_cat_vars, valid_df_cat_vars


# calculate loss for each model
def train(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    criteria: Callable,
    n_folds: int = 10,
) -> float:
    """
    Find the error for each cross-validation fold.

    :param model: Estimator from sklearn
    :param X: Training data containing all the features
    :param y: Labels
    :param criteria: Scorer from sklearn
    :param n_folds: number of cv folds
    :return: MSE for each fold
    """
    kf = model_selection.KFold(
        n_splits=n_folds, shuffle=True, random_state=23
    ).get_n_splits(X)

    error = model_selection.cross_val_score(
        model, X, y, scoring=criteria, cv=kf, n_jobs=-1, verbose=5
    )

    return -1 * error


def get_feature_df(
    df: pd.DataFrame, cat_vars: list = None, num_vars: list = None
) -> pd.DataFrame:
    """
    One-hot encoding the categorical variables and return the feature dataframe

    :param df: Original raw dataframe
    :param cat_vars: Categorical variables
    :param num_vars: Numeric variables
    :return: feature dataframe
    """
    for cat in cat_vars:
        df[cat] = df[cat].astype("category")
        df[cat] = df[cat].cat.codes
    cat_df = df[cat_vars]
    num_df = df[num_vars].apply(pd.to_numeric)
    return pd.concat([cat_df, num_df], axis=1)


if __name__ == "__main__":
    args = build_argparser().parse_args()

    path = "data/"
    print("Reading data...")
    data = pd.read_csv(os.path.join(path, "train_folds.csv"))

    if args.model == "baseline":
        for i in range(10):
            run_baseline(data, fold=i)

    else:

        train_features = pd.read_csv(os.path.join(path, "train_features.csv"))
        train_targets = pd.read_csv(os.path.join(path, "train_salaries.csv"))

        train_df = pd.concat([train_features, train_targets], axis=1)
        train_df.drop_duplicates(subset="jobId", inplace=True)
        train_df = train_df[train_df.salary > 0]

        cat_cols = ["companyId", "jobType", "degree", "major", "industry"]
        num_cols = ["yearsExperience", "milesFromMetropolis"]

        target = train_df.salary
        feature_df = pd.DataFrame(
            {
                "meanSalary": train_df.groupby(cat_cols)["salary"].transform(
                    "mean"
                )
            }
        )
        feature_df["minSalary"] = train_df.groupby(cat_cols)[
            "salary"
        ].transform("min")
        feature_df["maxSalary"] = train_df.groupby(cat_cols)[
            "salary"
        ].transform("max")
        feature_df["stdSalary"] = train_df.groupby(cat_cols)[
            "salary"
        ].transform("std")
        feature_df["medianSalary"] = train_df.groupby(cat_cols)[
            "salary"
        ].transform("median")
        # feature_df = pd.DataFrame(
        #     {"meanExp": train_df.groupby(cat_cols)["yearsExperience"].mean()}
        # )
        feature_df["meanExp"] = train_df.groupby(cat_cols)[
            "yearsExperience"
        ].transform("mean")
        feature_df["minExp"] = train_df.groupby(cat_cols)[
            "yearsExperience"
        ].transform("min")
        feature_df["maxExp"] = train_df.groupby(cat_cols)[
            "yearsExperience"
        ].transform("max")
        feature_df["stdExp"] = train_df.groupby(cat_cols)[
            "yearsExperience"
        ].transform("std")
        feature_df["medianExp"] = train_df.groupby(cat_cols)[
            "yearsExperience"
        ].transform("median")
        feature_df["minMiles"] = train_df.groupby(cat_cols)[
            "milesFromMetropolis"
        ].transform("min")
        feature_df["maxMiles"] = train_df.groupby(cat_cols)[
            "milesFromMetropolis"
        ].transform("max")
        feature_df["stdMiles"] = train_df.groupby(cat_cols)[
            "milesFromMetropolis"
        ].transform("std")
        feature_df["medianMiles"] = train_df.groupby(cat_cols)[
            "milesFromMetropolis"
        ].transform("median")
        train_df["milesYearsExp"] = (
            train_df["milesFromMetropolis"] / train_df["yearsExperience"]
        )
        # # feature_df["minSalary"] = train_df.groupby(cat_cols)["salary"].min()
        # # feature_df["maxSalary"] = train_df.groupby(cat_cols)["salary"].max()
        # # feature_df["stdSalary"] = train_df.groupby(cat_cols)["salary"].std()
        # # feature_df["medianSalary"] = train_df.groupby(cat_cols)[
        # #     "salary"
        # # ].median()
        # # # feature_df = pd.DataFrame(
        # # #     {"meanExp": train_df.groupby(cat_cols)["yearsExperience"].mean()}
        # # # )
        # # feature_df["meanExp"] = train_df.groupby(cat_cols)[
        # #     "yearsExperience"
        # # ].mean()
        # # feature_df["minExp"] = train_df.groupby(cat_cols)[
        # #     "yearsExperience"
        # # ].min()
        # # feature_df["maxExp"] = train_df.groupby(cat_cols)[
        # #     "yearsExperience"
        # # ].max()
        # # feature_df["stdExp"] = train_df.groupby(cat_cols)[
        # #     "yearsExperience"
        # # ].std()
        # # feature_df["medianExp"] = train_df.groupby(cat_cols)[
        # #     "yearsExperience"
        # # ].median()
        # # feature_df["minMiles"] = train_df.groupby(cat_cols)[
        # #     "milesFromMetropolis"
        # # ].min()
        # # feature_df["maxMiles"] = train_df.groupby(cat_cols)[
        # #     "milesFromMetropolis"
        # # ].max()
        # # feature_df["stdMiles"] = train_df.groupby(cat_cols)[
        # #     "milesFromMetropolis"
        # # ].std()
        # # feature_df["medianMiles"] = train_df.groupby(cat_cols)[
        # #     "milesFromMetropolis"
        # # ].median()
        # print(feature_df.head())
        # print(feature_df.shape)
        # # train_df = pd.merge(
        # #     left=train_df, right=feature_df, how="left", on=cat_cols
        # # )
        # train_df = pd.concat([train_df, feature_df], axis=1)
        # train_df.fillna(0, inplace=True)
        # print(train_df.head())
        # print(train_df.shape)

        # for col in cat_cols:
        #     le = preprocessing.LabelEncoder()
        #     le.fit(train_df[col])
        #     train_df[col] = le.transform(train_df[col])
        # features = train_df.columns
        # features = [col for col in features if col not in ["jobId", "salary"]]
        # print(features)
        # print(train_df[features].head())

        # X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
        #     train_df[features], target, test_size=0.2, random_state=23
        # )
        # lgbm = lgb.LGBMRegressor(n_jobs=-1)
        # rf = ensemble.RandomForestRegressor(
        #     n_estimators=60,
        #     n_jobs=-1,
        #     max_depth=15,
        #     min_samples_split=80,
        #     max_features=8,
        #     verbose=1,
        # )
        # # gbm = ensemble.GradientBoostingRegressor(
        # #     n_estimators=40, max_depth=7, loss="ls", verbose=1
        # # )
        # for model in [lgbm, rf]:
        #     model.fit(X_train, y_train)
        #     y_pred = model.predict(X_valid)

        #     loss = MSE(y_valid, y_pred)
        #     show_results(loss, model=model)

