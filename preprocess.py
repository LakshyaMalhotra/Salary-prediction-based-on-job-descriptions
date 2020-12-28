__author__ = "Lakshya Malhotra"
__copyright__ = "Copyright (c) 2020 Lakshya Malhotra"

# Import the libraries
import os

import pandas as pd
from sklearn import model_selection

def read_file(file: str) -> pd.DataFrame:
    """
    Read the csv file containing the data.
    :return: (pd.DataFrame)
    """
    return pd.read_csv(file)


def combine_data(df1: pd.DataFrame, df2: pd.DataFrame, key: str = None) -> pd.DataFrame:
    """
    Combine two dataframes based on a common key
    :param df1: dataframe containing features
    :param df2: dataframe containing target
    :param key: common key on which the dataframes are joined

    :return: pd.DataFrame
    """
    # check if both dataframes have same rows
    df = None
    try:
        features_key = set(df1[key].values)
        target_key = set(df2[key].values)
        assert len(features_key.intersection(target_key)) == len(df1)
        df = df1.set_index(key).join(df2.set_index(key)).reset_index()
    except AssertionError:
        print("Could not join dataframes with different number of rows...exiting!")
    return df


def clean_data(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Cleans the dataframe by dropping duplicates and rows with zero salaries
    :param df: raw dataframe
    :param col: column to be used for removing duplicates
    :return: clean dataframe
    """
    clean_df = df.drop_duplicates(subset=col)
    clean_df = clean_df[clean_df["salary"] > 0]

    assert isinstance(clean_df, pd.DataFrame)
    return clean_df

def create_folds(df: pd.DataFrame)->pd.DataFrame:
    """
    Create folds for cross-validation
    :param df: Original cleaned dataframe
    :return: pd.DataFrame
    """
    # create a new column and fill it with -1
    df['kfold'] = -1

    # shuffle all the rows of the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # instantiate the kfold cross validation
    kf = model_selection.KFold(n_splits=10)

    # fill the new kfold column
    for fold, (t_, v_) in enumerate(kf.split(X=df)):
        df.loc[v_, 'kfold'] = fold

    return df


if __name__ == "__main__":
    # define path to the data directory
    path = "data/"
    features_file = os.path.join(path, "train_features.csv")
    target_file = os.path.join(path, "train_salaries.csv")

    # get the features and target dataframes
    features_df = read_file(features_file)
    target_df = read_file(target_file)

    # combine both features and target dataframe
    train_df = combine_data(features_df, target_df, key="jobId")

    if train_df is None:
        raise TypeError("Dataframe is empty")
    print(train_df.head())

    # get the cleaned dataframe
    train_df = clean_data(train_df, col="jobId")

    # create folds in the dataframe
    train_df = create_folds(train_df)
    print(train_df.shape)
    print(train_df.kfold.value_counts())
    
    # save this dataframe to another file
    train_df.to_csv(os.path.join(path, 'train_folds.csv'), index=False)
