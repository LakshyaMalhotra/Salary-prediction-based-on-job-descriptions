__author__ = "Lakshya Malhotra"
__copyright__ = "Copyright (c) 2021 Lakshya Malhotra"

# Import the libraries

import os
import pandas as pd
from sklearn import model_selection, preprocessing
from typing import List

# Create data class to load and preprocess the data
class Data:
    def __init__(
        self,
        train_feature_file: str,
        train_target_file: str,
        test_file: str,
        cat_vars: List[str],
        num_vars: List[str],
        target_var: str,
        unique_var: str,
    ):
        """Data class to load and preprocess the data.

        Args:
            train_feature_file (str): Path to the training features
            train_target_file (str): Path to the training target
            test_file (str): Path to the test data
            cat_vars (list(str)): list of the names of categorical columns
            num_vars (list(str)): list of the names of numeric columns
            target_var (str): target variable (label)
            unique_var (str): Unique identifier for each training example

        """
        self.test_file = test_file
        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.target_var = target_var
        self.unique_var = unique_var
        self.label_encoders = {}
        self.train_df = self._create_train_df(
            train_feature_file, train_target_file
        )
        self.test_df = self._create_test_df(test_file)

    def _create_train_df(
        self,
        train_feature_file: str,
        train_target_file: str,
        preprocess=True,
        label_encode=True,
        kfold=False,
    ) -> pd.DataFrame:
        """
        Create and preprocess training data.
        """
        # load the data from the files
        feature_df = self._load_data(train_feature_file)
        target_df = self._load_data(train_target_file)
        train_df = self._concat_dfs(feature_df, target_df)

        # preprocess the dataframe if flagged
        if preprocess:
            train_df = self._clean_data(
                train_df, self.unique_var, self.target_var
            )
            train_df = self._shuffle_data(train_df)

        # label encode the dataframe if flagged
        if label_encode:
            self.label_encode_df(train_df, self.cat_vars)

        # create k-fold cross-validation in dataframe if flagged
        if kfold:
            train_df = self.create_folds(train_df)
        return train_df

    def _create_test_df(
        self, test_file: str, label_encode=True
    ) -> pd.DataFrame:
        """
        Create and preprocess test data.
        """
        test_df = self._load_data(test_file)
        if label_encode:
            self.label_encode_df(test_df, self.cat_vars)

        return test_df

    def _label_encode(
        self, df: pd.DataFrame, col: str, le: dict = None
    ) -> None:
        """
        Label encode the dataframe.
        """
        # if the label_encoder instance already created for the column, just
        # transform it else fit it first
        if le:
            df[col] = le.transform(df[col])

        else:
            le = preprocessing.LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
            self.label_encoders[col] = le

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the data from a file.
        """
        df = pd.read_csv(file_path)
        return df

    def _concat_dfs(
        self, feature_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join both feature and target dataframes.
        """
        return pd.concat([feature_df, target_df], axis=1)

    def _clean_data(
        self, df: pd.DataFrame, unique_var: str, target_var: str
    ) -> pd.DataFrame:
        """
        Clean the data by removing duplicates and remove rows with negative salary.
        """
        df = df.drop_duplicates(subset=unique_var)
        df = df[df[target_var] > 0]
        return df

    def _shuffle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffle the data.
        """
        return df.sample(frac=1).reset_index(drop=True)

    def label_encode_df(self, df: pd.DataFrame, cols: List[str]) -> None:
        """
        Label encode the categorical columns.
        """
        # iterate through the categorical columns
        for col in cols:
            if col in self.label_encoders:
                self._label_encode(df, col, self.label_encoders[col])
            else:
                self._label_encode(df, col)

    def create_folds(self, df: pd.DataFrame, n_folds: int = 10) -> pd.DataFrame:
        """
        Create k-folds for cross-validation.
        """
        # create a new column and fill it with -1
        df["kfold"] = -1
        df = self._shuffle_data(df)

        # instantiate the kfold cross validation
        kf = model_selection.KFold(n_splits=n_folds)

        # fill the new kfold column
        for fold, (_, v_) in enumerate(kf.split(X=df)):
            df[v_, "kfold"] = fold

        return df


class EngineerFeatures:
    def __init__(self, data: Data):
        self.data = data
        self.cat_vars = data.cat_vars
        self.num_vars = data.num_vars
        self.target = data.target_var
        self.groupby_cats = data.train_df.groupby(self.cat_vars)

    def add_features(self) -> None:
        """
        Create new features and merge the dataframes to training and test set.
        """
        feature_df = pd.DataFrame()
        aggs = ["mean", "min", "max", "std", "median"]

        # iterate through all the numeric variables and target
        for col in self.num_vars + [self.data.target_var]:
            for agg in aggs:
                feature_df[agg + "_" + col] = self._create_groupby_cols(
                    col, agg
                )
        feature_df.reset_index(inplace=True)

        # merge the feature data frame to training set
        self.data.train_df = self._merge_new_cols(
            self.data.train_df, feature_df, self.cat_vars
        )
        # merge the feature dataframe to test set
        self.data.test_df = self._merge_new_cols(
            self.data.test_df, feature_df, self.cat_vars
        )

    def _create_groupby_cols(self, col: str, agg_name: str) -> pd.Series:
        """
        Apply `groupby` on the column `col` with aggregate `agg_name`
        """
        return self.groupby_cats[col].agg(agg_name)

    def _merge_new_cols(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, keys: list = None,
    ) -> pd.DataFrame:
        """
        Merge the two dataframes on categorical columns
        """
        df = pd.merge(left=df, right=feature_df, how="left", on=keys)
        df.fillna(0, inplace=True)
        return df

    def get_df_info(self) -> None:
        print(self.data.train_df.head())
        print(self.data.train_df.info())
        print(self.data.test_df.head())
        print(self.data.test_df.info())


path = "data/"
train_feature_file = os.path.join(path, "train_features.csv")
train_target_file = os.path.join(path, "train_salaries.csv")
test_file = os.path.join(path, "test_features.csv")

cat_vars = ["companyId", "jobType", "degree", "major", "industry"]
num_vars = ["yearsExperience", "milesFromMetropolis"]
target_var = "salary"
unique_var = "jobId"

data = Data(
    train_feature_file,
    train_target_file,
    test_file,
    cat_vars=cat_vars,
    num_vars=num_vars,
    target_var=target_var,
    unique_var=unique_var,
)
print("Dataframe before feature engineering")
print(data.train_df.head())
fe = EngineerFeatures(data)
fe.add_features()
print("Dataframe after feature engineering")
fe.get_df_info()
