__author__ = "Lakshya Malhotra"
__copyright__ = "Copyright (c) 2021 Lakshya Malhotra"

# Library imports
import os
from typing import Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model

from preprocess import Data, EngineerFeatures


class Run:
    def __init__(self, data: Data, models=None, n_folds: int = 10):
        self.models = models
        self.best_model = None
        self.predictions = None
        self.n_folds = n_folds
        self.mean_mse = {}
        self.best_loss_fold = np.inf
        self.best_loss_model = np.inf
        self.data = data
        self.df = data.train_df
        self.features = [
            col for col in self.df.columns if col not in ["jobId", "salary"]
        ]
        self.target = self.data.target_var

    def add_model(self, model):
        if self.models is None:
            self.models = []
        self.models.append(model)

    def cross_validate(self):
        for model in self.models:
            for fold in range(self.n_folds):
                train, valid = self._get_data(fold)
                y_true, y_pred = self._run_model_cv(train, valid, model)
                loss = self._mean_squared_error(y_true, y_pred)
                print(f"Fold: {fold}, Loss: {loss}")
                if loss < self.best_loss_fold:
                    self.best_loss_fold = loss
                    model_name = (
                        "lgbm_best.sav"
                        if model == "LGBMRegressor()"
                        else f"{model.__name__}_best.sav"
                    )
                    joblib.dump(model, model_name)
            model_loss = self.best_loss_fold
            if model_loss < self.best_loss_model:
                self.best_loss_model = model_loss

    def _get_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = self.df[self.df["kfold"] != fold].reset_index(drop=True)
        valid = self.df[self.df["kfold"] == fold].reset_index(drop=True)

        return train, valid

    def _run_model_cv(
        self, train: pd.DataFrame, valid: pd.DataFrame, model
    ) -> Tuple[pd.Series, pd.Series]:
        X_train = train[self.features]
        y_train = train[self.target]

        X_valid = valid[self.features]
        y_valid = valid[self.target]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        return y_valid, y_pred

    def select_best_model(self):
        pass

    def select_best_fold(self):
        pass

    def best_model_fit(self):
        pass

    def best_model_prediction(self):
        pass

    @staticmethod
    def _mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
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
    fe = EngineerFeatures(data, n_folds=3)
    fe.add_features(kfold=True)
    print("Dataframe after feature engineering")
    fe.get_df_info()

    lgbm = lgb.LGBMRegressor(n_jobs=-1)

    run = Run(data, models=[lgbm], n_folds=3)
    run.cross_validate()
