__author__ = "Lakshya Malhotra"
__copyright__ = "Copyright (c) 2021 Lakshya Malhotra"

# Library imports
import os
import json
from typing import Tuple, Union, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from preprocess import Data, EngineerFeatures


class Model:
    def __init__(
        self,
        data: Data,
        models: List[Union[RandomForestRegressor, lgb.LGBMRegressor]] = None,
        n_folds: int = 10,
        model_dir: str = "models",
    ) -> None:
        """Run the models and perform K-fold cross-validation.

        Args:
        -----
            data (Data): Instance of `Data` class
            models (List[Union[RandomForestRegressor, lgb.LGBMRegressor]], optional): 
                Model to be used, it can be either a sklearn or lightGBM model. Defaults to None.
            n_folds (int, optional): Number of folds of K-fold cross-validation. Defaults to 10.
            model_dir (str, optional): Path for saving the models. Defaults to "models".
        """
        self.models = models
        self.best_model = None
        self.predictions = None
        self.n_folds = n_folds
        self.model_dir = model_dir
        self.mean_mse = {}
        self.fold_mse = []
        self.best_loss_fold = np.inf
        self.best_loss_model = np.inf
        self.data = data
        self.df = data.train_df
        self.features = [
            col for col in self.df.columns if col not in ["jobId", "salary"]
        ]
        self.target = self.data.target_var

    def add_model(
        self, model: List[Union[RandomForestRegressor, lgb.LGBMRegressor]]
    ) -> None:
        """Adds the models to be used in a list. 
        """
        if self.models is None:
            self.models = []
        self.models.append(model)

    def cross_validate(self) -> None:
        """Perform the K-fold cross-validation for all the models.
        """
        for model in self.models:
            for fold in range(self.n_folds):
                train, valid = self._get_data(fold)
                y_true, y_pred = self._run_model_cv(train, valid, model)
                loss = self._mean_squared_error(y_true, y_pred)
                save_message = self._save_model(loss, model)
                self._print_stats(fold, model, loss, save_message)
                self.fold_mse.append(loss)
            self.best_loss_fold = np.inf
            self.mean_mse[model] = np.mean(self.fold_mse)
            model_loss = self.best_loss_fold
            if model_loss < self.best_loss_model:
                self.best_loss_model = model_loss

    def _get_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get the data for a given fold.

        Args:
        -----
            fold (int): Fold to be used.

        Returns:
        --------
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing train and validation dataframes.
        """
        train = self.df[self.df["kfold"] != fold].reset_index(drop=True)
        valid = self.df[self.df["kfold"] == fold].reset_index(drop=True)

        return train, valid

    def _run_model_cv(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        model: List[Union[RandomForestRegressor, lgb.LGBMRegressor]],
    ) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Perform cross-validation step for a given fold and model.

        Args:
        -----
            train (pd.DataFrame): Training dataframe
            valid (pd.DataFrame): Validation dataframe
            model (List[Union[RandomForestRegressor, lgb.LGBMRegressor]]):

        Returns:
        --------
            Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]: Tuple containing true and predicted labels.
        """
        X_train = train[self.features]
        y_train = train[self.target]

        X_valid = valid[self.features]
        y_valid = valid[self.target]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        return y_valid, y_pred

    def _save_model(
        self,
        loss: float,
        model: List[Union[RandomForestRegressor, lgb.LGBMRegressor]],
    ) -> str:
        """Save the model in a binary file.

        Args:
        -----
            loss (float): Mean squared error
            model (List[Union[RandomForestRegressor, lgb.LGBMRegressor]]): model (regressor) object

        Returns:
        --------
            str: Save message.
        """
        if loss < self.best_loss_fold:
            self.best_loss_fold = loss
            model_name = f"{type(model).__name__}_best.sav"
            model_path = os.path.join(self.model_dir, model_name)
            message = joblib.dump(model, model_path)
            if message is not None:
                return f"Model saved in: {message[0]}"

        return "Loss didn't improve!"

    def select_best_model(
        self,
    ) -> List[Union[RandomForestRegressor, lgb.LGBMRegressor]]:
        """Select the best model on the basis of mean squared error.

        Returns:
        --------
            List[Union[RandomForestRegressor, lgb.LGBMRegressor]]: Model object for the best model.
        """
        best_model = min(self.mean_mse, key=self.mean_mse.get)
        return best_model

    def best_model_fit(self):
        pass

    def best_model_prediction(self):
        pass

    @staticmethod
    def _mean_squared_error(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
    ) -> float:
        """Calculate the mean squared error.

        Args:
        -----
            y_true (Union[pd.Series, np.ndarray]): Array or series containing actual target.
            y_pred (Union[pd.Series, np.ndarray]): Array of series containing model predictions.

        Returns:
        --------
            float: Mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def _print_stats(
        fold: int,
        model: List[Union[RandomForestRegressor, lgb.LGBMRegressor]],
        loss: float,
        print_message: str,
    ) -> None:
        """Print cross-validation results on screen.
        """
        print(f"Model: {model}, fold: {fold}")
        print(f"Loss: {loss}, {print_message}")


class Run:
    cat_vars = ["companyId", "jobType", "degree", "major", "industry"]
    num_vars = ["yearsExperience", "milesFromMetropolis"]
    target_var = "salary"
    unique_var = "jobId"

    def __init__(
        self, path: str, model_dir: str, param_file: str, n_folds: int,
    ):
        self.path = path
        self.model_dir = model_dir
        self.param_file = param_file
        self.n_folds = n_folds
        self.train_feature_file = os.path.join(path, "train_features.csv")
        self.train_target_file = os.path.join(path, "train_salaries.csv")
        self.test_file = os.path.join(path, "test_features.csv")

    def get_data(self):
        print("Loading and preprocessing data...")
        data = Data(
            self.train_feature_file,
            self.train_target_file,
            self.test_file,
            Run.cat_vars,
            Run.num_vars,
            Run.target_var,
            Run.unique_var,
        )
        print("Performing feature engineering and creating K-fold CV...")
        fe = EngineerFeatures(data, n_folds=self.n_folds)
        fe.add_features(kfold=True)

        return data

    def _get_hyperparams(self):
        print("Loading hyperparameters...")
        with open(os.path.join(self.model_dir, self.param_file), "r") as f:
            lgb_params = json.load(f)
        lgb_params = {k: v for k, v in lgb_params.items() if k != "regressor"}
        rf_params = {
            "n_estimators": 60,
            "max_depth": 15,
            "min_samples_split": 80,
            "max_features": 8,
        }
        return lgb_params, rf_params

    def _models(self):
        lgb_params, rf_params = self._get_hyperparams()
        print("Updating models with hyperparameters...")
        lgbm = lgb.LGBMRegressor(**lgb_params, n_jobs=-1, verbose=-1)
        rf = RandomForestRegressor(**rf_params, n_jobs=-1,)

        return lgbm, rf

    def load_models(self):
        data = self.get_data()
        model = Model(data, n_folds=self.n_folds)
        lgbm, rf = self._models()
        print("Loading models...")
        model.add_model(lgbm)
        model.add_model(rf)

        return model

    def run_cv(self):
        model = self.load_models()
        print("Running cross-validation...")
        model.cross_validate()
        print(model.select_best_model())


if __name__ == "__main__":
    path = "data/"
    model_dir = "models/"
    param_file = "best_hyperparams.json"
    run = Run(path, model_dir, param_file, n_folds=2)
    run.run_cv()
