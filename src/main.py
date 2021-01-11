__author__ = "Lakshya Malhotra"
__copyright__ = "Copyright (c) 2021 Lakshya Malhotra"

# library imports
import os
import json
from argparse import ArgumentParser
from typing import Tuple, Union, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from preprocess import Data, EngineerFeatures
import matplotlib.pyplot as plt
import seaborn as sns

# plot formatting options
sns.set_style("whitegrid")
sns.set_palette("deep")


def build_argparser() -> ArgumentParser:
    """Parse command line arguments.

    Returns:
    --------
        ArgumentParser: Parser object to parse command line args.
    """
    parser = ArgumentParser(
        description="Add arguments needed to run the models."
    )
    parser.add_argument(
        "--n_folds",
        required=True,
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Path to the models directory.",
    )
    parser.add_argument(
        "--params",
        required=True,
        type=str,
        help="Path to the file containing optimized hyperparameters.",
    )

    return parser


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
        self.best_model = None
        self.predictions = None
        self.data = data
        self.train_df = data.train_df
        self.test_df = data.test_df
        self.features = [
            col
            for col in self.train_df.columns
            if col not in ["jobId", "salary", "kfold"]
        ]
        self.scaled_features = [
            col for col in self.features if col not in self.data.cat_vars
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

    def _get_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get the data for a given fold.

        Args:
        -----
            fold (int): Fold to be used.

        Returns:
        --------
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing train and validation dataframes.
        """
        train = self.train_df[self.train_df["kfold"] != fold].reset_index(
            drop=True
        )
        valid = self.train_df[self.train_df["kfold"] == fold].reset_index(
            drop=True
        )

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
            Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]: 
            Tuple containing true and predicted labels.
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

    def select_best_model(self) -> None:
        """Select the best model on the basis of mean squared error.
        """
        self.best_model = min(self.mean_mse, key=self.mean_mse.get)

    def best_model_fit(self) -> None:
        """Fit the training data with the best model.
        """
        self.best_model.fit(
            self.train_df[self.features], self.train_df[self.target]
        )

    def best_model_predictions(self, save_predictions: bool = True) -> None:
        """Make predictions from the fitted model and save the predictions to a 
        CSV file.

        Args:
        -----
            save_predictions (bool): Whether to save predictions to a file. 
            Defaults to True.
        """
        self.predictions = self.best_model.predict(self.test_df[self.features])

        if save_predictions:
            job_ids = self.test_df["jobId"]
            assert len(job_ids) == len(self.predictions)
            results = pd.DataFrame(
                {"jobId": job_ids, "predicted_salary": self.predictions}
            )
            results.to_csv(
                os.path.join(self.model_dir, "predictions.csv"), index=False
            )

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
        """Print results on screen on each step of cross-validation.
        """
        print(f"Model: {model}, fold: {fold}")
        print(f"Loss: {loss}, {print_message}")

    def get_feature_importances(self) -> pd.DataFrame:
        """Calculate the feature importances for all the models.
        """
        # create dataframe to store feature importances
        feature_importances = pd.DataFrame({"feature": self.features})

        # iterate through all the models
        for model in self.models:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_importances[
                    f"{type(model).__name__}_importance"
                ] = importances
            else:
                continue
        feature_importances.sort_values(
            by="LGBMRegressor_importance", ascending=False, inplace=True
        )
        feature_importances.set_index("feature", inplace=True, drop=True)

        # Normalize LightGBM feature importances
        feature_importances["LGBMRegressor_importance"] = (
            feature_importances["LGBMRegressor_importance"]
            / feature_importances["LGBMRegressor_importance"].sum()
        )
        # return importance dataframe if exists
        if len(feature_importances.columns) == 0:
            return "Feature importances do not exist for any of the models."
        else:
            return feature_importances

    @staticmethod
    def plot_feature_importance(df: pd.DataFrame) -> None:
        """Plot the feature importances

        Args:
        -----
            df (pd.DataFrame): Dataframe containing feature importances from 
            various models.
        """
        cols = df.columns.tolist()
        fig, ax = plt.subplots(1, len(cols), sharey=True, figsize=(14, 8))
        y = df.index
        x0 = df.loc[:, cols[0]]
        sns.barplot(x=x0, y=y, ec="k", alpha=0.6, ax=ax[0])

        x1 = df.loc[:, cols[1]]
        sns.barplot(x=x1, y=y, ec="k", alpha=0.6, ax=ax[1])
        plt.suptitle("Feature Importances", fontsize=18)
        plt.tight_layout()
        plt.show()

    def print_summary(self) -> None:
        """Print model summaries, best model, and feature importances.
        """
        print("Model summaries:")
        # print MSE for each model
        for model in self.mean_mse:
            print(f"  {type(model).__name__}-MSE: {self.mean_mse[model]}")

        print(f"Best model: {self.best_model}")

        # get the feature importances dataframe and print it
        df = self.get_feature_importances()
        print("Feature importances:")
        print(df.head())

        # plot the feature importances
        Model.plot_feature_importance(df)


class Run:
    # defining some class variables
    cat_vars = ["companyId", "jobType", "degree", "major", "industry"]
    num_vars = ["yearsExperience", "milesFromMetropolis"]
    target_var = "salary"
    unique_var = "jobId"

    def __init__(
        self,
        path: str,
        model_dir: str,
        n_folds: int = 10,
        param_file: str = None,
    ):
        """Utility class to load models and their hyperparams and perform 
        cross-validation.

        Args:
        -----
            path (str): Path to the data directory.
            model_dir (str): Path to the saved models.
            n_folds (int, optional): Number of cross-validation folds. Defaults to 10.
            param_file (str, optional): Name of the file storing best hyperparamters 
            of the model . Defaults to None.
        """
        self.path = path
        self.model_dir = model_dir
        self.param_file = param_file
        self.n_folds = n_folds
        self.train_feature_file = os.path.join(path, "train_features.csv")
        self.train_target_file = os.path.join(path, "train_salaries.csv")
        self.test_file = os.path.join(path, "test_features.csv")

    def get_data(self, kfold: bool = True) -> Data:
        """Instantiate the data class and perform feature engineering.

        Args:
        -----
            kfold (bool, optional): Whether to perform cross-validation. Should 
            be set to `False` while doing hyperparameter tuning. Defaults to True.

        Returns:
        --------
            Data: Instance of the `Data` class
        """
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
        fe.add_features(kfold=kfold)

        return data

    def _get_hyperparams(self) -> Tuple[dict, dict]:
        """Load the optimized hyperparameters from a file for LightGBM model and 
        define parameter dict for Random forest.

        Returns:
        --------
            Tuple[dict, dict]: Dictionaries containing hyperparameters for the models.
        """
        print("Loading hyperparameters...")
        with open(os.path.join(self.model_dir, self.param_file), "r") as f:
            lgb_params = json.load(f)

        # remove the regressor name from the dictionary
        lgb_params = {k: v for k, v in lgb_params.items() if k != "regressor"}

        # define parameters for random forest
        rf_params = {
            "n_estimators": 60,
            "max_depth": 15,
            "min_samples_split": 80,
            "max_features": 8,
            "n_jobs": -1,
        }
        return lgb_params, rf_params

    def _models(self) -> Tuple[lgb.LGBMRegressor, RandomForestRegressor]:
        lgb_params, rf_params = self._get_hyperparams()
        print("Updating models with hyperparameters...")

        # define model objects`
        lgbm = lgb.LGBMRegressor(**lgb_params,)
        rf = RandomForestRegressor(**rf_params)

        return lgbm, rf

    def load_models(self, add_more_models=None) -> Model:
        """`Load the models and data to `Model` class. More models can be added 
        by the keyword argument `add_more_models`.

        Args:
        -----
            add_more_models (model_objects, optional): Add more models. Defaults to None.

        Returns:
        --------
            Model: Instance of `Model` class.
        """
        # get the `Data` instance
        data = self.get_data()

        # define the model instance
        model = Model(data, n_folds=self.n_folds)

        # get the models and add them to the `Model` instance
        lgbm, rf = self._models()
        print("Loading models...")
        model.add_model(lgbm)
        model.add_model(rf)

        # add more models if available
        if add_more_models is not None:
            model.add_model(add_more_models)

        return model

    def run_cv(self) -> None:
        """Perform the cross-validation.
        """
        model = self.load_models()
        print("Running cross-validation...")
        model.cross_validate()

        # fit the best model on the test data and make predictions
        model.select_best_model()
        model.best_model_fit()
        model.best_model_predictions()

        # plot the summary stats
        model.print_summary()


if __name__ == "__main__":
    args = build_argparser().parse_args()
    path = args.data_dir
    model_dir = args.model_dir
    param_file = args.params
    n_folds = args.n_folds
    run = Run(path, model_dir, n_folds=n_folds, param_file=param_file)
    run.run_cv()
