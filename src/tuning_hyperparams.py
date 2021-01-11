__author__ = "Lakshya Malhotra"
__copyright__ = "Copyright (c) 2021 Lakshya Malhotra"

# library imports
import os
import json
from typing import OrderedDict

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna
import lightgbm as lgb

from preprocess import Data, EngineerFeatures
from main import Run

# Optimizer class
class Optimize:
    # class variables to get the data and categorical columns from the `Data` class
    train_df = None
    cat_vars = []

    @staticmethod
    def optimize(trial) -> float:
        """
        Optimize the hyperparameters of the regression model and return the 
        loss for the given trial
        """
        # list of features
        features = [
            col
            for col in Optimize.train_df.columns
            if col not in ["jobId", "salary"]
        ]

        # features that need feature scaling
        scale_features = [
            col for col in features if col not in Optimize.cat_vars
        ]

        # randomly select the regressor to use for optimization
        regressor_name = trial.suggest_categorical("regressor", ["lgbr"])

        # get the features and target from the original dataframe
        X = Optimize.train_df.loc[:, features]
        y = Optimize.train_df.salary

        # split features and target into training and validation set
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=23
        )

        # define the respective hyperparameters for each regressor and
        # train the regressor
        if regressor_name == "lgbr":
            params = {
                "metric": "mean_squared_error",
                "verbosity": -1,
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
                "reg_lambda": trial.suggest_loguniform(
                    "reg_lambda", 1e-8, 10.0
                ),
                "num_leaves": trial.suggest_int("num_leaves", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 30),
                "learning_rate": trial.suggest_loguniform(
                    "learning_rate", 0.01, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.3, 1.0
                ),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 100
                ),
            }
            regressor_obj = lgb.LGBMRegressor(**params)
        elif regressor_name == "rf":
            params = {
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 30),
                "max_features": trial.suggest_categorical(
                    "max_features", ["log2", "sqrt", None]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 10
                ),
            }
            regressor_obj = RandomForestRegressor(**params)

        else:
            params = {
                "alpha": trial.suggest_loguniform("alpha", 0.1, 100),
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
                "normalize": trial.suggest_categorical(
                    "normalize", [True, False]
                ),
                "solver": trial.suggest_categorical(
                    "solver", ["auto", "svd", "saga", "lsqr", "cholesky"]
                ),
            }
            # scale the selected features
            ct = ColumnTransformer(
                [("scale", StandardScaler(), scale_features)],
                remainder="passthrough",
                n_jobs=-1,
            )
            ridge = Ridge(**params)
            regressor_obj = Pipeline(
                [("preprocessing", ct), ("ridge_lr", ridge)]
            )
        # fit the regressor on training data
        regressor_obj.fit(X_train, y_train)

        # make predictions on the validation data
        y_pred = regressor_obj.predict(X_valid)

        # get the loss from the predictions
        error = mean_squared_error(y_valid, y_pred)

        return error

    @staticmethod
    def write_to_json(path: str, best_params: dict) -> None:
        """
        Write the best hyperparameters to a JSON file
        """
        hyperparams_dict = json.dumps(best_params)
        path_to_json = os.path.join(path, "best_hyperparams_lgbr.json")
        with open(path_to_json, "w") as f:
            f.write(hyperparams_dict)

        return path_to_json

    @staticmethod
    def print_param_stats(
        best_params: dict, study: optuna.create_study
    ) -> None:
        """
        Display the results
        """
        print("Best parameters: ")
        print(best_params)
        od = optuna.importance.get_param_importances(study)
        od = OrderedDict(sorted(od.items(), key=lambda x: x[1], reverse=True))
        print("Parameter importance for the best model: ")
        for k, v in od.items():
            print(k, v)

        return od


if __name__ == "__main__":
    # define variables
    path = "data/"
    model_path = "models/"

    # create a `Run` instance and get data
    run = Run(path=path, model_dir=model_path)
    data = run.get_data(kfold=False)

    print("Optimizing the model hyperparameters...")
    # assign data to the class variable and instantiate the optimizer object
    Optimize.train_df = data.train_df
    Optimize.cat_vars = data.cat_vars
    opt = Optimize()

    # create a study object and start tuning the hyperprameters
    study = optuna.create_study(direction="minimize")
    study.optimize(opt.optimize, n_trials=15)
    best_params_ = study.best_params

    # store and display the results
    json_path = opt.write_to_json(model_path, best_params=best_params_)
    print(f"Best params are saved to file: {json_path}")
    od = opt.print_param_stats(best_params_, study=study)
    gp = optuna.visualization.plot_parallel_coordinate(
        study, params=od.keys()[:6]
    )
    gp.show()
