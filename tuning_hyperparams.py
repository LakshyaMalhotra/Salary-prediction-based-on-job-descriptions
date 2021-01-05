import os
import json

from sklearn import metrics
from sklearn import model_selection
import optuna
import lightgbm as lgb

from preprocess import Data, EngineerFeatures


class Optimize:
    train_df = None

    @staticmethod
    def optimize(trial):
        features = [
            col
            for col in Optimize.train_df.columns
            if col not in ["jobId", "salary"]
        ]

        X = Optimize.train_df.loc[:, features]
        y = Optimize.train_df.salary

        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
            X, y, test_size=0.2, random_state=23
        )
        d_train = lgb.Dataset(X_train, label=y_train)
        params = {
            "application": "regression",
            "metric": "mean_squared_error",
            "verbosity": -1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 6, 24),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 1.0, log=True
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.3, 1.0
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.4, 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        lgbm = lgb.train(params, d_train, num_boost_round=150)
        y_pred = lgbm.predict(X_valid)
        error = metrics.mean_squared_error(y_valid, y_pred)

        return error


if __name__ == "__main__":
    path = "data/"
    train_feature_file = os.path.join(path, "train_features.csv")
    train_target_file = os.path.join(path, "train_salaries.csv")
    test_file = os.path.join(path, "test_features.csv")

    cat_vars = ["companyId", "jobType", "degree", "major", "industry"]
    num_vars = ["yearsExperience", "milesFromMetropolis"]
    target_var = "salary"
    unique_var = "jobId"

    print("Loading and preprocessing data...")
    data = Data(
        train_feature_file,
        train_target_file,
        test_file,
        cat_vars=cat_vars,
        num_vars=num_vars,
        target_var=target_var,
        unique_var=unique_var,
    )
    print("Performing feature engineering and creating K-fold CV...")
    fe = EngineerFeatures(data)
    fe.add_features()

    Optimize.train_df = data.train_df
    opt = Optimize()

    study = optuna.create_study(direction="minimize")
    study.optimize(opt.optimize, n_trials=40)
    best_params_ = study.best_params
    hyperparams_dict = json.dumps(best_params_)
    with open("models/lgbm_best_hyperparams.json", "w") as f:
        f.write(hyperparams_dict)

    print(f"Best parameters: \n{best_params_}")
    od = optuna.importance.get_param_importances(study)
    for k, v in od.items():
        print(k, v)
