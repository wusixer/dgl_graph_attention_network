"""Use optuna module to auto-tune models"""
from typing import Dict

import optuna
from jax.config import config
from optuna.samplers import RandomSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

config.update("jax_debug_nans", True)


def evotune_mpnn_class(
    model,
    X,
    y,
    num_training_steps_kwargs: Dict = {},
    optimizer_step_size_kwargs: Dict = {},
    n_trials: int = 10,
):
    """
    evotune for MPNN class models
    One can choose to tune num_training_steps (num of epochs) and optimizer_step_size (learning rate)

    :param X: input data for MPNN class models
    :param y: outcome associated with input data, should be of shape (sample_size,), use np.squeeze(y) if not
    :params num_training_steps_kwargs: a dictionary with the format of,
                                num_training_steps_kwargs = {
                                "name": "num_training_steps", #requires to have as is
                                "low": 10, # one can change it
                                "high": 14,  # one can change it
                                "log" :True # one can change it
                            }, default is an empty dictionary with means no hyperparameters should be tuned

    :params optimizer_step_size_kwargs: a dictionary with the format of,
                                optimizer_step_size_kwargs = {
                                "name" : "optimizer_step_size",# this key value pair is required as is
                                "low" : 1e-5, # one can change the value
                                "high" : 1e-2, # one can change the value
                            }
    :params n_trials: number of experiments for optuna to run, each experiment is associated with one hyperparameter combination

    return:
            The ideal param combination (that one asked to tune) with the lowest loss error on input data in
            in given number of experiments
    """
    if (
        len(num_training_steps_kwargs) == 0
        and len(optimizer_step_size_kwargs) == 0
    ):
        raise ValueError("The hyperparameters to optimize cannot be empty")

    def objective(trial):
        param_dict = {}
        # defensive programming to check for empty values
        if len(num_training_steps_kwargs) != 0:
            num_training_steps = trial.suggest_int(**num_training_steps_kwargs)
            # update param_dict
            param_dict["num_training_steps"] = num_training_steps
        if len(optimizer_step_size_kwargs) != 0:
            optimizer_step_size = trial.suggest_uniform(
                **optimizer_step_size_kwargs
            )
            param_dict["optimizer_step_size"] = optimizer_step_size

        print(f"The params that were optimized is {param_dict}")
        # the model object is callable and takes a dict argument to update the parameters
        mpnn_obj = model(param_dict=param_dict)

        loss_history = mpnn_obj.fit(X, y).loss_history
        # mpnn_obj is used here to get the number of training steps, in case training steps
        # is not part of the tunning parameters
        loss = loss_history[mpnn_obj.num_training_steps - 1]
        return loss

    sampler = RandomSampler(seed=10)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params


def evotune_rf(
    X,
    y,
    rf_max_depth_kwargs: Dict = {},
    rf_n_estimator_kwargs: Dict = {},
    n_trials: int = 20,
):
    """
    evotune for random forest models
    One can choose to tune max_depth and n_estimators (num of trees in rf)

    :param X: input data for random forest models
    :param y: outcome associated with input data, should be of shape (sample_size,), use np.squeeze(y) if not
    :params rf_max_depth_kwargs: a dictionary with the format of,
                                rf_max_depth_kwargs = {
                                "name" : "max_depth", # this key value pair is required as is
                                "low" :2,  # one can change the value
                                "high" : 32, # one can change the value
                                "log" :True  #this key value pair is required as is
                            }, default is an empty dictionary with means no hyperparameters should be tuned

    :params rf_n_estimator_kwargs: a dictionary with the format of,
                                rf_n_estimator_kwargs = {
                                "name" : "n_estimators",# this key value pair is required as is
                                "low" : 64, # one can change the value
                                "high" : 128, # one can change the value
                                "log" : True # this key value pair is required as is
                            }

    :params n_trials: number of experiments for optuna to run, each experiment is associated with one hyperparameter combination

    return:
            The ideal param combination (that one asked to tune) with the lowest MSE error on input data in
            in given number of experiments

    """
    if len(rf_max_depth_kwargs) == 0 and len(rf_n_estimator_kwargs) == 0:
        raise ValueError("The hyperparameters to optimize cannot be empty")

    def objective(trial):
        param_dict = {}
        # defensive programming to check for empty values
        if len(rf_max_depth_kwargs) != 0:
            max_depth = trial.suggest_int(**rf_max_depth_kwargs)
            # update param_dict
            param_dict["max_depth"] = max_depth
        if len(rf_n_estimator_kwargs) != 0:
            n_estimators = trial.suggest_int(**rf_n_estimator_kwargs)
            param_dict["n_estimators"] = n_estimators

        print(f"The params that were optimized {param_dict}")
        # build rf model object
        rf_obj = RandomForestRegressor(oob_score=True, n_jobs=-1, **param_dict)
        # rf_obj = model(**param_dict) # this won't work since this class doens't have __call__

        rf_obj.fit(X, y)
        y_pred = rf_obj.predict(X)

        error = mean_squared_error(y, y_pred)

        return error

    sampler = RandomSampler(seed=10)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params
