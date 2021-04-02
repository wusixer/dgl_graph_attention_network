"""Functions related to training the neural network."""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as np
from jax import value_and_grad, vmap
from jax.experimental import optimizers


def mse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Return the mean squared error between y_hat and y_true.

    :param y_hat: NumPy array of model prediction values
    :param y_true: NumPy array of true values
    :returns: Scalar-valued mean of squared error.
    """
    return np.mean((y_hat - y_true) ** 2)


def mseloss(params, model: Callable, inputs: np.ndarray, outputs: np.ndarray):
    """
    Training task loss function.

    Always vmap the model function over inputs!

    :param params: Model parameters, passed into parameter ``model``.
    :param model: Model function, which takes in arguments (params, inputs)
    :param inputs: Input data passed into model function.
    :param outputs: Targets to learn.
    :returns: Scalar-valued mean of squared error.
    """
    y_hat = vmap(partial(model, params))(inputs)
    return mse(y_hat, outputs)


def step(
    i: int,
    state: optimizers.OptimizerState,
    loss_fun: Callable,
    apply_fun: Callable,
    update_fun: Callable,
    get_params: Callable,
    inputs: Tuple[np.ndarray, np.ndarray],
    outputs: np.ndarray,
) -> optimizers.OptimizerState:
    """
    Step function that we can use for training.

    Example usage:

    .. code-block:: python

        from functools import partial
        from patch_gnn.training import mseloss, step
        from jax import grad, jit

        # set up the step function
        init, update, get_params = adam(step_size=0.001)
        dmseloss = grad(mseloss)
        stepfunc = partial(
            step,
            dloss_fun=dmseloss,
            apply_fun=apply_fun,  # we assume you have this set up!
            update_fun=update,
            get_params=get_params
        )
        stepfunc = jit(stepfunc)  # jit-compile the func to make it fast!

        state = init(params)  # we assume you have params set up!
        for i in range(3000):
            state = step(i, state)
        params_final = get_params(state)

    :param i: Step iteration
    :param state: Current state of the parameters.
    :param dloss_fun: Gradient function.
        Accepts the parameters: (params, model, inputs, outputs).
    :param apply_fun: Model function. (Forward pass!)
        Accepts the parameters: (params, inputs).
    :param update_fun: Optimizer update function.
    :param get_params: Optimizer state unpacking function.
    :param inputs: Model inputs that get passed to dloss_fun and apply_fun.
    :param outputs: Ground truth model outputs that get passed to dloss_fun.

    :returns:
        Optimizer state (`state`) and loss score (`v`).
    """
    dloss_fun = value_and_grad(
        loss_fun
    )  # Create a function which evaluates both loss_fun and the gradient of loss_fun
    params = get_params(state)
    v, g = dloss_fun(
        params, apply_fun, inputs, outputs
    )  # v is the value of loss fun, g is the gradient of loss_fun
    state = update_fun(i, g, state)

    return state, v
