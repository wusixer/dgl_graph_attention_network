"""Test for training-related functions."""
from functools import partial

import jax.numpy as np
import pytest
from jax import grad, jit
from jax.experimental import stax
from jax.experimental.optimizers import adam
from jax.random import PRNGKey, normal, split

from patch_gnn.training import mseloss, step


@pytest.fixture
def linear_model():
    """Return linear model function pairs."""
    init_func, apply_func = stax.Dense(1)
    return init_func, apply_func


@pytest.fixture
def rng():
    """Return PRNGKey fixture."""
    key = PRNGKey(42)
    return key


def test_mseloss(linear_model, rng):
    """Test for mseloss function."""
    init_func, apply_func = linear_model

    k1, k2 = split(rng)
    _, params = init_func(rng=k2, input_shape=(-1, 1))

    k1, k2, k3 = split(k1, 3)
    inputs = normal(k2, shape=(10, 1))
    outputs = normal(k3, shape=(10, 1))

    loss = mseloss(params, apply_func, inputs, outputs)

    # Loss should be scalar, not vector.
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.isscalar.html
    assert np.ndim(loss) == 0
    # MSE should always be positive-valued
    assert loss >= 0


def test_step(linear_model, rng):
    """Run execution test for test_step."""
    init_func, apply_func = linear_model
    dmseloss = grad(mseloss)
    k1, k2 = split(rng)
    _, params = init_func(rng=k2, input_shape=(-1, 1))

    k1, k2, k3 = split(k1, 3)
    inputs = normal(k2, shape=(10, 1))
    outputs = normal(k3, shape=(10, 1))

    init, update, get_params = adam(step_size=0.001)
    global step
    step = partial(
        step,
        dloss_fun=dmseloss,
        apply_fun=apply_func,
        update_fun=update,
        get_params=get_params,
        inputs=inputs,
        outputs=outputs,
    )
    step = jit(step)

    state = init(params)
    for i in range(2):
        state = step(i, state)
