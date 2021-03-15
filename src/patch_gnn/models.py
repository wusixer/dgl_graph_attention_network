"""
Graph Neural Net Model
"""
from functools import partial

import jax.numpy as np
from jax import jit, vmap
from jax.experimental import stax
from jax.experimental.optimizers import adam
from jax.random import PRNGKey
from tqdm.auto import tqdm

from patch_gnn.layers import CustomGraphEmbedding, LinearRegression
from patch_gnn.training import mseloss, step

from .layers import GraphAttention, GraphSummation


class MPNN:
    """Shallow MPNN model in sklearn-compatible format.

    Single message passing step + linear regression on top.
    """

    def __init__(
        self,
        node_feature_shape,
        num_adjacency,
        num_training_steps: int = 100,
        optimizer_step_size=1e-5,
    ):
        """
        :param node_feature_shape: (num_nodes, num_feats)
        :param num_adjacency: number of adjacency-like matrices
        """
        model_init_fun, model_apply_fun = stax.serial(
            CustomGraphEmbedding(1024),
            LinearRegression(1),
        )
        self.model_apply_fun = model_apply_fun

        self.optimizer = adam(step_size=optimizer_step_size)

        output_shape, params = model_init_fun(
            PRNGKey(42), input_shape=(*node_feature_shape, num_adjacency)
        )

        self.params = params
        self.num_training_steps = num_training_steps
        self.state_history = []
        self.loss_history = []

    def fit(self, X, y):
        """Fit model.

        :param X: tuple(adjacency, node_features)
        :param y: vector(values to predict)
        """
        if len(y.shape) == 1:
            y = np.reshape(y, (-1, 1))
        init, update, get_params = self.optimizer
        training_step = partial(
            step,
            loss_fun=mseloss,
            apply_fun=self.model_apply_fun,
            update_fun=update,
            get_params=get_params,
            inputs=X,
            outputs=y,
        )
        training_step = jit(training_step)

        state = init(self.params)

        for i in tqdm(range(self.num_training_steps)):
            state, loss = training_step(i, state)
            self.state_history.append(state)
            self.loss_history.append(loss)

        self.params = get_params(state)
        return self

    def predict(self, X, checkpoint: int = None):
        """
        predict
        :param X: tuple(adjacency, node_features)
        """
        params = self.params
        if checkpoint:
            _, _, get_params = self.optimizer
            params = get_params(self.state_history[checkpoint])
        return vmap(partial(self.model_apply_fun, params))(X)


class DeepMPNN(MPNN):
    """Deep MPNN model in sklearn-compatible format.

    Single message passing step + additional hidden layer before linear regression on top.
    """

    def __init__(
        self,
        node_feature_shape,
        num_adjacency,
        num_training_steps: int = 100,
        optimizer_step_size=1e-5,
    ):
        """
        :param node_feature_shape: (num_nodes, num_feats)
        :param num_adjacency: number of adjacency-like matrices
        """
        model_init_fun, model_apply_fun = stax.serial(
            CustomGraphEmbedding(1024),
            # One hidden layer
            stax.Dense(512),
            stax.Relu,
            stax.Dense(1),
        )
        self.model_apply_fun = model_apply_fun

        self.optimizer = adam(step_size=optimizer_step_size)
        self.num_epochs = 100
        output_shape, params = model_init_fun(
            PRNGKey(42), input_shape=(*node_feature_shape, num_adjacency)
        )

        self.params = params
        self.num_training_steps = num_training_steps
        self.state_history = []
        self.loss_history = []


class DeepGAT(MPNN):
    """Deep GAT in sklearn-compatible format.

    We do one graph attention layer + a feed forward neural network.
    """

    def __init__(
        self,
        node_feature_shape,
        num_adjacency,
        num_training_steps: int = 100,
        optimizer_step_size=1e-5,
    ):
        """
        :param node_feature_shape: (num_nodes, num_feats)
        :param num_adjacency: number of adjacency-like matrices
        """
        model_init_fun, model_apply_fun = stax.serial(
            GraphAttention(n_output_dims=256),
            GraphSummation(),
            stax.Dense(64),
            stax.Relu,
            stax.Dense(16),
            stax.Relu,
            stax.Dense(1),
        )
        self.model_apply_fun = model_apply_fun

        self.optimizer = adam(step_size=optimizer_step_size)
        self.num_epochs = 100
        _, params = model_init_fun(
            PRNGKey(42), input_shape=(*node_feature_shape, num_adjacency)
        )

        self.params = params
        self.num_training_steps = num_training_steps
        self.state_history = []
        self.loss_history = []
