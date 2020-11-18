"""Patch-GNN layers."""
import jax.numpy as np
from jax import random, vmap
from jax.nn.initializers import glorot_normal


def MessagePassing(adjacency_weights_init=glorot_normal()):
    """
    Return init_fun, apply_fun pair for MessagePassing.

    :param A: An adjacency tensor of shape (n_nodes, n_nodes, n_adjacency_like_types)
    """

    def init_fun(rng, input_shape: tuple) -> tuple:
        """
        Initialize a Message Passing layer.

        We initialize an array of ``adjacency_weights``,
        which are of shape (n_adjacency_like_types, 1),
        so that we can elementwise multiply against the adjacency matrix
        to automatically learn what adjacency matrix is important.

        ``output_shape`` is identical to the input's shape.

        :param rng: The PRNGKey (from JAX)
            for random number generation _reproducibility_.
        :param input_shape: The shape of the graph's node feature matrix.
            Should be a tuple of (n_nodes, n_features, n_adjacencies)
        :returns: Tuple of (output_shape, params)
        """
        n_nodes, n_features, n_adjacencies = input_shape
        k1, k2 = random.split(rng)
        adjacency_weights = adjacency_weights_init(k1, (n_adjacencies, 1))
        return (n_nodes, n_features), (adjacency_weights)

    def apply_fun(params, inputs, **kwargs):
        """
        Apply message passing operation.

        We take the F matrix,
        and dot product against every adjacency-like matrix passed in.

        A `vmap` is needed here to vmap over all possible adjacencies.

        :param params: A vector of adjacency matrix weights.
        :param inputs: A 2-tuple of (adjacency tensor, node feature matrix).
        :returns: Weighted message passing between nodes and adjacency tensors.
        """
        adjacency_weights = params
        A, F = inputs
        mp = vmap(np.dot, in_axes=(-1, None), out_axes=(-1))(A, F)
        return np.squeeze(np.dot(mp, adjacency_weights))

    return init_fun, apply_fun


def GraphAverage():
    """Average all of the node features of a graph's node feature matrix."""

    def init_fun(rng, input_shape: tuple) -> tuple:
        """
        Initiailize parameters for GraphAverage layer.

        :param rng: A PRNGKey from JAX.
        :param input_shape: A 2-tuple of (n_nodes, n_features/n_activations)
        :returns: A 2-tuple of (n_features/n_activations, empty tuple)
        """
        output_dim = input_shape[-1]
        params = ()
        return (output_dim,), params

    def apply_fun(params, inputs: np.ndarray, **kwargs):
        """
        Average all node features.

        :param params: Should be an empty tuple.
        :param inputs: Array of inputs.
        """
        return np.mean(inputs, axis=0)

    return init_fun, apply_fun


def GraphSummation():
    """
    Return graph-level summation embedding.

    Performs a summation operation on node the feature matrix.
    """

    def init_fun(rng, input_shape: tuple) -> tuple:
        """
        Initialize parameters for GraphSummation layer.

        :param input_shape: (n_nodes, n_features/n_activations).
        :returns: Tuple of output_dim (n_features/n_activations, empty tuple)
        """
        output_dim = input_shape[-1]
        params = ()
        return (output_dim,), params

    def apply_fun(params, inputs: np.ndarray, **kwargs):
        """
        Sum up all node features.

        :param params: An empty tuple
        :param inputs: Node feature array for a graph.
        :returns: Summed up node feature array.
        """
        return np.sum(inputs, axis=0)

    return init_fun, apply_fun
