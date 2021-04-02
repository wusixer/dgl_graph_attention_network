"""Patch-GNN layers."""
from functools import partial

import jax.numpy as np
from jax import lax, nn, random, vmap
from jax._src.nn.functions import normalize
from jax.experimental import stax
from jax.nn.initializers import glorot_normal
from jax.random import normal


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
            n_adjacencies : n-degree adjancency like matrix
        :returns: Tuple of (output_shape, params)
        """
        n_nodes, n_features, n_adjacencies = input_shape
        k1, k2 = random.split(rng)
        adjacency_weights = adjacency_weights_init(k1, (n_adjacencies, 1))
        return (n_nodes, n_features), (adjacency_weights) # always return output_shape, parmas tuple

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
        A, F = inputs  # A: adjacency matrix, F: feature matrix
        # https://colinraffel.com/blog/you-don-t-know-jax.html#:~:text=minibatch%20of%20examples.-,jax.,there%20is%20only%20one%20argument.
        mp = vmap(np.dot, in_axes=(-1, None), out_axes=(-1))(
            A, F
        )  # in_axes= -1 means using the -1 index (a,b,c) would be c, of A and F when doing dot. shape (n_nodes, n_features, n_adjacency_like_matrics)
        return np.squeeze(
            np.dot(mp, adjacency_weights)
        )  # shape (n_nodes, n_features)

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
        output_dim = input_shape[-1]  # last dim of the input shape
        params = ()     # the params is empty because the layer doesn't have any parameters
        return (output_dim,), params

    def apply_fun(params, inputs: np.ndarray, **kwargs):
        """
        Sum up all node features. # by convention, the 1st param of apply_fun is params, but this layer doesn't require any param

        :param params: An empty tuple
        :param inputs: Node feature array for a graph. (n_nodes, n_features)
        :returns: Summed up node feature array. (n_features,)
        """
        return np.sum(inputs, axis=0)

    return init_fun, apply_fun


def CustomGraphEmbedding(n_output: int):
    """Return an embedding of a graph in n_output dimensions."""  # stax.serial linearly combines all the steps
    init_fun, apply_fun = stax.serial(
        MessagePassing(),
        stax.Dense(2048),
        stax.Sigmoid,
        GraphSummation(),
        stax.Dense(n_output),
    )
    return init_fun, apply_fun


def LinearRegression(num_outputs):
    """Linear regression layer."""
    init_fun, apply_fun = stax.serial(
        stax.Dense(num_outputs),
    )
    return init_fun, apply_fun


def LogisticRegression(num_outputs):
    """Logistic regression layer."""
    init_fun, apply_fun = stax.serial(
        stax.Dense(num_outputs),
        stax.Softmax,
    )
    return init_fun, apply_fun


def concat_nodes(node1, node2):
    """Concatenate two nodes together."""
    return np.concatenate([node1, node2])


def concatenate(node: np.ndarray, node_feats: np.ndarray):
    """Concatenate node with each node in node_feats.

    Behaviour is as follows.
    Given a node with features `f_0` and stacked node features
    `[f_0, f_1, f_2, ..., f_N]`,
    return a stacked concatenated feature array:
    `[(f_0, f_0), (f_0, f_1), (f_0, f_2), ..., (f_0, f_N)]`.

    :param node: A vector embedding of a single node in the graph.
        Should be of shape (n_input_features,)
    :param node_feats: Stacked vector embedding of all nodes in the graph.
        Should be of shape (n_nodes, n_input_features)
    :returns: A stacked array of concatenated node features.
    """
    return vmap(partial(concat_nodes, node))(node_feats)


def concatenate_node_features(node_feats):
    """Return node-by-node concatenated features.

    Given a node feature matrix of shape (n_nodes, n_features),
    this returns a matrix of shape (n_nodes, n_nodes, 2*n_features).
    """
    outputs = vmap(partial(concatenate, node_feats=node_feats))(node_feats)
    return outputs


# def normalize_if_nonzero(p_vect):
#     def if_zero(p):
#         return p

#     def if_nonzero(p):
#         return p / np.sum(p)

#     return lax.cond(np.sum(p_vect) == 0, if_zero, if_nonzero, p_vect)


def GraphAttention(n_output_dims: int, w_init=normal, a_init=normal):
    """Graph attention layer.

    Expects a 2-tuple of (adjacency matrix, node embeddings) as the input.
    """

    def init_fun(key, input_shape):
        """Graph attention layer init function."""
        _, n_features, _ = input_shape
        k1, k2, k3 = random.split(key, 3)
        w_node_projection = (
            w_init(key=k1, shape=(n_features, n_output_dims)) * 0.01
        )
        a_node_concat = a_init(key=k2, shape=(n_output_dims * 2,)) * 0.01
        node_feat_attn = w_init(key=k3, shape=(n_features,))
        return (n_output_dims,), (
            w_node_projection,
            a_node_concat,
            node_feat_attn,
        )

    def apply_fun(params, inputs, **kwargs):
        """Graph attention layer apply function."""
        attention, node_projection = node_attention(params, inputs)

        final_output = np.dot(attention, node_projection)
        return final_output

    return init_fun, apply_fun


def node_attention(params, inputs):
    """Compute node-by-node attention weights."""
    (w, a, nfa), (adjacency_matrix, node_embeddings) = params, inputs
    # w shape:   (n_features, n_output_dims)
    # a shape:   (n_output_dims * 2,)
    # nfa shape: (n_features,)

    # Firstly, we project the nodes to a different dimensioned space
    node_feat_attn = vmap(partial(np.multiply, np.abs(nfa)))(node_embeddings)
    node_projection = np.dot(node_feat_attn, w)

    # Next, we concatenate node features together
    # into an (n_nodes, n_nodes, 2 * n_output_dims) tensor.
    node_by_node_concat = concatenate_node_features(node_projection)

    # Then, we project the node-by-node concatenated features
    # down to the attention dims and apply a nonlinearity,
    # giving an (n_node, n_node) matrix.
    projection = np.dot(node_by_node_concat, a)
    output = nn.leaky_relu(projection, negative_slope=0.1)

    # Squeeze is applied to ensure we have 2D matrices.
    # Mask out irrelevant values early on.
    output = np.squeeze(output)

    # Finally, compute attention mapping for message passing.
    # attention = vmap(nn.softmax)(output) * np.squeeze(adjacency_matrix)
    attention = output * np.squeeze(adjacency_matrix)

    # Let's now do some experiments below.
    # If we add back this line:
    # attention = vmap(normalize_if_nonzero)(attention)
    # Then we get NaN issues.
    return attention, node_projection
