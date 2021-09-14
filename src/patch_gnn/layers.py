"""Patch-GNN layers."""
from functools import partial
from typing import Tuple

import jax.numpy as np
import numpy
import pandas as pd
from jax import lax, nn, random, vmap
from jax._src.nn.functions import normalize
from jax.experimental import stax
from jax.nn.initializers import glorot_normal
from jax.random import normal


def MessagePassing(adjacency_weights_init=glorot_normal()):
    """
    Return init_fun, apply_fun pair for MessagePassing. Message passing is A*F

    :param A: An adjacency tensor of shape (n_nodes, n_nodes, n_adjacency_like_types)
    """

    def init_fun(rng, input_shape: Tuple[int, int, int]) -> tuple:
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
        (
            n_nodes,
            n_features,
            n_adjacencies,
        ) = input_shape  # you tell the computer what input_shape is, not inferred from inputs in apply_fun
        k1, k2 = random.split(rng)
        adjacency_weights = adjacency_weights_init(
            k1, (n_adjacencies, 1)
        )  # weighted sum in all adjacency-like matrics
        return (n_nodes, n_features), (
            adjacency_weights
        )  # always return output_shape, parmas tuple

    def apply_fun(params, inputs: Tuple[np.ndarray, np.ndarray], **kwargs):
        """
        Apply message passing operation.

        We take the F matrix,
        and dot product against every adjacency-like matrix passed in.

        A `vmap` is needed here to vmap over all possible adjacencies.
        A is of shape (n_node, n_node, n_adjacency), F is of shape (n_node, n_features)
        we have 7 adjacency-like matrices of shape A' (5 nodes, 5 nodes),
        and our feature matrix is of shape F (5 node, 13 features),
        we would have 7 individual operations (1 for each adjacency like matrix),
        so that it is 7 x (5, 5)* (5, 13), this will return 7 x (5, 13) matrics,
        and then we stack by the axis of 7, where we get (5,13,7) -> (n_node, n_feature, n_adjacnecy)
        ,which is what the following code to get `mp`. np.dot applies the weighted of each
        adjacency matrix to mp, which will return (n_node, n_feature, 1), np.squeeze removes
        1 out, so we have (n_node, n_features)

        :param params: A vector of adjacency matrix weights.
        :param inputs: A 2-tuple of (adjacency tensor, node feature matrix).
        :returns: Weighted message passing between nodes and adjacency tensors.
                  of shape (n_node, n_features)
        """
        adjacency_weights = params
        A, F = inputs  # A: adjacency matrix, F: feature matrix
        # https://colinraffel.com/blog/you-don-t-know-jax.html#:~:text=minibatch%20of%20examples.-,jax.,there%20is%20only%20one%20argument.
        mp = vmap(np.dot, in_axes=(-1, None), out_axes=(-1))(
            A, F
        )  # in_axes= -1 means using the -1 index (a,b,c) would be c, of A and F when doing dot. shape (n_nodes, n_features, n_adjacency_like_matrics)
        # np.squeeze is to remove any axes of length 1 from the input matrix
        # if we only have 1 adjancency matrix, it will return (n_node, n_features)
        return np.squeeze(
            np.dot(mp, adjacency_weights)
        )  # shape (n_nodes, n_features) get the weighted sum of all message passing operations from each
        # adjacency-like matrices

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
        params = ()  # the params is empty because the layer doesn't have any parameters
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


# --- comment this out so that the user can have
# --- more freedom to define it in the model
# def CustomGraphEmbedding(n_output: int):
#    """Return an embedding of a graph in n_output dimensions."""  # stax.serial linearly combines all the steps
#    init_fun, apply_fun = stax.serial(
#        MessagePassing(),
#        stax.Dense(2048),
#        stax.Sigmoid,
#        GraphSummation(),
#        stax.Dense(n_output),
#    )
#    return init_fun, apply_fun


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
    `[f_0, f_1, f_2, ..., f_N]`, # of shape (n_features, )
    return a stacked concatenated feature array:
    `[(f_0, f_0), (f_0, f_1), (f_0, f_2), ..., (f_0, f_N)]`. #of shape ((2*n_feature),)

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

    # def concatenate_node_features_v2(node_feats):
    #    """Return node-by-node concatenated features.
    #    Given a node feature matrix of shape (n_nodes, n_features),
    #    this returns a matrix of shape (n_nodes, n_nodes, 2*n_features).
    #    """
    #    outputs = vmap(partial(concatenate, node_feats=node_feats))(node_feats)
    #    return outputs

    # def normalize_if_nonzero(p_vect):
    #     def if_zero(p):
    #         return p

    #     def if_nonzero(p):
    #         return p / np.sum(p)

    #     return lax.cond(np.sum(p_vect) == 0, if_zero, if_nonzero, p_vect)

    # def softmax_on_non_zero(attention, adj):
def softmax_on_non_zero(attention):
    """
    This function works on toy data however, it will encounter 0
    division in practice
    Apply softmax normalization on a matrix row-wise and ignore
    the 0 elements. adapt from here https://discuss.pytorch.org/t/apply-mask-softmax/14212/7

    Note that this function is under the assumption that
    it is of very little possibllity that an attention (before softmax normalizatoin)
    of a legit edge would be 0.

    :param attention: a column vector from a matrix output from `leaky_relu`
    :param adj: adjancency matrix to mask out the attention matrix
    :return softmax normalized attention
    """
    # option 1: need to get the non-zero part matrix out to prevent
    # 0 division, but still, there could be negatives so the sum would be lower than the max
    # which means the attnetion could be >1 for some nodes
    # need to exponetiate it before the sum
    attention_sum = np.sum(attention,axis=1)
    #according to https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nonzero.html
    # we cannot do vmap with jnp.nonzero implementation, for the @jit requires each graph 
    # has the same # of non-zero element. so can't do 
    #  `vmap(partial(node_attention, model.params[0]))(train_graph)`
    # but other than that it works
    max_node = len(np.nonzero(attention_sum)[0]) 
    a_softmax = attention[:max_node, :max_node]/attention_sum[:max_node][:,None]
    pad = attention.shape[0] - a_softmax.shape[0]
    a_softmax = np.pad(
            a_softmax,
            [(0, pad), (0, pad)],
        )
    return a_softmax
    #    a_softmax = attention/attention_sum
    #    return a_softmax
    # option 2: get https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError error
    # get the row-wise max entry
    # attention_max = np.max(attention,axis=1).reshape(attention.shape[0],1)
    # deduct row-wise max entry from each entry
    # attention_exp = np.exp(attention-attention_max)
    ## apply adjancency mask, so that edges with no info is zero-out
    # attention_exp = attention_exp * (adj) # this step masks
    # if not np.all(np.sum(attention_exp,axis=1)):
    #    print("0 in denominator")
    # a_softmax = attention_exp / (np.sum(attention_exp,axis=1)[:,None])
    # return a_softmax

    # def softmax_on_non_zero(vect):
    """
    This function is wrong!!
    Apply softmax normalization on a vector and ignore the 0 values
    For example: [-0.3, 0 , 0] denotes the attention of a node on three
    nodes, this node has no value on node 2 or 3, and only -0.3 value on
    itself, the softmax will return [1,0,0], given "0" doesn't participate in
    softmax calculation.

    Note that this function is under the assumption that
    it is of very little possibllity that an attention (before softmax normalizatoin)
    of a legit edge would be 0.

    :param vect: a column vector from a matrix output from `leaky_relu`
    :return a list with softmax normalized values
    """


#    return np.where(vect != 0, nn.softmax(vect), vect)


def get_norm_attn_matrix(atten_after_relu, graph):
    """
    Apply softmax normalization on a matrix of shape (n_max_node, n_max_node).
    Only mask out the nodes that do not exist and then apply softmax on
    presented nodes. n_max_nodes denote the number of nodes after padding
    with the graph that has the highest number of nodes

        e.g
            atten_after_relu = nn.leaky_relu(projection, negative_slope=0.1)
            attention = get_norm_attn_matrix(atten_after_relu)

    :param atten_after_relu: the attention matrix coming out from `leaky_relu`
        of shape (n_max_node, n_max_node)
    :param graph: dict[nx.graph] of associated protein patch, should be len of 1
    and match to the same graph for atten_after_relu.
    ie the input of graph can be {str: nx.graph}
    {'Q9H0U4-MGPGAASGGERPNLK', graphs['Q9H0U4-MGPGAASGGERPNLK']}

    :returns softmax_on_atten: a ndarrau of shape (n_max_node, n_max_node)
    with nonzero values only on the parts with actual nodes
    denoting how much each node pays attention to other nodes. The attention
    of each node on other node should sum up to 1

    """
    # apply mask to previous atten matrix of this patch
    num_nodes = len(list(graph.values())[0].nodes())
    atten_after_relu_modified = atten_after_relu[:num_nodes, :num_nodes]
    # apply softmax on this matrix
    softmax_on_atten = vmap(nn.softmax)(atten_after_relu_modified)
    pad = atten_after_relu.shape[1] - softmax_on_atten.shape[0]
    softmax_on_atten = np.pad(
        atten_after_relu_modified,
        [(0, pad), (0, pad)],
    )
    return softmax_on_atten


def GraphAttention(n_output_dims: int, w_init=normal, a_init=normal):
    """Graph attention layer.

    Expects a 2-tuple of (adjacency matrix, node embeddings) as the input.
    """

    def init_fun(key, input_shape):
        """Graph attention layer init function."""
        (
            _,
            n_features,
            _,
        ) = input_shape  # this is to keep the API consistent, it could also be n_features = input_shape to not make it consistent
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
        # get attention (n_max_node, n_max_node), with edge value 0
        # means there is no connection between two nodes
        attention, node_projection = node_attention(params, inputs)

        final_output = np.dot(
            attention, node_projection
        )  # (n_node, output_dim)
        return final_output

    return init_fun, apply_fun


# def GraphAttention_v2(n_output_dims: int, w_init=normal, a_init=normal):
#    """Graph attention layer.#
#
#    Expects a 2-tuple of (node_feats, graphs) as the input.
#    """
#
#    def init_fun(key, input_shape):
#        """Graph attention layer init function."""
#        _, n_features, _ = input_shape
#        k1, k2 = random.split(key, 2)
#        w_node_projection = (
#            w_init(key=k1, shape=(n_features, n_output_dims)) * 0.01
#        )
#        a_node_concat = a_init(key=k2, shape=(n_output_dims * 2,)) * 0.01

#        return (n_output_dims,), (
#            w_node_projection,
#            a_node_concat,
#        )

#    def apply_fun(params, inputs, **kwargs):
#        """Graph attention layer apply function."""
#        norm_attention =  node_attention_v2(params, inputs)
#        return norm_attention

#    return init_fun, apply_fun


def node_attention(params, inputs):
    """Compute node-by-node attention weights."""
    (w, a, nfa), (adjacency_matrix, node_embeddings) = params, inputs
    # w shape:   (n_features, n_output_dims)
    # a shape:   (n_output_dims * 2,) #the attention mechanism a is a single-layer feedforward neural network,
    # parametrized by a weight vector `a` with 2*n_output_dims dimension, I think a shape should be (n_node, n_output_dims * 2)
    # nfa shape: (n_features,) #1d array
    # adjacency_matrix : (n_node, n_node, n_adjacency_like_matrices)
    # node embeddings: (n_node, n_features) #2d array

    # Firstly, we project the nodes to a different dimensioned space
    # --- why this is np.abs, is it to keep the + or - sign ?
    # --- node_feat_attn should return shape (n_node, n_node)
    node_feat_attn = vmap(partial(np.multiply, np.abs(nfa)))(node_embeddings)
    # --- node_feat_attn = vmap(partial(np.multiply, np.abs(num_of_node)))(node_embeddings)
    # --- node_projection = np.dot(node_feats, w) # node_feats is (n_node, n_features)
    node_projection = np.dot(
        node_feat_attn, w
    )  # should be of shape (n_node, n_output_dims), but is shape (n_features, n_output_dims)

    # Next, we concatenate node features together
    # into an (n_nodes, n_nodes, 2 * n_output_dims) tensor.
    # --- the outcome here is (n_feature, n_feature, 2*n_output_dims)
    # --- or maybe the outcome here should be (n_node, 2*n_output_dims),*****
    # --- b/c for each node x we have [(fx, f0), (fx, f1), (fx,f2)... (fx, fn)]
    # --- then for all nodes, we have n_node of list above, so it's n_node * 2*n_output_dims
    node_by_node_concat = concatenate_node_features(
        node_projection
    )  # concatenate_node_features takes the input of (n_nodes, n_features)

    # Then, we project the node-by-node concatenated features
    # down to the attention dims and apply a nonlinearity,
    # giving an (n_node, n_node) matrix.
    projection = np.dot(node_by_node_concat, a)
    atten_leaky_relu = nn.leaky_relu(projection, negative_slope=0.1)#******
    #atten_tanh = nn.hard_tanh(projection)  # --works
    # Squeeze is applied to ensure we have 2D matrices.
    atten_leaky_relu = np.squeeze(atten_leaky_relu)  # (n_node, n_node)****

    # Finally, compute attention mapping for message passing.
    # this means we do element-wise mulitiplication of to maks out
    # the edges that are not connected to each other
    # attention = vmap(nn.softmax)(output) * np.squeeze(adjacency_matrix)
    attention = atten_leaky_relu * np.squeeze(#****
                adjacency_matrix#****
                )  # -- might not need to use adjacency matrix*****
    #norm_attention = atten_tanh * np.squeeze(adjacency_matrix)

    # apply softmax norm
    norm_attention = softmax_on_non_zero(attention)#******
    # np.squeeze on adjacency_matrix so it's (n_node, n_node)
    # norm_attention = softmax_on_non_zero(attention)#, np.squeeze(adjacency_matrix))
    # norm_attention = vmap(partial(softmax_on_non_zero, adj =adjacency_matrix ))(attention)
    return norm_attention, node_projection


def node_attention_v2(params, inputs):
    """
    Compute node-by-node attention weights and without using triangle
    matrix mask. This function works the same way as node_attention,
    the difference is that the input is (node_feats, graphs) rather
    than (adjacency_matrix, node_embeddings)

    """
    (w, a), (node_feats, graphs) = params, inputs
    # node_feats: (n_nodes, n_input_features)
    # w shape:   (n_features, n_output_dims)
    # a shape:   (n_output_dims * 2, 1)
    # graphs: dict[nx.graph], has the same amount keys as the number of graphs.
    # graphs also need to have the same order as protein patches

    # Firstly, we project the nodes to a different dimensioned space
    node_projection = np.dot(
        node_feats, w
    )  # node_feats is (n_node, n_features)

    # Next, we concatenate node features together
    # into an (n_nodes, n_nodes, 2 * n_output_dims) tensor.
    # b/c for each node x we have [(fx, f0), (fx, f1), (fx,f2)... (fx, fn)], which is n_node * (2*n_output)
    # then for all nodes, we have n_node of list above, so it's n_node * n_node * 2*n_output_dims
    node_by_node_concat = concatenate_node_features(
        node_projection
    )  # concatenate_node_features takes the input of (n_nodes, n_features)

    # Then, we project the node-by-node concatenated features
    # down to the attention dims and apply a nonlinearity,
    # giving an (n_node, n_node) matrix.
    projection = np.dot(node_by_node_concat, a)
    output = nn.leaky_relu(projection, negative_slope=0.1)

    # Finally, compute softmax on outputs
    # 1.take output, use graphs to marks out none-existing node for each graph
    # 2. apply softmax for on exisiting node, to get attention paid from one node to the rest
    # 3. apply padding so that the attention is the same shape as output
    norm_attention = []
    for graph_num in range(output.shape[0]):
        norm_attention_i = get_norm_attn_matrix(
            output[graph_num],
            {list(graphs.keys())[graph_num]: list(graphs.values())[graph_num]},
        )
        norm_attention.append(norm_attention_i)
    norm_attention = np.asarray(norm_attention)

    return norm_attention
