"""
Tests for patch_gnn layers.

Most of the tests here simply check that the output shape is as expected.
"""
from functools import partial

import networkx as nx
import pytest
from jax import random
from jax.experimental import stax
from proteingraph.conversion import generate_adjacency_tensor

from patch_gnn.graph import adjacency_matrix, identity_matrix, laplacian_matrix
from patch_gnn.layers import GraphAverage, GraphSummation, MessagePassing


@pytest.fixture
def adj_feat_matrices():
    """Return adjacency and feature matrices."""
    key = random.PRNGKey(42)
    G = nx.erdos_renyi_graph(n=100, p=0.2)
    adjacencies = [
        identity_matrix,
        partial(adjacency_matrix, power=1, name="adjacency_1"),
        partial(adjacency_matrix, power=2, name="adjacency_2"),
        partial(adjacency_matrix, power=3, name="adjacency_3"),
        laplacian_matrix,
    ]
    A = generate_adjacency_tensor(G, funcs=adjacencies, return_array=True)
    _, k2 = random.split(key)
    F = random.normal(k2, shape=(len(G), 30))
    return A, F


def test_MessagePassing(adj_feat_matrices):
    """Test MessagePassing layer."""
    key = random.PRNGKey(42)
    A, F = adj_feat_matrices

    init_fun, apply_fun = stax.serial(MessagePassing())
    _, k2 = random.split(key)
    output_shape, params = init_fun(k2, input_shape=(*F.shape, A.shape[-1]))
    output = apply_fun(params, inputs=(A, F))
    assert output.shape == output_shape


def test_GraphAverage(adj_feat_matrices):
    """Test GraphAverage layer."""
    key = random.PRNGKey(42)
    A, F = adj_feat_matrices
    model_init_fun, model_apply_fun = stax.serial(
        MessagePassing(),
        GraphAverage(),
    )

    output_shape, params = model_init_fun(
        key, input_shape=(*F.shape, A.shape[-1])
    )

    output = model_apply_fun(params, inputs=(A, F))
    assert output.shape == output_shape


def test_GraphSummation(adj_feat_matrices):
    """Test GraphSummation layer."""
    key = random.PRNGKey(42)
    A, F = adj_feat_matrices
    model_init_fun, model_apply_fun = stax.serial(
        MessagePassing(),
        GraphSummation(),
    )

    output_shape, params = model_init_fun(
        key, input_shape=(*F.shape, A.shape[-1])
    )

    output = model_apply_fun(params, inputs=(A, F))
    assert output.shape == output_shape
