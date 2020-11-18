"""Tests for graph-related functions."""
from random import choice

import networkx as nx
import pandas as pd
import pytest
import xarray as xr

from patch_gnn.graph import (
    adjacency_matrix,
    extract_neighborhood,
    generate_feature_dataframe,
    generate_patches,
    identity_matrix,
    laplacian_matrix,
    stack_adjacency_tensors,
    stack_feature_tensors,
    to_adjacency_xarray,
)


@pytest.mark.parametrize("num_nodes", range(10, 30))
@pytest.mark.parametrize("radius", range(1, 4))
def test_extract_neighborhood(num_nodes, radius):
    """Parametrized test for extract_neighborhood."""
    G = nx.erdos_renyi_graph(n=num_nodes, p=0.1)

    node = choice(list(G.nodes()))

    subG = extract_neighborhood(G, n=node, r=radius)

    assert node in set(subG.nodes())
    for n in subG.nodes():
        assert nx.shortest_path_length(G, n, node) <= radius


@pytest.fixture
def adjacency_funcs():
    """Generate list of adjacency functions."""
    funcs = [
        adjacency_matrix,
        identity_matrix,
        laplacian_matrix,
    ]
    return funcs


@pytest.mark.parametrize("num_nodes", range(10, 30))
def test_to_adjacency_xarray(num_nodes, adjacency_funcs):
    """Test construction of xarray adjacency tensor."""
    G = nx.erdos_renyi_graph(n=num_nodes, p=0.1)
    da = to_adjacency_xarray(G, adjacency_funcs)
    assert isinstance(da, xr.DataArray)
    assert da.shape == (len(G), len(G), len(adjacency_funcs))


def integer_graph(num_nodes: int) -> nx.Graph:
    """
    Generate integer graph with number of nodes specified.

    An "integer graph" is just a graph with integers as nodes.
    We add in an attribute field "integer" that is equal to the node name,
    so that we have metadata on it too.

    :param num_nodes: Number of nodes to generate.
    """
    G = nx.Graph()
    nodes = [i for i in range(num_nodes)]
    G.add_nodes_from(nodes)
    for n in G.nodes():
        G.nodes[n]["integer"] = n
    return G


@pytest.mark.parametrize("num_nodes", range(10, 30))
def test_generate_feature_dataframe(num_nodes):
    """Test generation of feature dataframe from node metadata."""
    G = integer_graph(num_nodes)
    node_df = generate_feature_dataframe(
        G, funcs=[lambda n, d: pd.Series(d, name=n)]
    )
    assert len(node_df) == len(G)
    assert "integer" in node_df.columns


@pytest.fixture
def integer_graphs():
    """Return two dummy integer graphs."""
    G1 = integer_graph(30)
    G2 = integer_graph(25)
    return [G1, G2]


def test_stack_feature_tensors(integer_graphs):
    """Test the stacking of feature tensors together."""
    graphs = integer_graphs
    maxsize = max(len(g) for g in graphs)

    funcs = [lambda n, d: pd.Series(d, name=n)]

    feats = stack_feature_tensors(graphs, funcs)

    assert feats.shape == (len(graphs), maxsize + 1, 1)


def test_stack_adjacency_tensors(integer_graphs, adjacency_funcs):
    """Test the stacking of adjacency tensors together."""
    graphs = integer_graphs
    maxsize = max(len(g) for g in graphs)
    arr = stack_adjacency_tensors(graphs, adjacency_funcs)
    assert arr.shape == (
        len(graphs),
        maxsize + 1,
        maxsize + 1,
        len(adjacency_funcs),
    )


@pytest.mark.parametrize("neighborhood_size", range(1, 4))
def test_generate_patches(neighborhood_size):
    """Test patch generation from a graph equals number of nodes in graph."""
    G = nx.erdos_renyi_graph(n=30, p=0.1)

    patches = generate_patches(G, neighborhood_size)
    assert len(patches) == len(G)
