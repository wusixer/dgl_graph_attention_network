"""Utilities for manipulating a protein graph."""
from functools import wraps
from typing import Callable, Hashable, List

import jax.numpy as np
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr


def extract_neighborhood(G: nx.Graph, n: Hashable, r: int) -> List[Hashable]:
    """
    Extract neighborhood of radius `r` for a given node `n`.

    Used to generate subgraphs of the original graph
    that contain only a subset of nodes
    within a radius of that given node.
    The original node is included in the list of nodes.

    Modified from SO: https://stackoverflow.com/a/22744470/1274908

    ## Parameters
    - `G`: A NetworkX Graph
    - `n`: A node in the graph `G`
    - `r`: The radius (number of degrees of separation)
        to extract subgraph.
    """
    path_lengths = nx.single_source_dijkstra_path_length(G, n)

    nodes = [node for node, length in path_lengths.items() if length <= r]
    return G.subgraph(nodes)


def format_adjacency(func: Callable):
    """
    Format adjacency matrix returned from diffusion-matrix function.

    Intended to be used as a decorator.

    Example:

    .. code-block:: python

        @format_adjacency
        def adjacency_matrix(G, power=1):
            return np.linalg.matrix_power(
                np.asarray(nx.adjacency_matrix(G)),
                power
            )

        funcs = []
        for power in range(5):
            funcs.append(
                partial(
                    adjacency_matrix,
                    power=power,
                    name=f"adjacency_{power}",
                )
            )
    """

    @wraps(func)
    def inner(G: nx.Graph, *args, **kwargs):
        """Inner function that gets called after being wrapped."""
        name = kwargs.pop("name", None)
        if name is None:
            name = func.__name__

        adj = func(G, *args, **kwargs)
        expected_shape = (len(G), len(G))
        if adj.shape != expected_shape:
            raise ValueError(
                "Adjacency matrix is not shaped correctly, "
                f"should be of shape {expected_shape}, "
                f"instead got shape {adj.shape}."
            )
        #### THE MAGIC HAPPENS HERE #####
        adj = np.expand_dims(adj, axis=-1)
        adj = xr.DataArray(adj)
        nodes = list(G.nodes())
        return xr.DataArray(
            adj,
            dims=["n1", "n2", "name"],
            coords={"n1": nodes, "n2": nodes, "name": [name]},
        )

    return inner


@format_adjacency
def adjacency_matrix(G: nx.Graph, power: int = 1) -> np.ndarray:
    """Return adjacency matrix raised to a power."""
    a = nx.adjacency_matrix(G).todense()
    a = np.asarray(a)
    a = np.linalg.matrix_power(a, power)
    return a


@format_adjacency
def identity_matrix(G: nx.Graph) -> np.ndarray:
    """Effectively just the self-loops."""
    return np.eye(len(G))


@format_adjacency
def laplacian_matrix(G: nx.Graph) -> np.ndarray:
    """Graph laplacian matrix."""
    A = nx.linalg.laplacian_matrix(G)
    return np.asarray(A.todense())


def to_adjacency_xarray(G: nx.Graph, funcs: List[Callable]):
    """
    Generate adjacency tensor in xarray.

    Uses the collection of functions in ``funcs``
    to build an xarray DataArray
    that houses the resulting "adjacency tensor".
    Each function in ``funcs`` should return
    an xarray DataArray with the dimensions
    ``n1``, ``n2``, and ``name``,
    giving rise to an array
    that is of shape ``(num_nodes, num_nodes, 1)``.

    We return xarray DataArrays, to make inspecting the data easy.
    To pass the underlying tensor data into other libraries,
    such as JAX, PyTorch and TensorFlow,
    you can ask for ``data_array.data``
    to get the underlying NumPy array.

    Generate adjacency tensor for a graph as an `xarray.DataArray`.

    :param G: A NetworkX-compatible Graph object.
    :param funcs: A list of callables that take in G and return an xr.DataArray
    :returns: An XArray DataArray which is of shape (n_nodes, n_nodes, n_funcs).
    """
    mats = []
    for func in funcs:
        mats.append(func(G))
    da = xr.concat(mats, dim="name")
    return da


def prep_adjacency_matrix(A: np.ndarray, size: int) -> np.ndarray:
    """
    Pad adjacency matrices to maximum number of nodes.

    Taken from here: https://ericmjl.github.io/essays-on-data-science/machine-learning/message-passing/#implementation-4-batched-padded-matrix-multiplication

    :param A: Adjacency-like matrices, of shape (n_nodes, n_nodes, 1)
    :param size: Upper-bound number of nodes to pad two axes of A to.
    """
    # We do need the 3rd dimension to be padded by nothing!
    return np.pad(
        A,
        [
            (0, size - A.shape[0]),
            (0, size - A.shape[0]),
            (0, 0),  # pad third dimension by nothing!
        ],
    )


def generate_feature_dataframe(
    G: nx.Graph, funcs: List[Callable]
) -> pd.DataFrame:
    """
    Return a pandas DataFrame representation of node metadata.

    `funcs` has to be list of callables whose signature is

        f(n, d) -> pd.Series

    where `n` is the graph node,
    `d` is the node metadata dictionary.
    The function must return a pandas Series whose name is the node.

    Example function:

    .. code-block:: python

        def x_vec(n: Hashable, d: Dict[Hashable, Any]) -> pd.Series:
            return pd.Series({"x_coord": d["x_coord"]}, name=n)

    One fairly strong assumption is that each func
    has all the information it needs to act
    stored on the metadata dictionary.

    If you need to reference an external piece of information,
    such as a dictionary to look up values,
    set up the function to accept the dictionary,
    and use `functools.partial`
    to "reduce" the function signature to just `(n, d)`.

    An example below:

    .. code-block:: python

        from functools import partial
        def get_molweight(n, d, mw_dict):
            return pd.Series({"mw": mw_dict[d["amino_acid"]]}, name=n)
        mw_dict = {"PHE": 165, "GLY": 75, ...}
        get_molweight_func = partial(get_molweight, mw_dict=mw_dict)
        generate_feature_dataframe(G, [get_molweight_func])

    The `name=n` piece is important;
    the `name` becomes the row index in the resulting dataframe.
    The series that is returned from each function
    need not only contain one key-value pair.

    You can have two or more, and that's completely fine;
    each key becomes a column in the resulting dataframe.

    A key design choice: We default to returning DataFrames,
    to make inspecting the data easy,
    but for consumption in tensor libraries,
    you can turn on returning a NumPy array
    by switching `return_array` to True.

    :param G: A NetworkX-compatible graph object.
    :param funcs: A list of functions that return a pandas Series.
    :returns: A pandas DataFrame.
    """
    matrix = []
    for n, d in G.nodes(data=True):
        series = []
        for func in funcs:
            res = func(n, d)
            if res.name != n:
                raise NameError(
                    f"function {func.__name__} returns a series "
                    "that is not named after the node."
                )
            series.append(res)
        matrix.append(pd.concat(series))

    return pd.DataFrame(matrix)


def prep_features(F, size):
    """
    Pad features to a particular size.

    Taken from here: https://ericmjl.github.io/essays-on-data-science/machine-learning/message-passing/#implementation-4-batched-padded-matrix-multiplication
    :param F: Features matrix. Of shape (n_nodes, n_features)
    :param size: Upper-bound number of nodes to pad F matrix to.
    """
    return np.pad(F, [(0, size - F.shape[0]), (0, 0)])


def generate_patches(G: nx.Graph, neighborhood_size: int) -> List[nx.Graph]:
    """
    Generate graph patches.

    This function generates neighborhood patches for each node,
    with each of a ``neighborhood_size`` degree of separation away

    :param G: A NetworkX graph.
    :param neighborhood_size: The number of degrees of separation
        from each node to consider inside the topological patch.
    """
    graphs = []
    for node in G.nodes():
        subg = extract_neighborhood(G, node, neighborhood_size)
        graphs.append(subg)
    return graphs


def stack_feature_tensors(
    Gs: List[nx.Graph], funcs: List[Callable], max_length: int = None
) -> np.ndarray:
    """
    Stack feature matrices for all graphs in a collection of graphs.

    :param Gs: The collection of graphs to make a feature tensor for.
    :param func: The graph node featurization funcs.
    :param max_length: For the case where the maximum length is expensive to calculate,
        pre-calculate this and pass it in here.
    :returns: A 3-dimensional numpy array of shape (n_graphs, n_nodes, n_feats)
    """
    feats = []

    if max_length is None:
        max_length = max(len(g) for g in Gs)

    for g in Gs:
        F = generate_feature_dataframe(g, funcs=funcs).values
        F = prep_features(F, max_length + 1)
        feats.append(F)
    return np.stack(feats)


def stack_adjacency_tensors(Gs: List[nx.Graph], funcs: List[Callable]):
    """
    Stack adjacency tensors for each graph in G into a big tensor.

    :param Gs: A list of NetworkX graphs.
    :param
    """
    max_length = max(len(g) for g in Gs)

    adjs = []
    for g in Gs:
        A = to_adjacency_xarray(g, funcs=funcs).data
        A = prep_adjacency_matrix(A, max_length + 1)
        adjs.append(A)
    return np.stack(adjs)
