from random import shuffle as shuffle_order
from typing import Callable, Dict, List

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from pyprojroot import here
from tqdm import tqdm

from patch_gnn.graph import fluc_features, sasa_features


def generate_sorted_feature_dataframe(
    G: nx.Graph, funcs: List[Callable]
) -> pd.DataFrame:
    """
    Same implementation as jax patch-gnn.graph.generate_feature_dataframe
    but the nodes are sorted

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
    for n, d in sorted(G.nodes(data=True)):
        series = []
        # the followings are for each AA in a graph, make a pd.series
        # note that funcs are operations to do pd.series transformation for
        # each AA
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


def get_graph_and_feat_df(graphs: Dict, df: pd.DataFrame):
    """
    Get networkx graph and associated features
    :params graphs: a dictionary with key being "accession-sequence"
                    number and values being networkx graph object
    :params df: a dataframe with "accession-sequence" and oxidation
                value for each graph
    :returns
    out_graph: a list of networkx graphs intersecting entries in df
    feats: a list of dataframe, each with (n_node, m_features) in the
            same order as out_graph, n_node correspond to the actual
            number of nodes in corresponding out_graph while m_features
            are fixed for all graphs
    """
    aa_props = pd.read_csv(
        here() / "data/amino_acid_properties.csv", index_col=0
    )
    funcs = [
        lambda n, d: pd.Series(aa_props[d["residue_name"]], name=n),
        sasa_features,
        fluc_features,
    ]

    out_graph = []
    feats = []
    for acc in df["accession-sequence"]:
        g = graphs[acc]
        out_graph.append(g)
        feat = np.array(generate_sorted_feature_dataframe(g, funcs).values)
        feats.append(feat)
    return out_graph, feats, acc


def convert_networkx_to_dgl(
    one_networkx_graph: nx.Graph, one_graph_feature: np.ndarray
):
    """
    Convert one networkx graph to dgl graph, before conversion,
    change node names to integer manually (this step is important otherwise the new node indices won't
    match to the old ones)
    :param one_networkx_graph: one nx.Graph
    :param one_graph_feature: ndarray of shape (n_node, m_features), n_node could vary in length
                              n_node match to the correpsonding num of nodes in one_networkx_graph
    """
    # update the node label, manually map each node to a integer
    networkx_nodes = list(one_networkx_graph.nodes())
    # create mapping:#https://gitmemory.com/issue/dmlc/dgl/466/480467474,
    # https://stackoverflow.com/questions/35831648/networkx-shuffles-nodes-order
    #  change mapping manually before converting to dgl --> key: num, val: AA letter
    mapping = dict(
        zip(list(range(len(networkx_nodes))), sorted(networkx_nodes))
    )
    networkx_nodes = nx.relabel_nodes(one_networkx_graph, mapping=mapping)
    dgl_graph = dgl.from_networkx(one_networkx_graph)
    dgl_graph.ndata["feat"] = torch.from_numpy(one_graph_feature).float()
    return dgl_graph, mapping


class PatchGNNDataset(DGLDataset):
    def __init__(
        self,
        name: str = "ghesquire_2011",
        networkx_graphs: List[nx.Graph] = None,
        labels: np.ndarray = None,
        features: List[np.ndarray] = None,
        convert_networkx_to_dgl: callable = convert_networkx_to_dgl,
    ):
        """
        init this class
        :params networkx_graph: a list of graphs in networkx format
        :params labels: the target of prediction

        :features: a list, each element is a feature matrix of the corresponding graph from networkx_graph,
                    should be of shape (n_node, m_features) with no padding. n_node varies for each graph,
                    m_features is fixed for all graphs.

        :param convert_networkx_to_dgl: a function that converts networkx obj to dgl object
        """
        self.networkx_graphs = networkx_graphs
        self.features = features
        self.labels = torch.from_numpy(labels).float()
        self.nodenames = [
            list(sorted(graph.nodes())) for graph in networkx_graphs
        ]
        super().__init__(name=name)

    def process(self):
        # convert networkx_graphs to dgl graphs
        dgl_graphs = []
        dgl_graph_nodes = []
        for idx in tqdm(range(len(self.networkx_graphs))):
            one_graph, one_graph_nodes = convert_networkx_to_dgl(
                self.networkx_graphs[idx], self.features[idx]
            )
            dgl_graphs.append(one_graph)
            dgl_graph_nodes.append(one_graph_nodes)

        self.dgl_graphs = dgl_graphs
        self.nodenames = dgl_graph_nodes

    def __getitem__(self, idx):
        return self.dgl_graphs[idx], self.nodenames[idx], self.labels[idx]
        # return self.dgl_graphs[idx], self.nodenames[idx], torch.from_numpy(self.features[idx]).float(), self.labels[idx]

    def __len__(self):
        return len(self.dgl_graphs)


def collate_fn(dataset):
    """
    Randomly shuffle graph, names, labels in the same way for each batch.
    Use it along with `dgl` dataloader func, see an example here https://github.com/dmlc/dgl/issues/644#issuecomment-707158735

    """
    # not ideal if one wants to do shuffle and also want to retrive nodenames
    # https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
    # print(samples, type(samples))
    graphs, nodename_dict, labels = map(
        list, zip(*dataset)
    )  # combine each position into graphs, dict_names, labels
    joined_lst = list(
        zip(graphs, nodename_dict, labels)
    )  # join them for random shuffling
    shuffle_order(joined_lst)  # shuffle dataset in place
    new_graphs, new_nodename_dict, new_labels = zip(*joined_lst)  # unpack
    batch_graph = dgl.batch(new_graphs)
    return batch_graph, torch.tensor(new_labels)
