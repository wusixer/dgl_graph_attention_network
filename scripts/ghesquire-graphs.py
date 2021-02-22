"""Generate radius 1 graphs for Ghesquire dataset."""

import os
import pickle as pkl
from functools import partial

import janitor
import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from multipledispatch import dispatch
from proteingraph import read_pdb
from pyprojroot import here
from tqdm.auto import tqdm

from patch_gnn.data import load_ghesquire
from patch_gnn.graph import extract_neighborhood, met_position

models_path = here() / "data/ghesquire_2011/models"
protein_models = os.listdir(models_path)


data = load_ghesquire()
data["accession"] = data["accession"].fillna(method="ffill")


@dispatch(float)
def split_delimiter(x, delimiter=";"):
    """Split delimiter helper function for floats."""
    return x


@dispatch(str)
def split_delimiter(x, delimiter=";"):
    """Split delimiter helper function for strings."""
    return x.split(delimiter)


processed_data = (
    data.drop_duplicates(  # .dropna(subset=["accession"])
        subset=["accession", "end"]
    )
    .transform_column("isoforms", split_delimiter)
    .explode("isoforms")
    .transform_column("isoforms", partial(split_delimiter, delimiter=" ("))
    .transform_column("isoforms", lambda x: x[0] if isinstance(x, list) else x)
    .transform_column(
        "isoforms", lambda x: x.strip(" ") if isinstance(x, str) else x
    )
    .drop_duplicates("sequence")
    .join_apply(met_position, "met_position")
)


models_path = here() / "data/ghesquire_2011/models"
protein_models = os.listdir(models_path)


built_models = [f.strip(".pdb") for f in os.listdir(models_path)]


def get_node(G: nx.Graph, pos: int):
    """Get a node by particular residue position."""
    node = [n for n, d in G.nodes(data=True) if d["residue_number"] == pos]
    if len(node) == 1:
        return node[0]
    raise Exception("Node not found!")


def load_model(model: str):
    """Safe loading of model in parallel computation."""
    try:
        m = read_pdb(models_path / f"{model}.pdb")
        return model, m
    except Exception as e:
        print(e)


results = Parallel(n_jobs=-1)(delayed(load_model)(m) for m in built_models)

results = [r for r in results if r is not None]

met_graphs = dict()

for accession, g in tqdm(results):
    row = processed_data.set_index("accession").loc[accession]
    pos = row["met_position"]
    seq = row["sequence"]

    try:
        metnode = get_node(g, pos)
        met_g = extract_neighborhood(g, metnode, 1)
        met_graphs[accession + "-" + seq] = met_g
    except Exception as e:
        print(e)

graph_pickle_dir = here() / "data/ghesquire_2011"
graph_pickle_dir.mkdir(exist_ok=True, parents=True)

with open(graph_pickle_dir / "graphs.pkl", "wb") as f:
    pkl.dump(met_graphs, f)
