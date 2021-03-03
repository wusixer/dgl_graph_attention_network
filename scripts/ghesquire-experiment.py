"""Script to reproducibly generate experimental requests for measurement.

Background: Kannan Sankar and Yang Yang (NBC) would like to generate
experimental measurements for a subset of the Ghesquire measurements.
This set of measurements should only cover
those entries for which we were able to:

1. Generate a PDB file for, and
2. Identify the methionine inside the structure that was oxidized,

This script generates the measurements based on the output of previous scripts.
To execute this script, you should use the Makefile:

```bash
make mlsubset
```
"""

import pickle as pkl

from pyprojroot import here

from patch_gnn.data import load_ghesquire
from patch_gnn.graph import met_position

graph_pickle_path = here() / "data/ghesquire_2011/graphs.pkl"

with open(graph_pickle_path, "rb") as f:
    graphs = pkl.load(f)

data = load_ghesquire()


filtered = (
    data.query("`accession-sequence` in @graphs.keys()")
    .query("ox_fwd_logit < 2.0")
    .join_apply(met_position, "met_position")
)


(
    filtered.sort_values("%ox_fwd", ascending=False).to_csv(
        here() / "data/ghesquire_2011/machine-learning-subset.csv"
    )
)
