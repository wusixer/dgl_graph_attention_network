from collections import defaultdict
from io import StringIO

import pandas as pd


def pops2df(fname, wanted_sasa="RESIDUE SASAs"):
    """
    Convert pops calculations to pandas DataFrame.

    Solvent calculations come to us from the [POPS program][pops].

    [pops]: https://github.com/Fraternalilab/POPSlegacy

    :param fname: Path to pops.out file.
    :param sasa_type: One of "atom",
        "residue", "chain", "molecule"
    """
    with open(fname, "r+") as f:
        sasa = f.readlines()

    sasa_raw = defaultdict(list)
    sasa_type = ""
    for line in sasa:
        if "===" in line:
            sasa_type = line.split("===")[1].strip(" ")
        sasa_raw[sasa_type].append(line)

    data = "".join(l.replace("\t\t", "\t") for l in sasa_raw[wanted_sasa][1:])
    df = pd.read_csv(StringIO(data), sep="\t")
    df["ResidNe"] = df["ResidNe"].str.strip(" ")
    return df
