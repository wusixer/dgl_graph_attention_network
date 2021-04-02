from collections import defaultdict
from io import StringIO

import pandas as pd

def fluc2df(fname):
    """
    Convert fluctuation calculations to pandas DataFrame.

    :param fname: Path to *.fluc file.
    """
    with open(fname, "r+") as f:
        fluc = f.readlines()

    #sasa_raw = defaultdict(list)
    #sasa_type = ""
    fluc_raw=[]
    for line in fluc:
        fluc_raw.append(line)

    data = "".join(l.replace("\t\t", "\t") for l in fluc_raw) 
    df = pd.read_csv(StringIO(data), sep=",")

    return df