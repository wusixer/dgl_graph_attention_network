"""Script to generate SASA score dataframes from Ghesquire dataset and Kannan's calculations."""

from easy_datastore import download_item
from dotenv import load_dotenv

load_dotenv()

item = "a1a0d3a7-6723-47f0-86cc-5f42e69435b5"
fpath = download_item(item)


import shutil
from pyprojroot import here

dest_path = here() / "data/ghesquire_2011/models.zip"
shutil.move(fpath, dest_path)

import zipfile


z = zipfile.ZipFile(file=dest_path)
z.extractall()
dest_path.unlink()


from patch_gnn.solvent import pops2df


import os
from pathlib import Path
from tqdm.auto import tqdm
from typing import List

pops_dir = Path("pops_out")
pops_files = [f for f in os.listdir(pops_dir) if f.endswith(".out")]
accessions = [f.split("_")[0] for f in pops_files]

sasa_dfs = dict()
for file, accession in tqdm(zip(pops_files, accessions)):
    sasa_dfs[accession] = pops2df(pops_dir / file, wanted_sasa="RESIDUE SASAs")


from pyprojroot import here
import pickle as pkl

with open(here() / "data/ghesquire_2011/sasa.pkl", "wb") as f:
    pkl.dump(sasa_dfs, f)
