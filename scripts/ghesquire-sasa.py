"""Script to generate SASA score dataframes from Ghesquire dataset and Kannan's calculations."""

import os
import pickle as pkl
import shutil
import zipfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from easy_datastore import download_item
from pyprojroot import here
from tqdm.auto import tqdm

from patch_gnn.solvent import pops2df

load_dotenv()

#item = "a1a0d3a7-6723-47f0-86cc-5f42e69435b5" #
#fpath = download_item(item) #


dest_path = here() / "data/ghesquire_2011/models.zip" # this file is a table taht has all the aa and its correpsonding sasa values
#shutil.move(fpath, dest_path)


z = zipfile.ZipFile(file=dest_path)
z.extractall()
dest_path.unlink()


pops_dir = Path("pops_out")
pops_files = [f for f in os.listdir(pops_dir) if f.endswith(".out")]
accessions = [f.split("_")[0] for f in pops_files]

sasa_dfs = dict()
for file, accession in tqdm(zip(pops_files, accessions)):
    sasa_dfs[accession] = pops2df(pops_dir / file, wanted_sasa="RESIDUE SASAs")


with open(here() / "data/ghesquire_2011/sasa.pkl", "wb") as f:
    pkl.dump(sasa_dfs, f)
