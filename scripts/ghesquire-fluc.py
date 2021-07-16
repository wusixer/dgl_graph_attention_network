"""Script to generate fluc score dataframes from Ghesquire dataset and Kannan's calculations."""

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

from patch_gnn.fluctuations import fluc2df

load_dotenv()

# currently the file is not on data_store yet
##item = "a1a0d3a7-6723-47f0-86cc-5f42e69435b5" #
##fpath = download_item(item) #

for fluc_file_zip in ["ANM", "NMA"]:
    dest_path = here() / f"data/ghesquire_2011/{fluc_file_zip}.zip"
    # shutil.move(fpath, dest_path)
    z = zipfile.ZipFile(file=dest_path)
    z.extractall()
    # dest_path.unlink() # not delete the folder in the end

    fluc_dir = Path(fluc_file_zip)
    fluc_files = [f for f in os.listdir(fluc_dir) if f.endswith(".fluc")]
    accessions = [f.split("_")[0] for f in fluc_files]

    fluc_dfs = dict()
    for file, accession in tqdm(zip(fluc_files, accessions)):
        fluc_dfs[accession] = fluc2df(
            fluc_dir / file
        )  # , wanted_amn="RESIDUE SASAs") # this part need to write a new function

    with open(here() / f"data/ghesquire_2011/{fluc_file_zip}.pkl", "wb") as f:
        pkl.dump(fluc_dfs, f)
