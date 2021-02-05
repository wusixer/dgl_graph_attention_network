"""Functions for handling raw data."""
import janitor
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from easy_datastore.functions import download_item
from pyprojroot import here
from scipy.special import logit

from .schemas import ghesquire_processed_schema


def load_ghesquire() -> pd.DataFrame:
    """Load Ghesquire data into memory."""
    load_dotenv()
    with open(
        here(project_files=[".here"])
        / "data/ghesquire_2011/datadescriptor.yaml",
        "r+",
    ) as f:
        descriptor = yaml.safe_load(f)

    datastore_id = descriptor["datastore-id"]
    fpath = download_item(datastore_id)

    df = pd.read_excel(fpath, engine="openpyxl").remove_empty()

    # Minimal preprocessing:
    # 1. Remove null values in "sequence" column.
    df = df.dropna(subset=["sequence"])
    # 2. Ensure that the output column ("%ox_fwd") is logit-transformed.

    def tfm(x):
        """Logit transform for %ox_fwd/rev function."""
        return logit(np.clip(x / 100, 0.01, 0.99))

    # outputs = logit(np.clip(df["%ox_fwd"] / 100, 0.01, 0.99))
    df = df.assign(**{"ox_fwd_logit": tfm(df["%ox_fwd"])})

    # 3. Ensure that %ox_rev is numeric and transformed properly.
    df = df.replace({"%ox_rev": {"": np.nan, " ": np.nan}}).change_type(
        "%ox_rev", float
    )
    df = df.assign(**{"ox_rev_logit": tfm(df["%ox_rev"])})

    # 4. Rename columns properly
    df = df.rename(columns={"treshhold": "threshold"})
    return ghesquire_processed_schema.validate(df)
