"""Functions for handling raw data."""
import janitor
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
#from easy_datastore.functions import download_item
from pyprojroot import here
from scipy.special import logit

from .schemas import ghesquire_processed_schema


def load_ghesquire() -> pd.DataFrame:
    """Load Ghesquire data into memory."""
    try:
        df = pd.read_excel(
            here() / "data" / "ghesquire_2011" / "Ghesquiere2011_Met.xlsx"
        )
    except:
        load_dotenv()
        with open(
            here(project_files=[".here"])
            / "data/ghesquire_2011/datadescriptor.yaml",
            "r+",
        ) as f:
            descriptor = yaml.safe_load(f)

        datastore_id = descriptor["datastore-id"]
        fpath = download_item(datastore_id)
        df = pd.read_excel(fpath, engine="openpyxl")

    # Minimal preprocessing:
    # 1. Remove null values in "sequence" column.
    df = df.remove_empty().dropna(subset=["sequence"])
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

    # 5. Forward fill the accession number
    df["accession"] = df["accession"].fillna(method="ffill")

    # 6. Create the accession-sequence column
    df = df.concatenate_columns(
        ["accession", "sequence"], "accession-sequence"
    )

    # 7. Drop the duplicates, which has the same accession value
    df = df.drop_duplicates(subset=["accession", "end"])
    # use a schema script to check types and properties for a pd.dataframe https://pandera.readthedocs.io/en/stable/
    return ghesquire_processed_schema.validate(df)
