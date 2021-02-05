"""Script to extract Ghesquire 2011 protein sequences into a FASTA file."""
import os
from functools import partial, singledispatch

import janitor
import pandas as pd
import pandas_flavor as pf
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from dotenv import load_dotenv
from psycopg2 import connect
from pyprojroot import here

from patch_gnn.data import load_ghesquire

data = load_ghesquire()


@singledispatch
def split_delimiter(x, delimiter=";"):
    """Split delimiter helper function."""
    raise NotImplementedError("Unsupported type!")


@split_delimiter.register(float)
def _split_delimiter(x, delimiter=";"):
    """Split delimiter helper function for floats."""
    return x


@split_delimiter.register(str)
def _split_delimiter(x, delimiter=";"):
    """Split delimiter helper function for strings."""
    return x.split(delimiter)


processed_data = (
    data.dropna(subset=["accession"])
    .transform_column("isoforms", split_delimiter)
    .explode("isoforms")
    .transform_column("isoforms", partial(split_delimiter, delimiter=" ("))
    .transform_column("isoforms", lambda x: x[0] if isinstance(x, list) else x)
    .transform_column(
        "isoforms", lambda x: x.strip(" ") if isinstance(x, str) else x
    )
)


# ## download sequences from HitHub
#
# We have an internal mirror of UniProt, hosted on HitHub.
# (Once again, CBTDS does all the right things!)
# We can query HH for UniProt sequences that way.
#


load_dotenv()

con = connect(dsn=os.getenv("HH_CONNECTION_STRING"))


wanted_accessions = processed_data["accession"].dropna().tolist()
wanted_accessions.extend(processed_data["isoforms"].dropna().tolist())


accession_data = pd.read_sql(
    f"select * from uniprot.ref_proteome where uniprot_accn in {tuple(wanted_accessions)}",
    con=con,
)


@pf.register_dataframe_method
def to_fasta(df, identifier_column_name, sequence_column_name, filename):
    """Write dataframe to FASTA file."""
    seq_records = []
    for r, d in df.iterrows():
        seq = Seq(d[sequence_column_name])
        seq_record = SeqRecord(
            seq, id=d[identifier_column_name], description="", name=""
        )
        seq_records.append(seq_record)
    SeqIO.write(seq_records, filename, format="fasta")


accession_data.to_fasta(
    identifier_column_name="uniprot_accn",
    sequence_column_name="seq",
    filename=here() / "data/ghesquire_2011/protein_sequences.fasta",
)
