"""Utilities for one-hot encoding of a protein sequence."""
import numpy as np
import pandas as pd

alphabet = "ACDEFGHIKLMNPQRSTVWY-"
oh_matrix = np.eye(len(alphabet))
oh_mapping = {l: arr for l, arr in zip(alphabet, oh_matrix)}


def encoder(protein_seq: str) -> np.ndarray:
    """
    Convert a protein sequence into a numpy array.

    For example, A -> [1, 0, ..., 0], C -> [0, 1, 0, ..., 0]
    AC -> [1, 0, ..., 0, 0, 1, ..., 0]

    :param seq: the protein sequence needs to be encoded
    :returns: a numpy array that encodes the protein sequence into an integer array.
    """
    encoded_array = list()
    for char in protein_seq:
        encoded_array.append(oh_mapping[char])
    return np.concatenate(encoded_array)


def padding(protein_seq: str, length: int) -> str:
    """
    Pad a protein sequence into certain length.

    For example, pad a protein sequence 'length 3' into 'length 5':
    "ACD" -> "ACD--"

    :param seq: the protein sequence needs to be paded
    :param length: the length that the padded protein will be
    :returns: a new protein sequence with '-' padded to the right
    """
    padd_len = length - len(protein_seq)
    return protein_seq + "".join("-" for i in range(padd_len))


def one_hot(df: pd.DataFrame, padding_length: int) -> np.ndarray:
    """
    Convert the `sequence` column of `df` to a one-hot array.
    
    :param df: a pandas dataframe with a column called "sequence" for one hot conversion
    :param padding_length: same as length in `padding`, the length that the padded protein will be
    """
    in_max_len = max(len(seq) for seq in df["sequence"])
    if padding_length < in_max_len:
        print(f"the input padding_length should be longer or equal to the max length of the inputs, resetting padding_length from {padding_length}  to max sequence length {in_max_len}")
        padding_length = in_max_len
    padded_seqs = list(padding(seq, padding_length) for seq in df["sequence"])
    encodings = [encoder(seq) for seq in padded_seqs]
    return np.vstack(encodings)
