"""Utilities for one-hot encoding of a protein sequence."""
import numpy as np
import pandas as pd

alphabet = "ACDEFGHIKLMNPQRSTVWY-"
oh_matrix = np.eye(len(alphabet))
oh_mapping = {l: arr for l, arr in zip(alphabet, oh_matrix)}


def encoder(protein_seq: str) -> np.ndarray:
    """
    Used to convert a protein sequence into a numpy
    array. For example, A -> [1, 0, ..., 0], C -> [0, 1, 0, ..., 0]
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
    Used to pad a protein sequence into certain length.
    For example, pad a protein sequence 'length 3' into 'length 5':
    "ACD" -> "ACD--"

    :param seq: the protein sequence needs to be paded
    :param length: the length that the padded protein will be
    :returns: a new protein sequence with '-' padded to the right
    """
    padd_len = length - len(protein_seq)
    return protein_seq + "".join("-" for i in range(padd_len))
