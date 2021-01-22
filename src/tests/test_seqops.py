"""Tests for protein sequence encoding functions."""

import numpy as np
import pandas as pd

from patch_gnn.seqops import alphabet, encoder, padding


def test_encoder():
    """Test for encoder function."""

    protein_encode = encoder("AC")
    a = np.zeros(len(alphabet))
    a[0] = 1
    b = np.zeros(len(alphabet))
    b[1] = 1
    test = np.concatenate((a, b))
    comparison = protein_encode == test
    assert True == comparison.all()


def test_padding():
    """Test for padding function."""

    protein_padded = padding("AC", 5)
    assert protein_padded == "AC---"
