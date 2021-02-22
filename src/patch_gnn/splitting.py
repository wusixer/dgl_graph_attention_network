"""Functions to split our dataset into training and testing sets."""

from math import floor
from typing import Tuple

import jax.numpy as np
import pandas as pd
from jax import random


def train_test_split(
    key, df: pd.DataFrame, train_fraction=0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train test split on Ghesquire dataframe."""
    idxs = np.array(df.index)
    perm = random.permutation(key, idxs)
    num_train = floor(train_fraction * len(idxs))
    train_idxs, test_idxs = perm[:num_train], perm[num_train:]
    return df.loc[train_idxs], df.loc[test_idxs]
