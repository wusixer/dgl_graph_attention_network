#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""Module for splitting data in cross validation"""

import math
import random

from torch.utils.data import Subset


def k_fold_split(dataset, k, shuffle=True):
    """
    Parameters
    -----------
    dataset
        A PatchGNNDataset dataset object
    k: int
        The number of folds.
    shuffle: bool
        Whether to shuffle the dataset before performing a k-fold split.

    Returns
    --------
    list of length k
        Each element is a tuple (train_set, val_set) corresponding to a fold.
    """
    assert (
        k >= 2
    ), "Expect the number of folds to be no smaller than 2, got {:d}".format(k)
    all_folds = []
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    frac_per_part = 1.0 / k
    data_size = len(dataset)
    for i in range(k):
        val_start = math.floor(data_size * i * frac_per_part)
        val_end = math.floor(data_size * (i + 1) * frac_per_part)
        val_indices = indices[val_start:val_end]
        val_subset = Subset(dataset, val_indices)
        train_indices = indices[:val_start] + indices[val_end:]
        train_subset = Subset(dataset, train_indices)
        all_folds.append((train_subset, val_subset))
    return all_folds
