"""Functions for building UniRep-based models."""
import jax.numpy as np
import pandas as pd
from jax_unirep import get_reps


def unirep_reps(df: pd.DataFrame) -> np.ndarray:
    """Convert the `sequence` column of `df` into UniRep reps."""
    h_avg, _, _ = get_reps(df["sequence"].tolist())
    return h_avg
