"""Distribution extraction helpers."""

from typing import List

import numpy as np

from clrs._src import dfs_sampling


def extract_probability_matrices(outs_or_preds) -> List[np.ndarray]:
  """Return list of node-parent probability matrices from CLRS outputs/preds."""
  return dfs_sampling.extract_probMatrices(outs_or_preds)

