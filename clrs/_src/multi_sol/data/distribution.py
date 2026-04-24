"""Distribution extraction helpers."""

from typing import List

import numpy as np


def extract_probability_matrices(outs_or_preds) -> List[np.ndarray]:
  """Return list of node-parent probability matrices from CLRS outputs/preds."""
  matrices = []
  for value in outs_or_preds:
    if isinstance(value, dict):
      dist_list = value["pi"].data
    else:
      dist_list = value.data
    matrices.extend(dist_list)
  return matrices
