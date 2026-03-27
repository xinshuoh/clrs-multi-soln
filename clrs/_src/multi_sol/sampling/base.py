"""Shared sampling helpers."""

from typing import List

import numpy as np

from clrs._src.multi_sol.data.distribution import extract_probability_matrices


def normalize_rows(prob_matrix: np.ndarray) -> np.ndarray:
  """Row-normalize probability matrix while preserving zero rows."""
  out = prob_matrix.astype(np.float64, copy=True)
  row_sums = out.sum(axis=1)
  nonzero = row_sums > 0
  out[nonzero] = out[nonzero] / row_sums[nonzero][:, None]
  return out


def extract_prob_matrices(outs_or_preds) -> List[np.ndarray]:
  return extract_probability_matrices(outs_or_preds)

