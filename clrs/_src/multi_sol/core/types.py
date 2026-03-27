"""Shared datatypes for multi-solution orchestration."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class AlgorithmBatch:
  """Batch payload extracted from CLRS feedback and predictions."""

  adjacency_matrices: np.ndarray
  source_nodes: np.ndarray
  predictions: Dict[str, object]
  outputs: object


@dataclass(frozen=True)
class SamplingResults:
  """Container for sampled trees and associated validity masks."""

  model_trees: List[np.ndarray]
  true_trees: List[np.ndarray]
  model_valid_mask: List[bool]
  true_valid_mask: List[bool]
  model_accuracy: float
  true_accuracy: float

